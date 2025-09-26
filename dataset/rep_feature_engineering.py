import os
import sys
from dotenv import load_dotenv
from termcolor import colored
import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import StandardScaler

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)

from z_utils.logging_config import get_logger

load_dotenv()
logger = get_logger(__name__)


class StockFeatureEngineer:
    """股票特征工程类"""

    def __init__(self):
        self.feature_names = []

    def aggregate_half_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """聚合为半日数据"""
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"] = df["datetime"].dt.date
        df["hour"] = df["datetime"].dt.hour
        df["minute"] = df["datetime"].dt.minute

        df["is_am"] = (
            ((df["hour"] == 9) & (df["minute"] >= 30))
            | (df["hour"] == 10)
            | ((df["hour"] == 11) & (df["minute"] <= 30))
        )
        df["is_pm"] = (
            (df["hour"] == 13)
            | (df["hour"] == 14)
            | ((df["hour"] == 15) & (df["minute"] <= 0))  # 到15:00收盘
        )

        df = df[df["is_am"] | df["is_pm"]].copy()
        df["half_day"] = df["is_am"].astype(int)

        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "amount": "sum",
        }

        # 排除分组键和布尔列
        exclude_cols = {
            "date",
            "half_day",
            "is_am",
            "is_pm",
            "datetime",
            "hour",
            "minute",
        }
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in agg_dict and col not in exclude_cols:
                agg_dict[col] = "mean"

        half_day_groups = df.groupby(["date", "half_day"]).agg(agg_dict)

        # 直接 reset 并确保不冲突
        half_day_groups = half_day_groups.reset_index()

        half_day_groups["date"] = pd.to_datetime(half_day_groups["date"])
        half_day_groups = half_day_groups.sort_values(["date", "half_day"]).reset_index(
            drop=True
        )
        half_day_groups["is_am"] = half_day_groups["half_day"].astype(bool)

        # 清洗：bfill/ffill后用mean填充
        numeric_cols_agg = [
            col
            for col in half_day_groups.columns
            if col not in ["date", "half_day", "is_am"]
        ]
        half_day_groups[numeric_cols_agg] = (
            half_day_groups[numeric_cols_agg].bfill().ffill()
        )
        for col in numeric_cols_agg:
            half_day_groups[col] = half_day_groups[col].fillna(
                half_day_groups[col].mean()
            )

        return half_day_groups

    def preprocess_half_day_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        在半日级别数据上进行特征工程
        输入 df 必须包含: date, is_am, open, high, low, close, volume, amount
        """
        df = df.copy()

        # 基础收益率（基于半日 close）
        df["returns"] = df["close"].pct_change()

        # 移动平均（只保留比率，periods=[5,20,40]）
        for period in [5, 20, 40]:
            ma_period = df["close"].rolling(window=period, min_periods=1).mean()
            df[f"close_ma_{period}_ratio"] = df["close"] / (ma_period + 1e-8)

        df["ma_diff_5_20"] = (
            df["close"].rolling(5, min_periods=1).mean()
            - df["close"].rolling(20, min_periods=1).mean()
        ) / (df["close"].rolling(20, min_periods=1).mean() + 1e-8)
        df["ma_diff_20_40"] = (
            df["close"].rolling(20, min_periods=1).mean()
            - df["close"].rolling(40, min_periods=1).mean()
        ) / (df["close"].rolling(40, min_periods=1).mean() + 1e-8)

        # MACD（保留macd和divergence）
        try:
            macd, macd_signal, _ = talib.MACD(df["close"].values)
            df["macd"] = macd
            df["macd_divergence"] = macd - macd_signal
        except Exception as e:
            logger.warning(f"MACD failed: {e}")
            df[["macd", "macd_divergence"]] = np.nan

        # RSI
        try:
            df["rsi"] = talib.RSI(df["close"].values, timeperiod=14)
        except Exception as e:
            logger.warning(f"RSI failed: {e}")
            df["rsi"] = np.nan

        # BBANDS
        try:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                df["close"].values, timeperiod=20, nbdevup=2, nbdevdn=2
            )
            df["bb_upper"] = bb_upper
            df["bb_lower"] = bb_lower
            df["bb_middle"] = bb_middle
            df["bb_width"] = (bb_upper - bb_lower) / (bb_middle + 1e-8)
            df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        except Exception as e:
            logger.warning(f"BBANDS failed: {e}")
            df[["bb_upper", "bb_lower", "bb_middle", "bb_width", "bb_position"]] = (
                np.nan
            )

        # 波动率（periods=[5,20,40]）
        for period in [5, 20, 40]:
            df[f"return_vol_{period}"] = (
                df["returns"].rolling(window=period, min_periods=1).std()
            )

        # 成交量
        df["volume_ma_20"] = df["volume"].rolling(window=20, min_periods=1).mean()
        df["volume_ratio"] = df["volume"] / (df["volume_ma_20"] + 1e-8)

        # 动量
        for period in [5, 20]:
            df[f"momentum_{period}"] = df["close"] / (df["close"].shift(period) + 1e-8)

        # 支撑/阻力距离（periods=[20,40]）
        for period in [20, 40]:
            high_period = df["high"].rolling(window=period, min_periods=1).max()
            low_period = df["low"].rolling(window=period, min_periods=1).min()
            df[f"resistance_distance_{period}"] = (high_period - df["close"]) / (
                df["close"] + 1e-8
            )
            df[f"support_distance_{period}"] = (df["close"] - low_period) / (
                df["close"] + 1e-8
            )

        # ADX/ATR
        try:
            df["adx"] = talib.ADX(
                df["high"].values, df["low"].values, df["close"].values, timeperiod=14
            )
            df["atr"] = talib.ATR(
                df["high"].values, df["low"].values, df["close"].values, timeperiod=14
            )
            df["atr_ratio"] = df["atr"] / (df["close"] + 1e-8)
        except Exception as e:
            logger.warning(f"ADX/ATR failed: {e}")
            df[["adx", "atr", "atr_ratio"]] = np.nan

        # 补充指标：STOCH
        try:
            slowk, slowd = talib.STOCH(
                df["high"].values,
                df["low"].values,
                df["close"].values,
                fastk_period=5,
                slowk_period=3,
                slowd_period=3,
            )
            df["stoch_k"] = slowk
            df["stoch_d"] = slowd
            df["stoch_overbought"] = (df["stoch_k"] > 80).astype(int)
            df["stoch_oversold"] = (df["stoch_k"] < 20).astype(int)
        except Exception as e:
            logger.warning(f"STOCH failed: {e}")
            df[["stoch_k", "stoch_d", "stoch_overbought", "stoch_oversold"]] = np.nan

        # CCI
        try:
            df["cci"] = talib.CCI(
                df["high"].values, df["low"].values, df["close"].values, timeperiod=14
            )
            df["cci_extreme"] = np.where(
                df["cci"] > 100,
                df["cci"] - 100,
                np.where(df["cci"] < -100, df["cci"] + 100, 0),
            )
        except Exception as e:
            logger.warning(f"CCI failed: {e}")
            df[["cci", "cci_extreme"]] = np.nan

        # Williams %R
        try:
            df["williams_r"] = talib.WILLR(
                df["high"].values, df["low"].values, df["close"].values, timeperiod=14
            )
            df["williams_extreme"] = np.where(
                df["williams_r"] < -80,
                df["williams_r"] + 100,
                np.where(df["williams_r"] > -20, df["williams_r"] + 20, 0),
            )
        except Exception as e:
            logger.warning(f"Williams %R failed: {e}")
            df[["williams_r", "williams_extreme"]] = np.nan

        # OBV
        try:
            df["obv"] = talib.OBV(df["close"].values, df["volume"].values)
            df["obv_change"] = df["obv"].pct_change()
        except Exception as e:
            logger.warning(f"OBV failed: {e}")
            df[["obv", "obv_change"]] = np.nan

        # MFI
        try:
            df["mfi"] = talib.MFI(
                df["high"].values,
                df["low"].values,
                df["close"].values,
                df["volume"].values,
                timeperiod=14,
            )
            df["mfi_extreme"] = np.where(
                df["mfi"] > 80,
                df["mfi"] - 80,
                np.where(df["mfi"] < 20, df["mfi"] - 20, 0),
            )
        except Exception as e:
            logger.warning(f"MFI failed: {e}")
            df[["mfi", "mfi_extreme"]] = np.nan

        # 极值专用特征
        df["price_acceleration"] = df["returns"] - df["returns"].shift(1)
        df["high_change"] = (df["high"] - df["close"].shift(1)) / (
            df["close"].shift(1) + 1e-8
        )
        df["low_change"] = (df["low"] - df["close"].shift(1)) / (
            df["close"].shift(1) + 1e-8
        )
        df["extreme_range"] = (df["high"] - df["low"]) / (df["close"] + 1e-8)
        df["volatility_breakout"] = df["return_vol_5"] / (df["return_vol_40"] + 1e-8)
        df["volatility_expansion"] = (df["volatility_breakout"] > 1.5).astype(int)
        df["price_volume_correlation"] = (
            df["returns"].rolling(10).corr(df["volume"].pct_change())
        )
        df["price_volume_divergence"] = (df["price_volume_correlation"] < -0.5).astype(
            int
        )
        df["bb_squeeze"] = (
            df["bb_width"] < df["bb_width"].rolling(20).quantile(0.1)
        ).astype(int)
        df["consecutive_extreme_up"] = (df["returns"] > 0.02).rolling(3).sum()
        df["consecutive_extreme_down"] = (df["returns"] < -0.02).rolling(3).sum()
        df["normalized_high"] = (df["high"] - df["close"]) / (df["atr"] + 1e-8)
        df["normalized_low"] = (df["close"] - df["low"]) / (df["atr"] + 1e-8)
        df["price_position_20"] = (
            df["close"] - df["low"].rolling(20, min_periods=1).min()
        ) / (
            df["high"].rolling(20, min_periods=1).max()
            - df["low"].rolling(20, min_periods=1).min()
            + 1e-8
        )

        df = df.bfill().ffill()
        return df

    def normalize_features(
        self, df: pd.DataFrame, fit_scaler: bool = True, scaler=None
    ):
        """
        标准化特征列 排除 date, is_am, half_day
        """
        feature_cols = [
            col for col in df.columns if col not in ["date", "is_am", "half_day"]
        ]
        # 填充NaN为0，避免Scaler报错
        df[feature_cols] = df[feature_cols].fillna(0)

        if fit_scaler:
            scaler = StandardScaler()
            normalized_values = scaler.fit_transform(df[feature_cols])
        else:
            if scaler is None:
                raise ValueError("scaler must be provided when fit_scaler=False")
            normalized_values = scaler.transform(df[feature_cols])

        normalized_df = df.copy()
        normalized_df[feature_cols] = normalized_values

        return normalized_df, scaler


if __name__ == "__main__":
    """
    uv run dataset/rep_feature_engineering.py
    """
    file_path = "no_git_oic/SH.603678_2025.csv"
    df = pd.read_csv(file_path)
    logger.info(colored("\n%s", "green"), df.head(3))

    feature_engineer = StockFeatureEngineer()

    df_agg = feature_engineer.aggregate_half_days(df)  # 先聚合
    df_feature = feature_engineer.preprocess_half_day_features(df_agg)  # 特征工程
    logger.info(colored("\n%s", "green"), df_feature.head())
