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
            | ((df["hour"] == 15) & (df["minute"] == 0))
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

        # 清洗
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
                half_day_groups[col].median()
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
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # 移动平均（现在 rolling(5) = 5个半日 ≈ 2.5天）
        for period in [5, 20, 60]:
            df[f"ma_{period}"] = (
                df["close"].rolling(window=period, min_periods=1).mean()
            )
            df[f"close_ma_{period}_ratio"] = df["close"] / (df[f"ma_{period}"] + 1e-8)

        df["ma_diff_5_20"] = (df["ma_5"] - df["ma_20"]) / (df["ma_20"] + 1e-8)
        df["ma_diff_20_60"] = (df["ma_20"] - df["ma_60"]) / (df["ma_60"] + 1e-8)

        # 技术指标（在半日 close 上计算）
        try:
            macd, macd_signal, macd_hist = talib.MACD(df["close"].values)
            df["macd"] = macd
            df["macd_signal"] = macd_signal
            df["macd_hist"] = macd_hist
            df["macd_divergence"] = df["macd"] - df["macd_signal"]
        except Exception as e:
            logger.warning(f"MACD failed: {e}")
            df[["macd", "macd_signal", "macd_hist", "macd_divergence"]] = np.nan

        try:
            df["rsi"] = talib.RSI(df["close"].values, timeperiod=14)
            df["rsi_extreme"] = np.where(
                df["rsi"] > 80,
                df["rsi"] - 80,
                np.where(df["rsi"] < 20, df["rsi"] - 20, 0),
            )
        except Exception as e:
            logger.warning(f"RSI failed: {e}")
            df[["rsi", "rsi_extreme"]] = np.nan

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

        # 波动率、成交量、动量等（全部基于半日）
        for period in [5, 20, 60]:
            df[f"return_vol_{period}"] = (
                df["returns"].rolling(window=period, min_periods=1).std()
            )
            df[f"high_low_ratio_{period}"] = (
                ((df["high"] - df["low"]) / (df["close"] + 1e-8))
                .rolling(window=period, min_periods=1)
                .mean()
            )

        df["volume_ma_20"] = df["volume"].rolling(window=20, min_periods=1).mean()
        df["volume_ratio"] = df["volume"] / (df["volume_ma_20"] + 1e-8)
        df["volume_change"] = df["volume"].pct_change()

        for period in [5, 20]:
            df[f"momentum_{period}"] = df["close"] / (df["close"].shift(period) + 1e-8)

        for period in [20, 60]:
            df[f"high_{period}"] = (
                df["high"].rolling(window=period, min_periods=1).max()
            )
            df[f"low_{period}"] = df["low"].rolling(window=period, min_periods=1).min()
            df[f"resistance_distance_{period}"] = (
                df[f"high_{period}"] - df["close"]
            ) / (df["close"] + 1e-8)
            df[f"support_distance_{period}"] = (df["close"] - df[f"low_{period}"]) / (
                df["close"] + 1e-8
            )

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

        # 极值专用特征（全部基于半日）
        df["price_acceleration"] = df["returns"] - df["returns"].shift(1)
        df["acceleration_change"] = df["price_acceleration"] - df[
            "price_acceleration"
        ].shift(1)
        df["high_change"] = (df["high"] - df["close"].shift(1)) / (
            df["close"].shift(1) + 1e-8
        )
        df["low_change"] = (df["low"] - df["close"].shift(1)) / (
            df["close"].shift(1) + 1e-8
        )
        df["extreme_range"] = (df["high"] - df["low"]) / (df["close"] + 1e-8)
        df["volatility_breakout"] = df["return_vol_5"] / (df["return_vol_60"] + 1e-8)
        df["volatility_expansion"] = (df["volatility_breakout"] > 1.5).astype(int)
        df["price_volume_correlation"] = (
            df["returns"].rolling(10).corr(df["volume_change"])
        )
        df["price_volume_divergence"] = (df["price_volume_correlation"] < -0.5).astype(
            int
        )
        df["rsi_extreme_high"] = (df["rsi"] > 80).astype(int)
        df["rsi_extreme_low"] = (df["rsi"] < 20).astype(int)
        df["bb_breakout_upper"] = (df["high"] > df["bb_upper"]).astype(int)
        df["bb_breakout_lower"] = (df["low"] < df["bb_lower"]).astype(int)
        df["bb_squeeze"] = (
            df["bb_width"] < df["bb_width"].rolling(20).quantile(0.1)
        ).astype(int)
        df["consecutive_extreme_up"] = (df["returns"] > 0.02).rolling(3).sum()
        df["consecutive_extreme_down"] = (df["returns"] < -0.02).rolling(3).sum()
        df["normalized_high"] = (df["high"] - df["close"]) / (df["atr"] + 1e-8)
        df["normalized_low"] = (df["close"] - df["low"]) / (df["atr"] + 1e-8)
        df["price_position_20"] = (df["close"] - df["low_20"]) / (
            df["high_20"] - df["low_20"] + 1e-8
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
