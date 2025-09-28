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
            | ((df["hour"] == 15) & (df["minute"] <= 0))
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

        half_day_groups = df.groupby(["date", "half_day"]).agg(agg_dict).reset_index()
        half_day_groups["date"] = pd.to_datetime(half_day_groups["date"])
        half_day_groups = half_day_groups.sort_values(["date", "half_day"]).reset_index(
            drop=True
        )
        half_day_groups["is_am"] = half_day_groups["half_day"].astype(bool)

        # 清洗缺失值
        numeric_cols = [
            col
            for col in half_day_groups.columns
            if col not in ["date", "half_day", "is_am"]
        ]
        half_day_groups[numeric_cols] = half_day_groups[numeric_cols].bfill().ffill()
        for col in numeric_cols:
            half_day_groups[col] = half_day_groups[col].fillna(
                half_day_groups[col].mean()
            )

        return half_day_groups

    def preprocess_half_day_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特征工程，聚焦 high/low 极值预测
        """
        df = df.copy()

        # 基础收益率
        df["returns"] = df["close"].pct_change()

        # 移动平均比率（短中期，添加40期长视角）
        for period in [5, 20, 40]:
            ma = df["close"].rolling(window=period, min_periods=1).mean()
            df[f"close_ma_{period}_ratio"] = df["close"] / (ma + 1e-8)

        df["ma_diff_5_20"] = (
            df["close"].rolling(5, min_periods=1).mean()
            - df["close"].rolling(20, min_periods=1).mean()
        ) / (df["close"].rolling(20, min_periods=1).mean() + 1e-8)
        df["ma_diff_20_40"] = (
            df["close"].rolling(20, min_periods=1).mean()
            - df["close"].rolling(40, min_periods=1).mean()
        ) / (df["close"].rolling(40, min_periods=1).mean() + 1e-8)

        # MACD divergence
        try:
            macd, macd_signal, _ = talib.MACD(
                df["close"].values, fastperiod=12, slowperiod=26, signalperiod=9
            )
            df["macd_divergence"] = macd - macd_signal
        except Exception as e:
            logger.warning(f"MACD failed: {e}")
            df["macd_divergence"] = np.nan

        # BBANDS（含squeeze）
        try:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                df["close"].values, timeperiod=20, nbdevup=2, nbdevdn=2
            )
            df["bb_width"] = (bb_upper - bb_lower) / (bb_middle + 1e-8)
            df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-8)
            df["bb_squeeze"] = (
                df["bb_width"] < df["bb_width"].rolling(20, min_periods=1).quantile(0.1)
            ).astype(int)
        except Exception as e:
            logger.warning(f"BBANDS failed: {e}")
            df[["bb_width", "bb_position", "bb_squeeze"]] = np.nan

        # 波动率（短中长，增强突破/扩张）
        for period in [5, 20, 40]:
            df[f"return_vol_{period}"] = (
                df["returns"].rolling(window=period, min_periods=1).std()
            )
        df["volatility_breakout"] = df["return_vol_5"] / (df["return_vol_40"] + 1e-8)
        df["volatility_expansion"] = (df["volatility_breakout"] > 1.5).astype(int)

        # 成交量
        df["volume_ma_20"] = df["volume"].rolling(window=20, min_periods=1).mean()
        df["volume_ratio"] = df["volume"] / (df["volume_ma_20"] + 1e-8)

        # 动量（短中期）
        for period in [5, 20]:
            df[f"momentum_{period}"] = df["close"] / (df["close"].shift(period) + 1e-8)

        # 支撑/阻力（中长期）
        for period in [20, 40]:
            high_period = df["high"].rolling(window=period, min_periods=1).max()
            low_period = df["low"].rolling(window=period, min_periods=1).min()
            df[f"resistance_distance_{period}"] = (high_period - df["close"]) / (
                df["close"] + 1e-8
            )
            df[f"support_distance_{period}"] = (df["close"] - low_period) / (
                df["close"] + 1e-8
            )

        # ATR & ADX（波动+趋势强度）
        try:
            df["atr"] = talib.ATR(
                df["high"].values, df["low"].values, df["close"].values, timeperiod=14
            )
            df["atr_ratio"] = df["atr"] / (df["close"] + 1e-8)
            df["adx"] = talib.ADX(
                df["high"].values, df["low"].values, df["close"].values, timeperiod=14
            )
        except Exception as e:
            logger.warning(f"ATR/ADX failed: {e}")
            df[["atr", "atr_ratio", "adx"]] = np.nan

        # 极值专用特征（增强连续/背离）
        df["price_acceleration"] = df["returns"] - df["returns"].shift(1)
        df["high_change"] = (df["high"] - df["close"].shift(1)) / (
            df["close"].shift(1) + 1e-8
        )
        df["low_change"] = (df["low"] - df["close"].shift(1)) / (
            df["close"].shift(1) + 1e-8
        )
        df["extreme_range"] = (df["high"] - df["low"]) / (df["close"] + 1e-8)
        df["normalized_high"] = (df["high"] - df["close"]) / (df["atr"] + 1e-8)
        df["normalized_low"] = (df["close"] - df["low"]) / (df["atr"] + 1e-8)
        df["price_position_20"] = (
            df["close"] - df["low"].rolling(20, min_periods=1).min()
        ) / (
            df["high"].rolling(20, min_periods=1).max()
            - df["low"].rolling(20, min_periods=1).min()
            + 1e-8
        )
        df["consecutive_extreme_up"] = (df["returns"] > 0.02).rolling(3).sum()
        df["consecutive_extreme_down"] = (df["returns"] < -0.02).rolling(3).sum()
        df["price_volume_correlation"] = (
            df["returns"].rolling(10).corr(df["volume"].pct_change())
        )
        df["price_volume_divergence"] = (df["price_volume_correlation"] < -0.5).astype(
            int
        )

        # 最终填充（bfill/ffill + mean，避免0偏置）
        df = df.bfill().ffill()
        numeric_cols = [
            col for col in df.columns if df[col].dtype in ["float64", "int64"]
        ]
        for col in numeric_cols:
            if col not in ["date", "half_day", "is_am"]:
                df[col] = df[col].fillna(df[col].mean())

        return df

    def normalize_features(
        self, df: pd.DataFrame, fit_scaler: bool = True, scaler=None
    ):
        feature_cols = [
            col for col in df.columns if col not in ["date", "is_am", "half_day"]
        ]
        df[feature_cols] = df[feature_cols].fillna(
            df[feature_cols].mean()
        )  # 用mean填充，提升稳定性

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
