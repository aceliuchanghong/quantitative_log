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

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标特征"""
        df = df.copy()

        # 基础收益率
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # 核心移动平均线 - 避免过度重叠
        for period in [5, 20, 60]:  # 选择有代表性的周期
            df[f"ma_{period}"] = df["close"].rolling(window=period).mean()
            df[f"close_ma_{period}_ratio"] = df["close"] / df[f"ma_{period}"]

        # MA差异 - 不同时间尺度的对比
        df["ma_diff_5_20"] = (df["ma_5"] - df["ma_20"]) / df["ma_20"]
        df["ma_diff_20_60"] = (df["ma_20"] - df["ma_60"]) / df["ma_60"]

        # MACD系统 - 保留核心指标
        try:
            macd, macd_signal, macd_hist = talib.MACD(df["close"])
            df["macd"] = macd
            df["macd_signal"] = macd_signal  # 这一行是关键
            df["macd_hist"] = macd_hist
            df["macd_divergence"] = df["macd"] - df["macd_signal"]
        except Exception as e:
            logger.warning(f"MACD calculation failed: {e}")
            # 如果MACD计算失败，创建空列或使用默认值
            df["macd"] = np.nan
            df["macd_signal"] = np.nan
            df["macd_hist"] = np.nan
            df["macd_divergence"] = np.nan

        # RSI - 选择代表性周期
        try:
            df["rsi"] = talib.RSI(df["close"], timeperiod=14)
            df["rsi_extreme"] = np.where(
                df["rsi"] > 80,
                df["rsi"] - 80,
                np.where(df["rsi"] < 20, df["rsi"] - 20, 0),
            )
        except Exception as e:
            logger.warning(f"RSI calculation failed: {e}")
            df["rsi"] = np.nan
            df["rsi_extreme"] = np.nan

        # 布林带系统 - 保留关键指标
        try:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                df["close"], timeperiod=20, nbdevup=2, nbdevdn=2
            )
            df["bb_upper"] = bb_upper
            df["bb_lower"] = bb_lower
            df["bb_middle"] = bb_middle
            df["bb_width"] = (bb_upper - bb_lower) / bb_middle
            df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        except Exception as e:
            logger.warning(f"BBANDS calculation failed: {e}")
            df["bb_upper"] = np.nan
            df["bb_lower"] = np.nan
            df["bb_middle"] = np.nan
            df["bb_width"] = np.nan
            df["bb_position"] = np.nan

        # 随机指标 - 保留K线
        try:
            k, d = talib.STOCH(
                df["high"],
                df["low"],
                df["close"],
                fastk_period=9,
                slowk_period=3,
                slowd_period=3,
            )
            df["stoch_k"] = k
            df["stoch_extreme"] = np.where(
                df["stoch_k"] > 80,
                df["stoch_k"] - 80,
                np.where(df["stoch_k"] < 20, df["stoch_k"] - 20, 0),
            )
        except Exception as e:
            logger.warning(f"STOCH calculation failed: {e}")
            df["stoch_k"] = np.nan
            df["stoch_extreme"] = np.nan

        # 波动率指标 - 选择代表性周期
        for period in [5, 20, 60]:  # 避免过多相似指标
            df[f"return_vol_{period}"] = df["returns"].rolling(window=period).std()
            df[f"high_low_ratio_{period}"] = (
                ((df["high"] - df["low"]) / df["close"]).rolling(window=period).mean()
            )

        # 成交量分析 - 核心指标
        df["volume_ma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / (df["volume_ma_20"] + 1e-8)
        df["volume_change"] = df["volume"].pct_change()

        # 动量指标 - 选择代表性周期
        for period in [5, 20]:  # 避免过多相似指标
            df[f"momentum_{period}"] = df["close"] / df["close"].shift(period)

        # 支撑阻力位 - 选择代表性周期
        for period in [20, 60]:  # 避免过多相似指标
            df[f"high_{period}"] = df["high"].rolling(window=period).max()
            df[f"low_{period}"] = df["low"].rolling(window=period).min()
            df[f"resistance_distance_{period}"] = (
                df[f"high_{period}"] - df["close"]
            ) / df["close"]
            df[f"support_distance_{period}"] = (df["close"] - df[f"low_{period}"]) / df[
                "close"
            ]

        # 趋势指标 - 保留核心
        try:
            df["adx"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)
            df["atr"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
            df["atr_ratio"] = df["atr"] / df["close"]
        except Exception as e:
            logger.warning(f"ADX/ATR calculation failed: {e}")
            df["adx"] = np.nan
            df["atr"] = np.nan
            df["atr_ratio"] = np.nan

        df = df.bfill().ffill()
        return df

    def add_extremes_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加专门针对极值预测的特征"""
        df = df.copy()

        # 价格变化特征 - 避免重复计算
        df["price_acceleration"] = df["returns"] - df["returns"].shift(1)
        df["acceleration_change"] = df["price_acceleration"] - df[
            "price_acceleration"
        ].shift(1)

        # 极值形态 - 核心指标
        df["high_change"] = (df["high"] - df["close"].shift(1)) / df["close"].shift(1)
        df["low_change"] = (df["low"] - df["close"].shift(1)) / df["close"].shift(1)
        df["extreme_range"] = (df["high"] - df["low"]) / df["close"]

        # 波动率突变 - 核心指标
        df["volatility_breakout"] = df["return_vol_5"] / (df["return_vol_60"] + 1e-8)
        df["volatility_expansion"] = (df["volatility_breakout"] > 1.5).astype(int)

        # 价量背离 - 核心指标
        df["price_volume_correlation"] = (
            df["returns"].rolling(10).corr(df["volume_change"])
        )
        df["price_volume_divergence"] = (df["price_volume_correlation"] < -0.5).astype(
            int
        )

        # RSI极值特征
        df["rsi_extreme_high"] = (df["rsi"] > 80).astype(int)
        df["rsi_extreme_low"] = (df["rsi"] < 20).astype(int)

        # 布林带突破特征
        df["bb_breakout_upper"] = (df["high"] > df["bb_upper"]).astype(int)
        df["bb_breakout_lower"] = (df["low"] < df["bb_lower"]).astype(int)
        df["bb_squeeze"] = (
            df["bb_width"] < df["bb_width"].rolling(20).quantile(0.1)
        ).astype(int)

        # 连续极值 - 避免过多指标
        df["consecutive_extreme_up"] = (df["returns"] > 0.02).rolling(3).sum()
        df["consecutive_extreme_down"] = (df["returns"] < -0.02).rolling(3).sum()

        # ATR标准化的极值
        df["normalized_high"] = (df["high"] - df["close"]) / (df["atr"] + 1e-8)
        df["normalized_low"] = (df["close"] - df["low"]) / (df["atr"] + 1e-8)

        # 价格位置指标 - 避免重复
        df["price_position_20"] = (df["close"] - df["low_20"]) / (
            df["high_20"] - df["low_20"] + 1e-8
        )

        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加时间特征 - 核心时段特征"""
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])

        # 基础时间特征
        df["hour"] = df["datetime"].dt.hour
        df["minute"] = df["datetime"].dt.minute
        df["day_of_week"] = df["datetime"].dt.dayofweek

        # 关键交易时段特征 - 极值多发时段
        df["is_opening"] = ((df["hour"] == 9) & (df["minute"] <= 45)).astype(int)
        df["is_closing"] = ((df["hour"] == 14) & (df["minute"] >= 45)).astype(int)

        return df

    def add_market_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加市场微观结构特征"""
        df = df.copy()

        # 买卖压力指标 - 核心指标
        df["buying_pressure"] = np.where(
            df["close"] > df["open"],
            df["volume"] * (df["close"] - df["open"]) / (df["open"] + 1e-8),
            0,
        )
        df["selling_pressure"] = np.where(
            df["close"] < df["open"],
            df["volume"] * (df["open"] - df["close"]) / (df["open"] + 1e-8),
            0,
        )
        df["net_pressure"] = df["buying_pressure"] - df["selling_pressure"]

        # 价格冲击
        df["price_impact"] = abs(df["close"] - df["open"]) / (df["volume"] + 1e-8)

        # 订单流不平衡
        df["order_flow_imbalance"] = (df["close"] - (df["high"] + df["low"]) / 2) / (
            df["high"] - df["low"] + 1e-8
        )

        return df

    def aggregate_half_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """聚合为半日数据"""
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"] = df["datetime"].dt.date
        df["hour"] = df["datetime"].dt.hour
        df["minute"] = df["datetime"].dt.minute

        # 定义半日时段
        df["is_am"] = (
            ((df["hour"] == 9) & (df["minute"] >= 30))
            | ((df["hour"] == 10) | (df["hour"] == 11))
            | ((df["hour"] == 12) & (df["minute"] <= 30))
        )
        df["is_pm"] = ((df["hour"] == 13) & (df["minute"] >= 0)) | (
            (df["hour"] == 14) | ((df["hour"] == 15) & (df["minute"] <= 0))
        )

        # 只保留交易时间
        df = df[df["is_am"] | df["is_pm"]].copy()

        # 创建半日标识
        df["half_day"] = df["is_am"].astype(int)  # 0 for PM, 1 for AM

        # 聚合字典
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "amount": "sum",
        }

        # 对数值列使用均值聚合
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in agg_dict:
                agg_dict[col] = "mean"

        # 分组聚合
        half_day_groups = df.groupby(["date", "half_day"]).agg(agg_dict).reset_index()
        half_day_groups["date"] = pd.to_datetime(half_day_groups["date"])
        half_day_groups = half_day_groups.sort_values(["date", "half_day"]).reset_index(
            drop=True
        )

        # 添加半日标识列
        half_day_groups["is_am"] = half_day_groups["half_day"].astype(bool)

        # 数据清洗
        numeric_cols_agg = [
            col
            for col in half_day_groups.columns
            if col not in ["date", "half_day", "is_am"]
        ]
        half_day_groups[numeric_cols_agg] = (
            half_day_groups[numeric_cols_agg].bfill().ffill()
        )

        # 填充剩余缺失值
        for col in numeric_cols_agg:
            half_day_groups[col] = half_day_groups[col].fillna(
                half_day_groups[col].median()
            )

        # 检查是否有缺失值
        missing_count = half_day_groups.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values after aggregation")

        return half_day_groups

    def normalize_features(
        self, df: pd.DataFrame, exclude_cols: list = None
    ) -> tuple[pd.DataFrame, StandardScaler]:
        """特征标准化"""
        df = df.copy()
        if exclude_cols is None:
            exclude_cols = ["date", "is_am", "half_day", "datetime"]

        feature_cols = [
            col
            for col in df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
        ]

        if not feature_cols:
            raise ValueError("No numeric feature columns found for normalization.")

        # 数据清洗
        for col in feature_cols:
            # 替换无穷值
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            # 限制极值
            df[col] = df[col].clip(lower=-1e6, upper=1e6)
            # 填充缺失值
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

        # 检查仍有无穷值
        inf_count = np.isinf(df[feature_cols].values).sum()
        if inf_count > 0:
            logger.warning(f"Still found {inf_count} inf values after cleaning")

        # 标准化
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])

        return df, scaler

    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """完整的特征预处理流程"""
        logger.info(colored("Adding technical indicators (no redundancy)...", "cyan"))
        df = self.add_technical_indicators(df)

        logger.info(
            colored("Adding extremes-specific features (no redundancy)...", "cyan")
        )
        df = self.add_extremes_specific_features(df)

        logger.info(colored("Adding time features (core features only)...", "cyan"))
        df = self.add_time_features(df)

        logger.info(
            colored(
                "Adding market microstructure features (core features only)...", "cyan"
            )
        )
        df = self.add_market_microstructure(df)

        logger.info(
            colored(
                f"Feature engineering completed. Total features: {len(df.columns)}",
                "green",
            )
        )

        return df


if __name__ == "__main__":
    """
    uv run dataset/rep_feature_engineering.py
    """
    file_path = "no_git_oic/SH.603678_2025.csv"
    df = pd.read_csv(file_path)
    logger.info(colored("\n%s", "green"), df.head(3))

    feature_engineer = StockFeatureEngineer()

    df = feature_engineer.preprocess_features(df)
    logger.info(colored("\n%s", "green"), df.head())
