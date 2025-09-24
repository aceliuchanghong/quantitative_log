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
        """添加技术指标"""
        df = df.copy()

        # 价格相关指标
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # 移动平均线
        for period in [5, 10, 20, 30, 60]:
            df[f"ma_{period}"] = df["close"].rolling(window=period).mean()
            df[f"close_ma_{period}_ratio"] = df["close"] / df[f"ma_{period}"]

        # MACD
        macd, macd_signal, macd_hist = talib.MACD(df["close"])
        df["macd"] = macd
        df["macd_signal"] = macd_signal
        df["macd_hist"] = macd_hist

        # RSI
        df["rsi"] = talib.RSI(df["close"], timeperiod=14)

        # 布林带
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df["close"])
        df["bb_upper"] = bb_upper
        df["bb_lower"] = bb_lower
        df["bb_middle"] = bb_middle
        df["bb_width"] = (bb_upper - bb_lower) / bb_middle
        df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)

        # KDJ指标
        k, d = talib.STOCH(
            df["high"],
            df["low"],
            df["close"],
            fastk_period=9,
            slowk_period=3,
            slowd_period=3,
        )
        df["k"] = k
        df["d"] = d
        df["j"] = 3 * df["k"] - 2 * df["d"]

        # 波动率指标
        for period in [5, 10, 20]:
            df[f"return_vol_{period}"] = df["returns"].rolling(window=period).std()
            df[f"high_low_vol_{period}"] = (
                ((df["high"] - df["low"]) / df["close"]).rolling(window=period).std()
            )

        # 成交量指标
        df["volume_ma"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]
        df["volume_change"] = df["volume"].pct_change()

        # 价格动量
        for period in [5, 10, 20]:
            df[f"momentum_{period}"] = df["close"] / df["close"].shift(period)
            df[f"roc_{period}"] = talib.ROC(df["close"], timeperiod=period)

        # 价格范围
        for period in [5, 10, 20]:
            df[f"high_{period}"] = df["high"].rolling(window=period).max()
            df[f"low_{period}"] = df["low"].rolling(window=period).min()
            df[f"high_low_range_{period}"] = (
                df[f"high_{period}"] - df[f"low_{period}"]
            ) / df["close"]

        # ADX (趋势强度)
        df["adx"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)

        # ATR (真实波幅)
        df["atr"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
        df["atr_ratio"] = df["atr"] / df["close"]

        # CCI (商品通道指数)
        df["cci"] = talib.CCI(df["high"], df["low"], df["close"], timeperiod=14)

        # 填充NaN值（修复：使用 bfill 和 ffill 避免 FutureWarning）
        df = df.bfill().ffill()

        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加时间特征"""
        df = df.copy()

        df["datetime"] = pd.to_datetime(df["datetime"])

        # 日期时间特征
        df["hour"] = df["datetime"].dt.hour
        df["minute"] = df["datetime"].dt.minute
        df["day_of_week"] = df["datetime"].dt.dayofweek
        df["month"] = df["datetime"].dt.month
        df["quarter"] = df["datetime"].dt.quarter
        df["day_of_year"] = df["datetime"].dt.dayofyear

        # 交易时段特征
        df["is_morning_rush"] = ((df["hour"] == 9) & (df["minute"] <= 30)).astype(int)
        df["is_lunch_break"] = ((df["hour"] == 11) & (df["minute"] >= 30)).astype(int)
        df["is_afternoon_active"] = ((df["hour"] == 14) & (df["minute"] <= 30)).astype(
            int
        )

        # 一周中的一天one-hot编码
        for i in range(5):
            df[f"day_{i}"] = (df["day_of_week"] == i).astype(int)

        # 月份one-hot编码
        for i in range(1, 13):
            df[f"month_{i}"] = (df["month"] == i).astype(int)

        return df

    def add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加价格形态特征"""
        df = df.copy()

        # 价格与开盘价的关系
        df["high_open_ratio"] = df["high"] / df["open"]
        df["low_open_ratio"] = df["low"] / df["open"]
        df["close_open_ratio"] = df["close"] / df["open"]

        # 价格区间占比
        df["body_size"] = abs(df["close"] - df["open"]) / (
            df["high"] - df["low"] + 1e-8
        )
        df["upper_shadow"] = (df["high"] - np.maximum(df["open"], df["close"])) / (
            df["high"] - df["low"] + 1e-8
        )
        df["lower_shadow"] = (np.minimum(df["open"], df["close"]) - df["low"]) / (
            df["high"] - df["low"] + 1e-8
        )

        # 价格位置
        df["price_position"] = (df["close"] - df["low"]) / (
            df["high"] - df["low"] + 1e-8
        )

        return df

    def aggregate_half_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """按日期 + 上午/下午分组，聚合每个半日的极值和其他特征"""
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])

        # 添加日期和上午/下午标识
        df["date"] = df["datetime"].dt.date
        df["hour"] = df["datetime"].dt.hour
        df["is_am"] = ((df["hour"] >= 9) & (df["hour"] <= 11)).astype(bool)

        # 过滤交易时段：上午 9:00-11:30，下午 13:00-15:00
        df = df[df["is_am"] | ((df["hour"] >= 13) & (df["hour"] <= 15))]

        # 构建聚合字典：基础价格/成交量使用特定聚合，其他数值特征使用 mean
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "amount": "sum",
        }
        # 对于其他数值列，使用 mean（排除非数值如 datetime, date, hour, is_am）
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in agg_dict:
                agg_dict[col] = "mean"

        # 执行聚合（groupby 会自动包含 date 和 is_am）
        half_day_groups = df.groupby(["date", "is_am"]).agg(agg_dict).reset_index()

        # 转换 date 为 datetime（便于后续排序）
        half_day_groups["date"] = pd.to_datetime(half_day_groups["date"])

        # 排序：按日期 + is_am（AM 先于 PM）
        half_day_groups = half_day_groups.sort_values(["date", "is_am"]).reset_index(
            drop=True
        )

        # 填充任何 NaN（聚合后可能残留）：先 bfill/ffill，再用中位数填充剩余
        numeric_cols_agg = [
            col for col in half_day_groups.columns if col not in ["date", "is_am"]
        ]
        half_day_groups[numeric_cols_agg] = (
            half_day_groups[numeric_cols_agg].bfill().ffill()
        )
        for col in numeric_cols_agg:
            half_day_groups[col] = half_day_groups[col].fillna(
                half_day_groups[col].median()
            )

        return half_day_groups

    def normalize_features(
        self, df: pd.DataFrame, exclude_cols: list = None
    ) -> tuple[pd.DataFrame, StandardScaler]:
        """对数值特征进行 StandardScaler 归一化（修复：添加 inf/极端值处理）"""
        df = df.copy()
        if exclude_cols is None:
            exclude_cols = ["date", "is_am", "datetime"]  # 默认排除非特征列

        # 选择数值特征列
        feature_cols = [
            col
            for col in df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
        ]
        if not feature_cols:
            raise ValueError("No numeric feature columns found for normalization.")

        # 处理 inf、极端值和 NaN（修复关键步骤）
        for col in feature_cols:
            # 替换 inf 为 NaN
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            # 限制极端值（针对股票数据，±1e6 足够）
            df[col] = df[col].clip(lower=-1e6, upper=1e6)
            # 用中位数填充 NaN
            df[col] = df[col].fillna(df[col].median())

        # 验证无 inf（可选日志）
        inf_count = np.isinf(df[feature_cols].values).sum()
        if inf_count > 0:
            logger.warning(
                f"Still found {inf_count} inf values after cleaning; check data."
            )

        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])

        return df, scaler

    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """完整特征工程流程（简化：动态特征列，无需手动列表）"""
        df = self.add_technical_indicators(df)
        df = self.add_time_features(df)
        df = self.add_price_patterns(df)

        # 保留所有列（包括 datetime），后续聚合/标准化会处理
        return df


if __name__ == "__main__":
    """
    uv run dataset/rep_feature_engineering.py
    """
    file_path = "no_git_oic/SH.603678_2025.csv"
    df = pd.read_csv(file_path)
    logger.info(colored("%s", "green"), df.head(3))

    feature_engineer = StockFeatureEngineer()

    df = feature_engineer.preprocess_features(df)
    logger.info(colored("%s", "green"), df.head())
