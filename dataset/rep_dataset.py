import os
import glob
import sys
from dotenv import load_dotenv
from termcolor import colored
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)

from z_utils.logging_config import get_logger
from dataset.rep_feature_engineering import StockFeatureEngineer

load_dotenv()
logger = get_logger(__name__)


class RollingExtremaDataset(Dataset):
    def __init__(self, file_path, window_size=10, split="all", split_ratio=0.8):
        """
        Args:
            file_path (str): CSV 文件路径（或目录）
            window_size (int): 滑动窗口大小（半日数量），必须为偶数
            split (str): 数据集分割方式，'all' 为全集，'train' 为训练集，'test' 为测试集
            split_ratio (float): 训练集比例(0 到 1)，默认 0.8
        """
        if window_size % 2 != 0:
            raise ValueError(
                "window_size must be even to align with full trading days."
            )

        # 加载数据
        if os.path.isdir(file_path):
            csv_files = sorted(glob.glob(os.path.join(file_path, "*.csv")))
            if not csv_files:
                raise ValueError(f"No CSV files found in directory: {file_path}")
            logger.info(
                colored(
                    f"Loading {len(csv_files)} CSV files from directory: {file_path}",
                    "green",
                )
            )

            dfs = []
            for f in csv_files:
                df = pd.read_csv(f)
                dfs.append(df)
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_csv(file_path)

        logger.info(colored(f"Raw data shape: {df.shape}", "yellow"))

        # 使用无冗余的特征工程
        feature_engineer = StockFeatureEngineer()
        df = feature_engineer.preprocess_features(df)

        logger.info(colored(f"After feature engineering shape: {df.shape}", "yellow"))

        # 聚合为半日数据（手动实现以修复groupby后reset_index冲突问题）
        df_agg = df.copy()
        df_agg["datetime"] = pd.to_datetime(df_agg["datetime"])
        df_agg["date"] = df_agg["datetime"].dt.date
        df_agg["hour"] = df_agg["datetime"].dt.hour
        df_agg["minute"] = df_agg["datetime"].dt.minute

        # 定义半日时段
        df_agg["is_am"] = (
            ((df_agg["hour"] == 9) & (df_agg["minute"] >= 30))
            | ((df_agg["hour"] == 10) | (df_agg["hour"] == 11))
            | ((df_agg["hour"] == 12) & (df_agg["minute"] <= 30))
        )
        df_agg["is_pm"] = ((df_agg["hour"] == 13) & (df_agg["minute"] >= 0)) | (
            (df_agg["hour"] == 14) | ((df_agg["hour"] == 15) & (df_agg["minute"] <= 0))
        )

        # 只保留交易时间
        df_agg = df_agg[df_agg["is_am"] | df_agg["is_pm"]].copy()

        # 创建半日标识
        df_agg["half_day"] = df_agg["is_am"].astype(int)  # 0 for PM, 1 for AM

        # 聚合字典
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "amount": "sum",
        }

        # 对数值列使用均值聚合，排除分组键以避免reset_index冲突
        numeric_cols = [
            col
            for col in df_agg.select_dtypes(include=[np.number]).columns
            if col not in ["date", "half_day"]
        ]
        for col in numeric_cols:
            if col not in agg_dict:
                agg_dict[col] = "mean"

        # 分组聚合
        half_day_groups = (
            df_agg.groupby(["date", "half_day"]).agg(agg_dict).reset_index()
        )
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

        logger.info(
            colored(f"Half-day groups shape: {half_day_groups.shape}", "yellow")
        )

        # 标准化特征
        half_day_groups, self.scaler = feature_engineer.normalize_features(
            half_day_groups
        )

        # 提取特征列
        self.feature_cols = [
            col
            for col in half_day_groups.columns
            if col not in ["date", "is_am", "half_day"]
        ]
        self.features = half_day_groups[self.feature_cols].values.astype(np.float32)
        self.dates = half_day_groups["date"].values
        self.is_am = half_day_groups["is_am"].values

        # 确保数据完整性
        self.num_half_days = len(self.features)
        if self.num_half_days % 2 != 0:
            logger.warning("Total half-days is odd; truncating last incomplete day.")
            self.num_half_days -= 1
            self.features = self.features[: self.num_half_days]
            self.dates = self.dates[: self.num_half_days]
            self.is_am = self.is_am[: self.num_half_days]

        self.num_days = self.num_half_days // 2
        self.window_size = window_size

        # 生成样本 - 按交易日滚动
        self.sample_starts = []
        self.target_idxs = []

        # 从 window_size//2 天开始，确保有足够的历史数据
        for k in range(window_size // 2, self.num_days):
            # 预测第 k 天 AM (上午)
            start_idx_am = (
                k - window_size // 2
            ) * 2  # 从 (k - window_size//2) 天 AM 开始
            target_idx_am = k * 2  # 第 k 天 AM
            if start_idx_am >= 0 and start_idx_am + window_size <= target_idx_am:
                self.sample_starts.append(start_idx_am)
                self.target_idxs.append(target_idx_am)

            # 预测第 k 天 PM (下午) - 使用真实 AM 数据
            start_idx_pm = start_idx_am + 1  # 加入真实 AM
            target_idx_pm = k * 2 + 1  # 第 k 天 PM
            if start_idx_pm >= 0 and start_idx_pm + window_size <= target_idx_pm:
                self.sample_starts.append(start_idx_pm)
                self.target_idxs.append(target_idx_pm)

        # 数据分割
        total_samples = len(self.sample_starts)
        logger.info(colored(f"Total samples before split: {total_samples}", "yellow"))

        if split == "train":
            end_idx = int(total_samples * split_ratio)
            self.sample_starts = self.sample_starts[:end_idx]
            self.target_idxs = self.target_idxs[:end_idx]
            logger.info(
                colored(
                    f"Using train split: {len(self.sample_starts)} samples", "green"
                )
            )
        elif split == "test":
            start_idx = int(total_samples * split_ratio)
            self.sample_starts = self.sample_starts[start_idx:]
            self.target_idxs = self.target_idxs[start_idx:]
            logger.info(
                colored(f"Using test split: {len(self.sample_starts)} samples", "green")
            )
        elif split != "all":
            raise ValueError(
                f"Invalid split: {split}. Must be 'all', 'train' or 'test'."
            )

        # 获取high和low的索引
        if "high" not in self.feature_cols or "low" not in self.feature_cols:
            raise ValueError("Features must include 'high' and 'low'")

        self.high_idx = self.feature_cols.index("high")
        self.low_idx = self.feature_cols.index("low")

        logger.info(
            colored(
                f"Dataset initialized: {len(self.sample_starts)} samples "
                f"(from {self.num_days} days, window_size={self.window_size}, "
                f"features={len(self.feature_cols)}, split={split})",
                "green",
            )
        )

        # 输出一些特征名称供参考
        logger.info(
            colored(f"Sample feature names: {self.feature_cols[:5]}...", "cyan")
        )

    def __getitem__(self, idx):
        start_idx = self.sample_starts[idx]
        x = self.features[start_idx : start_idx + self.window_size]

        target_idx = self.target_idxs[idx]
        y = np.array(
            [
                self.features[target_idx, self.high_idx],
                self.features[target_idx, self.low_idx],
            ],
            dtype=np.float32,
        )

        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        return len(self.sample_starts)


if __name__ == "__main__":
    """
    Rolling Extrema Predictor Dataset
    ### 按交易日滚动,滑动 window_size 个半日预测下一个半日
        - 对于第 `k` 天 (k >= window_size//2):
            - 用过去 window_size 个半日 (从 `(k - window_size//2)` 天 AM 开始) 预测第 `k` 天 AM
            - 然后滑动一步，加入真实 AM 预测第 `k` 天 PM
        - 输入 `x`：形状 `(window_size, feature_dim)`
        - 目标 `y`：形状 `(2,)`，即 `[high, low]`

    uv run dataset/rep_dataset.py
    """
    import argparse

    parser = argparse.ArgumentParser(description="测试dataset")
    parser.add_argument(
        "--file_path", type=str, default="no_git_oic/", help="同时支持目录和文件"
    )
    args = parser.parse_args()

    dataset = RollingExtremaDataset(args.file_path)
    x, y = dataset[0]
    logger.info(colored("x.shape:%s", "green"), x.shape)
    logger.info(colored("y:%s", "green"), y)
    logger.info(colored("num_features = %s", "green"), x.shape[1])

    # 创建训练集和测试集
    train_dataset = RollingExtremaDataset(
        args.file_path, split="train", split_ratio=0.8
    )
    test_dataset = RollingExtremaDataset(args.file_path, split="test", split_ratio=0.8)

    # 测试第一个样本
    if len(train_dataset) > 0:
        x, y = train_dataset[0]
        logger.info(colored("Train x.shape: %s", "green"), x.shape)
        logger.info(colored("Train y: %s", "green"), y)
        logger.info(colored("Train dataset size: %d", "green"), len(train_dataset))

    if len(test_dataset) > 0:
        x, y = test_dataset[0]
        logger.info(colored("Test x.shape: %s", "green"), x.shape)
        logger.info(colored("Test y: %s", "green"), y)
        logger.info(colored("Test dataset size: %d", "green"), len(test_dataset))
