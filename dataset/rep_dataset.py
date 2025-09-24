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

        # 加载数据（支持目录或单个文件）
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

        # 特征工程：添加指标、时间、价格形态
        feature_engineer = StockFeatureEngineer()
        df = feature_engineer.preprocess_features(df)

        # 半日聚合（移入 StockFeatureEngineer）
        half_day_groups = feature_engineer.aggregate_half_days(df)

        # 归一化（移入 StockFeatureEngineer，仅数值特征）
        half_day_groups, self.scaler = feature_engineer.normalize_features(
            half_day_groups
        )

        # 设置特征列（排除 date, is_am）
        self.feature_cols = [
            col for col in half_day_groups.columns if col not in ["date", "is_am"]
        ]
        self.features = half_day_groups[self.feature_cols].values.astype(np.float32)
        self.dates = half_day_groups["date"].values
        self.is_am = half_day_groups["is_am"].values

        self.num_half_days = len(self.features)
        if self.num_half_days % 2 != 0:
            logger.warning("Total half-days is odd; truncating last incomplete day.")
            self.num_half_days -= 1
            self.features = self.features[: self.num_half_days]
            self.dates = self.dates[: self.num_half_days]
            self.is_am = self.is_am[: self.num_half_days]

        self.num_days = self.num_half_days // 2
        self.window_size = window_size

        # 构建滚动样本：匹配目标描述（预测 k 天 AM/PM，滑动加入真实 AM）
        self.sample_starts = []
        self.target_idxs = []

        min_start_day = window_size // 2  # 至少需要这么多天
        for k in range(min_start_day, self.num_days):
            # 预测第 k 天 AM：从 (k - window_size//2) 天 AM 开始的 window_size 个半日
            start_am = (k - window_size // 2) * 2
            target_am = k * 2  # 第 k 天 AM 索引
            if start_am >= 0 and start_am + self.window_size <= target_am:
                self.sample_starts.append(start_am)
                self.target_idxs.append(target_am)

            # 预测第 k 天 PM：滑动一步，从 (k - window_size//2 + 0.5) 天（即 start_am +1）开始，加入真实 AM
            start_pm = start_am + 1
            target_pm = k * 2 + 1  # 第 k 天 PM 索引
            if start_pm >= 0 and start_pm + self.window_size <= target_pm:
                self.sample_starts.append(start_pm)
                self.target_idxs.append(target_pm)

        # 根据 split 参数分割数据集（时间序列顺序，避免泄漏）
        total_samples = len(self.sample_starts)
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

        # high/low 列索引（确保存在）
        if "high" not in self.feature_cols or "low" not in self.feature_cols:
            raise ValueError("Features must include 'high' and 'low'")
        self.high_idx = self.feature_cols.index("high")
        self.low_idx = self.feature_cols.index("low")

        logger.info(
            colored(
                f"Dataset initialized: {len(self.sample_starts)} samples "
                f"(from {self.num_days} days, window_size={self.window_size}, split={split})",
                "green",
            )
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
