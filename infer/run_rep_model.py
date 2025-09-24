import os
import sys
from dotenv import load_dotenv
from termcolor import colored
import torch
import numpy as np

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)

from z_utils.logging_config import get_logger
from dataset.rep_dataset import RollingExtremaDataset
from model.rep_lstm_model import LSTMPredictor
from common.rep_config import (
    file_path,
    hidden_dim,
    num_layers,
    output_dim,
    dropout,
    split_ratio,
    save_model_path,
)

load_dotenv()
logger = get_logger(__name__)


def infer_and_compare(model, test_dataset):
    """
    对测试集每个样本进行推理并对比输出，包括反归一化后的真实值
    """
    model.eval()

    # 获取数据集属性
    high_idx = test_dataset.high_idx
    low_idx = test_dataset.low_idx
    feature_cols = test_dataset.feature_cols
    scaler = test_dataset.scaler

    print(
        "\nPrediction (normalized)\t\tActual (normalized)\t\tPred (high, low)\t\t\tActual (high, low)"
    )
    print("-" * 120)

    with torch.no_grad():
        for idx in range(min(10, len(test_dataset))):  # 前10个样本
            x, y = test_dataset[idx]

            # 添加 batch 维度进行推理
            x_batch = x.unsqueeze(0)  # (1, seq_len, features)
            pred = model(x_batch)  # (1, output_dim)
            pred = pred.squeeze(0).cpu().numpy()  # 转换为 numpy，便于打印
            y_np = y.cpu().numpy()

            # 反归一化预测值
            n_samples = 1
            dummy_features = np.zeros((n_samples, len(feature_cols)))  # 占位符
            dummy_features[0, high_idx] = pred[0]  # pred high
            dummy_features[0, low_idx] = pred[1]  # pred low
            pred_original = scaler.inverse_transform(dummy_features)
            pred_high_low = pred_original[0, [high_idx, low_idx]]  # 只取 high/low

            # 反归一化实际值
            dummy_features[0, high_idx] = y_np[0]  # target high
            dummy_features[0, low_idx] = y_np[1]  # target low
            target_original = scaler.inverse_transform(dummy_features)
            target_high_low = target_original[0, [high_idx, low_idx]]

            # 格式化输出
            pred_norm_str = f"{pred[0]:.4f}, {pred[1]:.4f}"
            actual_norm_str = f"{y_np[0]:.4f}, {y_np[1]:.4f}"
            pred_real_str = f"{pred_high_low[0]:.4f}, {pred_high_low[1]:.4f}"
            actual_real_str = f"{target_high_low[0]:.4f}, {target_high_low[1]:.4f}"
            print(
                f"{pred_norm_str}\t\t\t"
                f"{actual_norm_str}\t\t\t"
                f"{pred_real_str}\t\t\t"
                f"{actual_real_str}"
            )


def main(
    file_path,
    hidden_dim,
    num_layers,
    output_dim,
    dropout,
    split_ratio,
    model_path,
):
    """
    主函数：加载模型和数据集，进行推理对比
    """

    # 加载测试数据集
    test_dataset = RollingExtremaDataset(
        file_path, split="test", split_ratio=split_ratio
    )
    logger.info(colored(f"Test dataset size: {len(test_dataset)}", "yellow"))

    # 示例数据形状
    sample_x, sample_y = test_dataset[0]
    num_features = sample_x.shape[1]
    logger.info(colored(f"Test x.shape: {sample_x.shape}", "yellow"))
    logger.info(colored(f"Test y: {sample_y}", "yellow"))

    # 初始化模型并加载权重
    model = LSTMPredictor(num_features, hidden_dim, num_layers, output_dim, dropout)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        logger.info(colored(f"Model loaded from {model_path}", "green"))
    else:
        logger.error(colored(f"Model file {model_path} not found!", "red"))
        return

    # 进行推理对比
    logger.info(colored("Starting inference and comparison...", "blue"))
    infer_and_compare(model, test_dataset)


if __name__ == "__main__":
    """
    uv run infer/run_rep_model.py
    """
    main(
        file_path,
        hidden_dim,
        num_layers,
        output_dim,
        dropout,
        split_ratio,
        save_model_path,
    )
