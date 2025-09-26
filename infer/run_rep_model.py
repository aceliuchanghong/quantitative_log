import os
import sys
from dotenv import load_dotenv
from termcolor import colored
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)

from z_utils.logging_config import get_logger
from dataset.rep_dataset import RollingExtremaDataset
from model.rep_lstm_model import LSTMPredictor
from common.rep_config import (
    hidden_dim,
    num_layers,
    output_dim,
    dropout,
    split_ratio,
    save_model_path,
    batch_size,
)

load_dotenv()
logger = get_logger(__name__)


def infer_model(model, infer_loader, infer_dataset, return_targets=False):
    """
    Args:
        model: 训练好的模型
        infer_loader: DataLoader 应返回 (x_batch, y_batch) 或 (x_batch,)
        infer_dataset: 包含 scaler、high_idx、low_idx、feature_cols 等信息
        return_targets: 是否返回反归一化后的真实值（如果存在）
    """
    model.eval()
    all_preds = []
    all_targets = []

    # === 新增：提前判断是否有 targets ===
    try:
        first_batch = next(iter(infer_loader))
        has_targets = len(first_batch) == 2
    except StopIteration:
        logger.warning("Inference dataloader is empty.")
        return np.array([]), np.array([]) if return_targets else np.array([])

    with torch.no_grad():
        for batch_idx, batch in enumerate(infer_loader):
            if has_targets:
                x_batch, y_batch = batch
                all_targets.append(y_batch.cpu().numpy())
            else:
                x_batch = batch

            outputs = model(x_batch)
            all_preds.append(outputs.cpu().numpy())

            if batch_idx == 0:
                msg = f"Batch {batch_idx} Pred (normalized): {outputs[0].numpy()}"
                if has_targets:
                    msg += f", Target (normalized): {y_batch[0].numpy()}"
                logger.info(colored(msg, "blue"))

    if not all_preds:
        logger.warning("No predictions generated.")
        return np.array([]), np.array([]) if return_targets else np.array([])

    # 合并预测
    all_preds_full = np.concatenate(all_preds)
    n_samples_full = len(all_preds_full)

    # 反归一化预测值
    high_idx = infer_dataset.high_idx
    low_idx = infer_dataset.low_idx
    feature_cols = infer_dataset.feature_cols

    dummy_features = np.zeros((n_samples_full, len(feature_cols)))
    for i in range(n_samples_full):
        dummy_features[i, high_idx] = all_preds_full[i, 0]
        dummy_features[i, low_idx] = all_preds_full[i, 1]
    pred_original = infer_dataset.scaler.inverse_transform(dummy_features)
    pred_high_low = pred_original[:, [high_idx, low_idx]]

    # 处理真实值（如果存在）
    target_high_low = None
    if has_targets and return_targets:
        all_targets_full = np.concatenate(all_targets)
        dummy_targets = np.zeros((n_samples_full, len(feature_cols)))
        for i in range(n_samples_full):
            dummy_targets[i, high_idx] = all_targets_full[i, 0]
            dummy_targets[i, low_idx] = all_targets_full[i, 1]
        target_original = infer_dataset.scaler.inverse_transform(dummy_targets)
        target_high_low = target_original[:, [high_idx, low_idx]]

    # 打印前3个样本
    col_width = 30
    if has_targets:
        print("\n" + "=" * (col_width * 4))
        print(
            f"{'Prediction (normalized)':<{col_width}} {'Actual (normalized)':<{col_width}} "
            f"{'Pred (high, low)':<{col_width}} {'Actual (high, low)':<{col_width}}"
        )
        print("=" * (col_width * 4))

        preds_flat = all_preds_full[:3]
        targets_flat = np.concatenate(all_targets)[:3]
        pred_orig_flat = pred_high_low[:3]
        target_orig_flat = target_high_low[:3] if target_high_low is not None else None

        for i in range(min(3, len(preds_flat))):
            pred_norm = f"{preds_flat[i, 0]:.4f}, {preds_flat[i, 1]:.4f}"
            actual_norm = f"{targets_flat[i, 0]:.4f}, {targets_flat[i, 1]:.4f}"
            pred_orig = f"{pred_orig_flat[i, 0]:.4f}, {pred_orig_flat[i, 1]:.4f}"
            actual_orig = (
                f"{target_orig_flat[i, 0]:.4f}, {target_orig_flat[i, 1]:.4f}"
                if target_orig_flat is not None
                else "N/A"
            )

            print(
                f"{pred_norm:<{col_width}} {actual_norm:<{col_width}} "
                f"{pred_orig:<{col_width}} {actual_orig:<{col_width}}"
            )
        print("=" * (col_width * 4))
    else:
        preds_flat = all_preds_full[:3]
        print("\n" + "=" * (col_width * 2))
        print(
            f"{'Prediction (normalized)':<{col_width}} {'Pred (high, low)':<{col_width}}"
        )
        print("=" * (col_width * 2))
        for i in range(min(3, len(preds_flat))):
            pred_norm = f"{preds_flat[i, 0]:.4f}, {preds_flat[i, 1]:.4f}"
            pred_orig = f"{pred_high_low[i, 0]:.4f}, {pred_high_low[i, 1]:.4f}"
            print(f"{pred_norm:<{col_width}} {pred_orig:<{col_width}}")
        print("=" * (col_width * 2))

    if return_targets and target_high_low is not None:
        return pred_high_low, target_high_low
    else:
        return pred_high_low


def main(
    file_path,
    hidden_dim,
    num_layers,
    output_dim,
    dropout,
    split_ratio,
    model_path,
    batch_size,
):
    """
    主函数：加载模型和数据集，进行推理对比
    """

    # 加载测试数据集
    infer_dataset = RollingExtremaDataset(
        file_path, split="all", split_ratio=split_ratio
    )
    logger.info(colored(f"Test dataset size: {len(infer_dataset)}", "yellow"))

    # 示例数据形状
    sample_x, sample_y = infer_dataset[0]
    num_features = sample_x.shape[1]
    logger.info(colored(f"Test x.shape: {sample_x.shape}", "yellow"))
    logger.info(colored(f"Test y: {sample_y}", "yellow"))

    # 初始化模型并加载权重
    model = LSTMPredictor(num_features, hidden_dim, num_layers, output_dim, dropout)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(colored(f"Model loaded from {model_path}", "green"))
    else:
        logger.error(colored(f"Model file {model_path} not found!", "red"))
        return

    # 进行推理对比
    logger.info(colored("Starting inference and comparison...", "blue"))
    infer_loader = DataLoader(infer_dataset, batch_size=batch_size, shuffle=False)
    infer_model(model, infer_loader, infer_dataset, True)


if __name__ == "__main__":
    """
    uv run infer/run_rep_model.py
    """
    main(
        "no_git_oic/test/",
        hidden_dim,
        num_layers,
        output_dim,
        dropout,
        split_ratio,
        save_model_path,
        batch_size,
    )
