import os
import sys
from dotenv import load_dotenv
from termcolor import colored
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)

from z_utils.logging_config import get_logger
from dataset.rep_dataset import RollingExtremaDataset
from model.rep_lstm_model import LSTMPredictor
from common.rep_config import (
    file_path,
    window_size,
    batch_size,
    num_epochs,
    hidden_dim,
    num_layers,
    output_dim,
    dropout,
    learning_rate,
    split_ratio,
    save_model_path,
)

load_dotenv()
logger = get_logger(__name__)


def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=10,
    save_model_path=None,
    patience=10,
):
    """训练函数"""
    best_loss = float("inf")
    patience_counter = 0
    train_losses = []

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        batch_count = 0

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            # if batch_idx % 50 == 0:
            #     logger.debug(
            #         f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}"
            #     )

        avg_loss = total_loss / batch_count
        train_losses.append(avg_loss)

        # 调整学习率
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_loss)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            colored(
                f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}",
                "green",
            )
        )

        # 早停和模型保存
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            if save_model_path:
                os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "loss": best_loss,
                        "feature_cols": train_loader.dataset.feature_cols,
                        "scaler": train_loader.dataset.scaler,
                    },
                    save_model_path,
                )
                logger.info(
                    colored(
                        f"Best model saved at epoch {epoch+1} with loss {best_loss:.4f}",
                        "yellow",
                    )
                )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    colored(
                        f"Early stopping at epoch {epoch+1} (patience {patience} exceeded)",
                        "red",
                    )
                )
                break

    return train_losses


def evaluate_model(model, test_loader, criterion, test_dataset):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

            if batch_idx == 0:
                logger.info(
                    colored(
                        f"Batch {batch_idx} Pred (normalized): {outputs[0].numpy()}, Target (normalized): {y_batch[0].numpy()}",
                        "blue",
                    )
                )

    avg_loss = total_loss / len(test_loader)
    logger.info(colored(f"Test Avg MSE: {avg_loss:.4f}", "blue"))

    # 反归一化：合并预测和实际值
    preds_flat = np.concatenate(all_preds)[:5]  # 前5个样本
    targets_flat = np.concatenate(all_targets)[:5]

    # 扩展为 (n_samples, n_features) 形状，只取 high/low 两列
    high_idx = test_dataset.high_idx
    low_idx = test_dataset.low_idx
    feature_cols = test_dataset.feature_cols  # 获取所有特征列

    # 创建反归一化数组：每个样本的 [..., high, low, ...] 但只填充 high/low
    n_samples = len(preds_flat)
    dummy_features = np.zeros((n_samples, len(feature_cols)))  # 占位符
    for i in range(n_samples):
        dummy_features[i, high_idx] = preds_flat[i, 0]  # pred high
        dummy_features[i, low_idx] = preds_flat[i, 1]  # pred low
    pred_original = test_dataset.scaler.inverse_transform(dummy_features)
    pred_high_low = pred_original[:, [high_idx, low_idx]]  # 只取 high/low

    # 类似地处理 targets
    for i in range(n_samples):
        dummy_features[i, high_idx] = targets_flat[i, 0]  # target high
        dummy_features[i, low_idx] = targets_flat[i, 1]  # target low
    target_original = test_dataset.scaler.inverse_transform(dummy_features)
    target_high_low = target_original[:, [high_idx, low_idx]]

    # 打印表格：归一化 vs 原始价格
    col_width = 30
    print("\n" + "=" * (col_width * 4))
    print(
        f"{'Prediction (normalized)':<{col_width}} {'Actual (normalized)':<{col_width}} {'Pred (high, low)':<{col_width}} {'Actual (high, low)':<{col_width}}"
    )
    print("=" * (col_width * 4))

    for i in range(3):
        pred_norm = f"{preds_flat[i, 0]:.4f}, {preds_flat[i, 1]:.4f}"
        actual_norm = f"{targets_flat[i, 0]:.4f}, {targets_flat[i, 1]:.4f}"
        pred_orig = f"{pred_high_low[i, 0]:.4f}, {pred_high_low[i, 1]:.4f}"
        actual_orig = f"{target_high_low[i, 0]:.4f}, {target_high_low[i, 1]:.4f}"

        print(
            f"{pred_norm:<{col_width}} {actual_norm:<{col_width}} {pred_orig:<{col_width}} {actual_orig:<{col_width}}"
        )

    print("=" * (col_width * 4))

    return avg_loss


def main(
    file_path,
    window_size,
    batch_size,
    num_epochs,
    hidden_dim,
    num_layers,
    output_dim,
    dropout,
    learning_rate,
    split_ratio,
    save_model_path,
):
    """
    主函数：加载数据、训练模型、评估并保存
    """

    # 加载数据集
    train_dataset = RollingExtremaDataset(
        file_path, window_size=window_size, split="train", split_ratio=split_ratio
    )
    test_dataset = RollingExtremaDataset(
        file_path, window_size=window_size, split="test", split_ratio=split_ratio
    )

    logger.info(colored(f"Train dataset size: {len(train_dataset)}", "yellow"))
    logger.info(colored(f"Test dataset size: {len(test_dataset)}", "yellow"))

    sample_x, sample_y = train_dataset[0]
    num_features = sample_x.shape[1]
    logger.info(colored(f"Train x.shape: {sample_x.shape}", "yellow"))
    logger.info(colored(f"Train y: {sample_y}", "yellow"))

    sample_x_test, sample_y_test = test_dataset[0]
    logger.info(colored(f"Test x.shape: {sample_x_test.shape}", "yellow"))
    logger.info(colored(f"Test y: {sample_y_test}", "yellow"))

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = LSTMPredictor(num_features, hidden_dim, num_layers, output_dim, dropout)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # 训练
    logger.info(colored("Starting training...", "green"))
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs)

    # 评估
    logger.info(colored("Starting evaluation...", "blue"))
    evaluate_model(model, test_loader, criterion, test_dataset)

    # 保存模型
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    torch.save(model.state_dict(), save_model_path)
    logger.info(
        colored("Model saved to %s", "magenta"), os.path.dirname(save_model_path)
    )


if __name__ == "__main__":
    """
    uv run train/train_eval_rep.py

    Test Avg MSE: 0.0416 ==> 2024+2025
    """
    main(
        file_path,
        window_size,
        batch_size,
        num_epochs,
        hidden_dim,
        num_layers,
        output_dim,
        dropout,
        learning_rate,
        split_ratio,
        save_model_path,
    )
