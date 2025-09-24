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
    features,
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
    num_epochs=10,
    save_model_path=None,
    patience=5,
):
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    best_loss = float("inf")
    patience_counter = 0
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            # 梯度裁剪，防爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

            # 每 k batch 打印一个输出样例
            if batch_idx % 100 == 0:
                logger.info(
                    f"Epoch {epoch+1}, Batch {batch_idx}, Output sample: {outputs[0].detach().cpu().numpy()}"
                )

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        logger.info(
            colored(
                f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}", "green"
            )
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # 保存最佳模型
            if save_model_path:
                os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
                torch.save(model.state_dict(), save_model_path)
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


def evaluate_model(model, test_loader, criterion):
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

            # 打印 batch 的样例
            if batch_idx % 20 == 0:
                logger.info(
                    colored(
                        f"Batch {batch_idx} Pred: {outputs[0].numpy()}, Target: {y_batch[0].numpy()}",
                        "blue",
                    )
                )

    avg_loss = total_loss / len(test_loader)
    logger.info(colored(f"Test Avg MSE: {avg_loss:.4f}", "blue"))

    # 合并打印前 5 个预测 vs 实际（表格形式）
    preds_flat = np.concatenate(all_preds)[:5]
    targets_flat = np.concatenate(all_targets)[:5]
    print("Prediction\t\tActual")
    print("-" * 40)
    for p, t in zip(preds_flat, targets_flat):
        print(f"{p[0]:.4f}, {p[1]:.4f}\t\t{t[0]:.4f}, {t[1]:.4f}")

    return avg_loss


def main(
    file_path,
    features,
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
        file_path, split="train", split_ratio=split_ratio
    )
    test_dataset = RollingExtremaDataset(
        file_path, split="test", split_ratio=split_ratio
    )

    logger.info(colored(f"Train dataset size: {len(train_dataset)}", "yellow"))
    logger.info(colored(f"Test dataset size: {len(test_dataset)}", "yellow"))

    sample_x, sample_y = train_dataset[0]
    logger.info(colored(f"Train x.shape: {sample_x.shape}", "yellow"))
    logger.info(colored(f"Train y: {sample_y}", "yellow"))

    sample_x_test, sample_y_test = test_dataset[0]
    logger.info(colored(f"Test x.shape: {sample_x_test.shape}", "yellow"))
    logger.info(colored(f"Test y: {sample_y_test}", "yellow"))

    for i in range(min(5, len(train_dataset))):
        x_sample, y_sample = train_dataset[i]
        msg = (
            f"Train Sample {i}: x shape={x_sample.shape}, "
            f"x mean={x_sample.mean():.4f}, x std={x_sample.std():.4f}, y={y_sample}"
        )
        logger.debug(colored("%s", "blue"), msg)

    for i in range(min(3, len(test_dataset))):
        x_sample, y_sample = test_dataset[i]
        msg = (
            f"Test Sample {i}: x shape={x_sample.shape}, "
            f"x mean={x_sample.mean():.4f}, x std={x_sample.std():.4f}, y={y_sample}"
        )
        logger.debug(colored("%s", "blue"), msg)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = LSTMPredictor(features, hidden_dim, num_layers, output_dim, dropout)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练
    logger.info(colored("Starting training...", "green"))
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # 评估
    logger.info(colored("Starting evaluation...", "blue"))
    evaluate_model(model, test_loader, criterion)

    # 保存模型
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    torch.save(model.state_dict(), save_model_path)
    logger.info(
        colored("Model saved to %s", "magenta"), os.path.dirname(save_model_path)
    )


if __name__ == "__main__":
    """
    uv run train/train_eval_rep.py
    """
    main(
        file_path,
        features,
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
