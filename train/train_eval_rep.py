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
            # if batch_idx % 100 == 0:
            #     logger.info(
            #         f"Epoch {epoch+1}, Batch {batch_idx}, Output sample: {outputs[0].detach().cpu().numpy()}"
            #     )

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
    print(
        "\nPrediction (normalized)\t\tActual (normalized)\t\tPred (high, low)\t\t\tActual (high, low)"
    )
    print("-" * 120)
    for i in range(5):
        print(
            f"{preds_flat[i, 0]:.4f}, {preds_flat[i, 1]:.4f}\t\t\t"
            f"{targets_flat[i, 0]:.4f}, {targets_flat[i, 1]:.4f}\t\t\t"
            f"{pred_high_low[i, 0]:.4f}, {pred_high_low[i, 1]:.4f}\t\t\t"
            f"{target_high_low[i, 0]:.4f}, {target_high_low[i, 1]:.4f}"
        )

    return avg_loss


def main(
    file_path,
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
    num_features = sample_x.shape[1]
    logger.info(colored(f"Train x.shape: {sample_x.shape}", "yellow"))
    logger.info(colored(f"Train y: {sample_y}", "yellow"))

    sample_x_test, sample_y_test = test_dataset[0]
    logger.info(colored(f"Test x.shape: {sample_x_test.shape}", "yellow"))
    logger.info(colored(f"Test y: {sample_y_test}", "yellow"))

    # for i in range(min(5, len(train_dataset))):
    #     x_sample, y_sample = train_dataset[i]
    #     msg = (
    #         f"Train Sample {i}: x shape={x_sample.shape}, "
    #         f"x mean={x_sample.mean():.4f}, x std={x_sample.std():.4f}, y={y_sample}"
    #     )
    #     logger.debug(colored("%s", "blue"), msg)

    # for i in range(min(3, len(test_dataset))):
    #     x_sample, y_sample = test_dataset[i]
    #     msg = (
    #         f"Test Sample {i}: x shape={x_sample.shape}, "
    #         f"x mean={x_sample.mean():.4f}, x std={x_sample.std():.4f}, y={y_sample}"
    #     )
    #     logger.debug(colored("%s", "blue"), msg)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = LSTMPredictor(num_features, hidden_dim, num_layers, output_dim, dropout)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练
    logger.info(colored("Starting training...", "green"))
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # 评估
    logger.info(colored("Starting evaluation...", "blue"))
    evaluate_model(model, test_loader, criterion, test_dataset)

    # 保存模型
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    torch.save(model.state_dict(), save_model_path)
    logger.info(
        colored("Model saved to %s", "magenta"), os.path.dirname(save_model_path)
    )


import optuna


def objective(trial):
    # 动态采样超参数
    hidden_dim = trial.suggest_int("hidden_dim", 32, 128)  # 范围：32~128
    num_layers = trial.suggest_int("num_layers", 1, 3)  # 1~3 层
    dropout = trial.suggest_float("dropout", 0.1, 0.5)  # 0.1~0.5
    learning_rate = trial.suggest_float(
        "learning_rate", 1e-4, 1e-2, log=True
    )  # log 尺度
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])  # 离散选项

    # 加载数据集（固定 split_ratio）
    train_dataset = RollingExtremaDataset(
        file_path, split="train", split_ratio=split_ratio
    )
    test_dataset = RollingExtremaDataset(
        file_path, split="test", split_ratio=split_ratio
    )

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 模型
    num_features = train_dataset[0][0].shape[1]
    model = LSTMPredictor(num_features, hidden_dim, num_layers, output_dim, dropout)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练
    train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)

    # 评估
    test_loss = evaluate_model(
        model, test_loader, criterion, test_dataset
    )  # 传入 test_dataset 以支持反归一化

    return test_loss  # Optuna 最小化这个值


def main_optuna(
    file_path, split_ratio, save_model_path=None, num_epochs=None, n_trials=100
):
    study = optuna.create_study(direction="minimize")  # 最小化 MSE
    study.optimize(objective, n_trials=n_trials)  # 运行 k 次试验

    logger.info(colored(f"Best trial: {study.best_trial.value:.4f}", "green"))
    logger.info(colored(f"Best params: {study.best_trial.params}", "green"))


if __name__ == "__main__":
    """
    uv run train/train_eval_rep.py

    Test Avg MSE: 0.0534 ==> 2024+2025
    """
    main(
        file_path,
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
    # main_optuna(file_path, split_ratio, save_model_path, num_epochs)
