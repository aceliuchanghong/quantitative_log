import os
import sys
from dotenv import load_dotenv
from termcolor import colored
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)

from z_utils.logging_config import get_logger
from dataset.rep_dataset import RollingExtremaDataset

load_dotenv()
logger = get_logger(__name__)


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # encoder: 取最后hidden
        out = self.fc(hn[-1])  # decoder: 全连接层 FC 回归
        return out


if __name__ == "__main__":
    """
    uv run model/rep_lstm_model.py
    """
    features = 6  # 数据的特征维度
    file_path = "no_git_oic/SH.603678_2025.csv"

    model = LSTMPredictor(features)
    train_dataset = RollingExtremaDataset(file_path, split="train", split_ratio=0.8)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    for x_batch, y_batch in train_loader:
        logger.info(colored("Batch x shape:%s", "green"), x_batch.shape)
        logger.info(colored("Batch y shape:%s", "green"), y_batch.shape)
        break

    with torch.no_grad():
        pred = model(x_batch)
        logger.info(colored("Model output shape:%s", "green"), pred.shape)
        logger.info(colored("\nPredictions:\n%s", "green"), pred)
        logger.info(colored("\nTargets:\n%s", "green"), y_batch)
