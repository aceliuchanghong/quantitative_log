import os
import sys
from dotenv import load_dotenv
from termcolor import colored
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)

from z_utils.logging_config import get_logger
from dataset.rep_dataset import RollingExtremaDataset

load_dotenv()
logger = get_logger(__name__)


class LSTMPredictor(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=64, num_layers=2, output_dim=2, dropout=0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.lstm = nn.LSTM(
            self.input_dim,
            self.hidden_dim,
            self.num_layers,
            dropout=self.dropout,  # 在 LSTM 中，dropout 只在层与层之间应用，不在时间步之间应用
            batch_first=True,
        )
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(
            x
        )  # encoder: 取最后hidden, hn[-1]==>(batch_size, hidden_dim)
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
