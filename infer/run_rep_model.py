import os
import sys
from dotenv import load_dotenv
from termcolor import colored
import torch

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)

from z_utils.logging_config import get_logger
from dataset.rep_dataset import RollingExtremaDataset
from model.rep_lstm_model import LSTMPredictor

load_dotenv()
logger = get_logger(__name__)


def infer_and_compare(model, test_dataset):
    """
    对测试集每个样本进行推理并对比输出
    """
    model.eval()
    print(colored("Prediction\t\tActual", "cyan", attrs=["bold"]))  # 表头
    print("-" * 40)

    with torch.no_grad():
        for idx, (x, y) in enumerate(test_dataset):
            # logger.debug(
            #     colored("Sample %s x mean: %.4f", "blue"), idx, x.mean().item()
            # )
            # 添加 batch 维度进行推理
            x_batch = x.unsqueeze(0)  # (1, seq_len, features)
            pred = model(x_batch)  # (1, output_dim)
            pred = pred.squeeze(0).cpu().numpy()  # 转换为 numpy，便于打印
            y_np = y.cpu().numpy()

            # 格式化输出：每行一个样本
            pred_str = f"{pred[0]:.4f}, {pred[1]:.4f}"
            actual_str = f"{y_np[0]:.4f}, {y_np[1]:.4f}"
            print(f"{pred_str}\t\t{actual_str}")

            if idx >= 9:
                break


def main():
    """
    主函数：加载模型和数据集，进行推理对比
    """
    features = 6  # 数据的特征维度
    hidden_dim = 32
    num_layers = 2
    output_dim = 2
    dropout = 0.1
    split_ratio = 0.9
    file_path = "no_git_oic/"
    model_path = "no_git_oic/models/lstm_predictor.pth"

    # 加载测试数据集
    test_dataset = RollingExtremaDataset(
        file_path, split="test", split_ratio=split_ratio
    )
    logger.info(colored(f"Test dataset size: {len(test_dataset)}", "yellow"))

    # 示例数据形状
    sample_x, sample_y = test_dataset[0]
    logger.info(colored(f"Test x.shape: {sample_x.shape}", "yellow"))
    logger.info(colored(f"Test y: {sample_y}", "yellow"))

    # 初始化模型并加载权重
    model = LSTMPredictor(features, hidden_dim, num_layers, output_dim, dropout)

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
    main()
