import os
import sys
import pandas as pd
from dotenv import load_dotenv
from termcolor import colored

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)
from z_utils.logging_config import get_logger

load_dotenv()
logger = get_logger(__name__)


if __name__ == "__main__":
    """
    uv run dataset/mixer_dataset.py
    """
    file_path = "no_git_oic/2012/SH.600000.csv"
    df = pd.read_csv(file_path)
    df = df.drop(columns=["amount"])
    logger.info(colored("\n%s", "green"), df.head(3))
