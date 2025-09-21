import os
import sys
from dotenv import load_dotenv
from termcolor import colored

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)

from z_utils.logging_config import get_logger

load_dotenv()
logger = get_logger(__name__)


def get_ticker(stock_code: str) -> str:
    """
    根据A股股票代码判断并返回所需的 ticker 格式
    上海证券交易所的股票代码以'6'开头
    深圳证券交易所的股票代码以'0'或'3'开头

    Args:
        stock_code (str): 6位数的A股股票代码.

    Returns:
        str: 适用的股票代码 (e.g., '600519.SS').

    Raises:
        ValueError: 如果股票代码不是以'6', '0', '3'开头，则抛出此异常
    """
    if (
        not isinstance(stock_code, str)
        or not stock_code.isdigit()
        or len(stock_code) != 6
    ):
        raise ValueError(f"无效的股票代码格式: '{stock_code}'。应为6位数字字符串。")

    if stock_code.startswith("6"):
        return f"{stock_code}.SS"
    elif stock_code.startswith("0") or stock_code.startswith("3"):
        return f"{stock_code}.SZ"
    else:
        raise ValueError(
            f"无法识别的A股股票代码: '{stock_code}'。应以上海'6'或深圳'0'、'3'开头。"
        )


if __name__ == "__main__":
    """
    uv run tools/fix_code_tool.py
    """
    stock_code = "603678"
    stock_code_ticker = get_ticker(stock_code)
    logger.info(colored("%s", "green"), stock_code_ticker)
