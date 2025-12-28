import os
import sys
import asyncio
import pandas as pd
from dotenv import load_dotenv
from termcolor import colored
from tqdm.asyncio import tqdm

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
)
from z_utils.logging_config import get_logger
from z_test.deal_ths_data.ohlcv_data_core import (
    get_stock_intraday_data,
    get_stock_intraday_data_async,
    thslogindemo,
)
from z_test.deal_ths_data.get_trade_date import get_trading_days
from z_test.deal_ths_data.get_trade_code import get_historical_codes_baostock_core

load_dotenv()
logger = get_logger(__name__)


def format_stock_code(symbol: str) -> str:
    """
    将 6位数字代码 转换为 带后缀的格式 (iFinD 标准)
    """
    symbol = str(symbol).strip()

    if not symbol.isdigit():
        return symbol

    # 补齐6位
    if len(symbol) < 6:
        symbol = symbol.zfill(6)

    # 判断规则
    if symbol.startswith(("600", "601", "603", "605", "688")):
        return f"{symbol}.SH"
    elif symbol.startswith(("000", "001", "002", "300", "301")):
        return f"{symbol}.SZ"
    elif symbol.startswith(("43", "83", "87", "88")):
        return f"{symbol}.BJ"
    else:
        return symbol


def save_data_to_csv(df: pd.DataFrame, stock_code: str, target_date: str) -> str:
    """
    保存数据到指定目录: no_git_oic/historical_data/{YYYYMMDD}/{code}.csv
    """
    if df is None or df.empty:
        # logger.warning(f"数据为空，跳过保存: {target_date} {stock_code}")
        return

    # 转换日期格式: 2025-12-03 -> 20251203
    date_str = target_date.replace("-", "")

    # 构造目录路径
    base_dir = "no_git_oic/historical_data"
    target_dir = os.path.join(base_dir, date_str)

    # 递归创建目录
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 构造文件名并保存
    file_path = os.path.join(target_dir, f"{stock_code}.csv")
    df.to_csv(file_path, index=False, encoding="utf-8")

    # print(colored(f"文件已保存至: {file_path}", "green"))
    return file_path


async def download_single_stock(
    code: str, this_date: str, semaphore: asyncio.Semaphore
):
    """
    单个股票的异步下载任务，带并发控制
    """
    async with semaphore:  # 控制并发数
        try:
            formatted_code = format_stock_code(code)
            df = await get_stock_intraday_data_async(formatted_code, this_date)
            save_data_to_csv(df, formatted_code, this_date)
        except Exception as e:
            logger.error(f"Error processing {code} on {this_date}: {e}")


async def process_date_stocks_async(
    codes: list, this_date: str, max_concurrent: int = 20
):
    """
    并发处理某一天内的所有股票
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [download_single_stock(code, this_date, semaphore) for code in codes]

    # 使用 tqdm.gather 展示异步进度
    await tqdm.gather(
        *tasks, desc=f"  Processing {this_date}", unit="stock", leave=False
    )


if __name__ == "__main__":
    """
    uv run z_test/deal_ths_data/get_ohlcv_data.py
    """
    # 首先登陆
    thslogindemo()
    # 获取日期
    start_date = "2011-06-15"
    end_date = "2025-12-26"
    trading_days = get_trading_days(start_date, end_date)
    date_list = trading_days.sort_values().strftime("%Y-%m-%d").tolist()
    # 循环每个日期
    for this_date in tqdm(date_list, desc="Overall Progress", unit="date"):
        print(colored(f"\nWorking on: {this_date}", "light_green"))

        # 获取该日所有代码
        codes = get_historical_codes_baostock_core(this_date)

        # 4运行异步事件循环处理当天的所有股票
        asyncio.run(process_date_stocks_async(codes, this_date, max_concurrent=50))
