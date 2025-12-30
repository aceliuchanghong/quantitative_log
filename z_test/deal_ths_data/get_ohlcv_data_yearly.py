import os
import sys
import asyncio
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from termcolor import colored

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
)
from z_utils.logging_config import get_logger
from z_test.deal_ths_data.ohlcv_data_core import (
    get_stock_range_data_async,
    thslogindemo,
)
from z_test.deal_ths_data.get_trade_date import get_trading_days
from z_test.deal_ths_data.get_trade_code import get_historical_codes_baostock_core
from z_test.deal_ths_data.get_ohlcv_data import format_stock_code
from z_utils.db_cache import cache_to_duckdb

load_dotenv()
logger = get_logger(__name__)

from typing import List


@cache_to_duckdb()
def aggregate_and_filter_codes(date_list: List[str]) -> List[str]:
    """
    聚合多个日期的股票代码，去重并排除以 '399' 开头的代码。

    参数:
        date_list (List[str]): 日期字符串列表，格式需与 get_historical_codes_baostock_core 兼容。

    返回:
        List[str]: 去重且不以 '399' 开头的股票代码列表。
    """
    import baostock as bs

    bs.login()

    all_codes = []
    for this_date in date_list:
        codes = get_historical_codes_baostock_core(this_date)
        if codes:  # 确保 codes 非空且可迭代
            all_codes.extend(codes)

    # 使用集合去重，再转回列表
    unique_codes = set(all_codes)

    # 过滤掉以 '399' 开头的代码
    filtered_codes = [code for code in unique_codes if not code.startswith("399")]

    # 对结果排序以保证可重现性
    filtered_codes.sort()

    bs.logout()

    return filtered_codes


def save_data_to_csv_yearly(
    df: pd.DataFrame, stock_code: str, target_date_year: str
) -> str:
    """
    保存数据到指定目录: no_git_oic/historical_data_yearly/{target_date_year}/{stock_code}.csv
    """
    if df is None or df.empty:
        return

    # 转换日期格式: 2025-12-03 -> 20251203
    date_str = target_date_year[:4]

    # 构造目录路径
    base_dir = "no_git_oic/historical_data_yearly"
    target_dir = os.path.join(base_dir, date_str)

    # 递归创建目录
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 构造文件名并保存
    file_path = os.path.join(target_dir, f"{stock_code}.csv")
    df.to_csv(file_path, index=False, encoding="utf-8")

    # print(colored(f"文件已保存至: {file_path}", "green"))
    return file_path


async def process_one(sem, year, code):
    """
    单个任务的处理逻辑，使用 sem 控制并发
    """
    async with sem:  # 获取信号量，超过限制则等待
        formatted_code = format_stock_code(code)

        if len(formatted_code) > 8:
            # logger.info(colored("year:%s|code:%s", "blue"), year, formatted_code)
            try:
                # 1. 执行异步网络请求
                df = await get_stock_range_data_async(
                    formatted_code, f"{year}-01-01", f"{year}-12-31"
                )

                # 2. 保存数据
                if df is not None and not df.empty:
                    await asyncio.to_thread(
                        save_data_to_csv_yearly, df, formatted_code, year
                    )
            except Exception as e:
                logger.error(f"处理出错 code:{formatted_code} year:{year} error:{e}")


if __name__ == "__main__":
    """
    uv run z_test/deal_ths_data/get_ohlcv_data_yearly.py
    """
    # 首先登陆
    thslogindemo()
    # 获取日期
    start_date = "2011-06-21"
    end_date = "2025-12-31"
    max_concurrent = 10
    trading_days = get_trading_days(start_date, end_date)
    date_list = trading_days.sort_values().strftime("%Y-%m-%d").tolist()
    filtered_codes = aggregate_and_filter_codes(date_list)
    year_list = [year for year in range(2011, 2026)]
    # print(colored(f"{year_list}", "light_yellow"))

    async def main():
        # 初始化信号量
        sem = asyncio.Semaphore(max_concurrent)
        tasks = []
        for year in year_list:
            year = str(year)
            logger.info(colored("year:%s", "green"), year)
            for code in filtered_codes:
                # 创建每一个任务，并放入列表
                task = asyncio.create_task(process_one(sem, year, code))
                tasks.append(task)
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="下载进度"):
            await f

    asyncio.run(main())
