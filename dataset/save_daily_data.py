import os
import sys
from dotenv import load_dotenv
from termcolor import colored
import pandas as pd
import time
from functools import partial

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)

from z_utils.db_cache import cache_to_duckdb
from z_utils.logging_config import get_logger
from tools.basic_info_tool import get_a_share_stock_list, get_trading_days
from tools.get_data_tool import get_intraday_data_yf, get_intraday_data_ak

load_dotenv()
logger = get_logger(__name__)


@cache_to_duckdb(db_name="no_git_oic/202509.duckdb")
def fetch_stock_data_for_day(stock_code: str, trade_day: str) -> pd.DataFrame:
    """
    尝试获取指定股票在指定交易日的日内数据，优先尝试 ak 失败则尝试 yf 最多重试2轮共4次。
    成功则返回非空 DataFrame 失败则返回 None。
    """
    # 定义两个获取函数，带参数
    funcs = [
        partial(get_intraday_data_ak, stock_code, trade_day),
        partial(get_intraday_data_yf, stock_code, trade_day),
    ]
    func_names = ["ak", "yf"]

    for attempt in range(4):
        current_index = attempt % 2
        current_func = funcs[current_index]
        current_name = func_names[current_index]

        try:
            logger.debug(
                f"尝试使用 {current_name} 获取 {stock_code} | {trade_day} 数据..."
            )
            df = current_func()
            if not df.empty:
                return df
            else:
                logger.warning(f"{current_name} 返回空数据，切换到另一个源...")
        except Exception as e:
            logger.warning(
                colored("%s 使用 %s 获取 %s 数据失败: %s", "yellow"),
                trade_day,
                current_name,
                stock_code,
                e,
            )

        # 等待1秒再切换
        time.sleep(1)

    # 所有尝试均失败
    logger.error(f"所有尝试均失败: {stock_code} | {trade_day}")
    return None


def run_daily_save(input_date):
    """
    input_date: 日期字符串 格式如 "2024-05-10"
    """
    input_date = pd.to_datetime(input_date)

    # 获取从 input_date 这天往前数 7 天的 list
    past_seven_list = [
        date.strftime("%Y-%m-%d")
        for date in pd.date_range(
            start=input_date - pd.Timedelta(days=6), end=input_date, freq="D"
        )
    ]
    # 获取交易日
    trading_days_list = get_trading_days(past_seven_list[0], past_seven_list[-1])
    logger.debug(colored("%s", "blue"), trading_days_list)

    # 所有股票的代码
    all_code = get_a_share_stock_list(trading_days_list[-1])
    all_code = all_code["Symbol"]

    for trade_day in trading_days_list:
        for stock_code in all_code:
            stock_code_str = str(stock_code)
            logger.debug(colored("working on:%s|%s", "blue"), stock_code_str, trade_day)
            df = fetch_stock_data_for_day(stock_code_str, trade_day)
            if df is not None and not df.empty:
                logger.info(colored("\n%s", "green"), df.head())


if __name__ == "__main__":
    """
    uv run dataset/save_daily_data.py
    """
    input_date = "2025-09-21"
    re_run = False  # True,False
    run_daily_save(input_date)
