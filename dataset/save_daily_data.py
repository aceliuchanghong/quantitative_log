import os
import sys
from dotenv import load_dotenv
from termcolor import colored
import pandas as pd
from functools import partial
from tqdm import tqdm
from collections import defaultdict

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

_source_stats = defaultdict(lambda: {"success": 0, "failure": 0})


def get_success_rate(name: str) -> float:
    """计算指定源的成功率"""
    stats = _source_stats[name]
    total = stats["success"] + stats["failure"]
    return stats["success"] / total if total > 0 else 0.5  # 默认 0.5 避免除零


def record_success(name: str):
    """记录一次成功"""
    _source_stats[name]["success"] += 1


def record_failure(name: str):
    """记录一次失败"""
    _source_stats[name]["failure"] += 1


@cache_to_duckdb(db_name="no_git_oic/202509.duckdb")
def fetch_stock_data_for_day(stock_code: str, trade_day: str) -> pd.DataFrame:
    """
    尝试获取指定股票在指定交易日的日内数据，按成功率动态排序数据源。
    成功则返回非空 DataFrame 失败则返回 None。
    """
    sources = [
        ("ak", partial(get_intraday_data_ak, stock_code, trade_day)),
        ("yf", partial(get_intraday_data_yf, stock_code, trade_day)),
    ]

    sorted_sources = sorted(sources, key=lambda x: get_success_rate(x[0]), reverse=True)

    for name, func in sorted_sources:
        try:
            logger.debug(
                f"尝试使用 {name} (成功率: {get_success_rate(name):.2%}) 获取 {stock_code} | {trade_day} 数据..."
            )
            df = func()
            if not df.empty:
                record_success(name)
                return df
            else:
                logger.warning(f"{name} 返回空数据，切换到下一个源...")
                record_failure(name)
        except Exception as e:
            logger.warning(f"{trade_day} 使用 {name} 获取 {stock_code} 数据失败: {e}")
            record_failure(name)

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

    # 外层进度条：按交易日
    for trade_day in tqdm(trading_days_list, desc="Processing Days", unit="day"):
        # 内层进度条：按股票代码
        for stock_code in tqdm(
            all_code,
            desc=f"Processing Stocks on {trade_day}",
            unit="stock",
            leave=False,
        ):
            stock_code_str = str(stock_code).zfill(6)
            logger.debug(colored("working on:%s|%s", "blue"), stock_code_str, trade_day)
            df = fetch_stock_data_for_day(stock_code_str, trade_day)
            if df is not None and not df.empty:
                logger.info(
                    colored("%s|%s\n%s", "green"), stock_code_str, trade_day, df.head()
                )


if __name__ == "__main__":
    """
    uv run dataset/save_daily_data.py
    """
    input_date = "2025-09-21"
    re_run = False  # True,False
    run_daily_save(input_date)
