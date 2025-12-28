import os
import sys
import akshare as ak
import pandas as pd
from dotenv import load_dotenv
from termcolor import colored

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
)
from z_utils.logging_config import get_logger
from z_utils.db_cache import cache_to_duckdb

load_dotenv()
logger = get_logger(__name__)


@cache_to_duckdb(debug=False)
def get_trading_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    使用 akshare 获取指定日期区间内的所有 A 股交易日。
    """
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
    except Exception as e:
        raise ValueError(f"日期格式无效: {e}")

    if start > end:
        raise ValueError("起始日期不能晚于结束日期。")

    try:
        trade_dates_df = ak.tool_trade_date_hist_sina()
        all_trading_days = pd.to_datetime(trade_dates_df["trade_date"])
    except Exception as e:
        raise RuntimeError(f"获取交易日历失败: {e}")

    # 筛选并返回
    mask = (all_trading_days >= start) & (all_trading_days <= end)
    selected_days = all_trading_days[mask].sort_values()

    return pd.DatetimeIndex(selected_days)


if __name__ == "__main__":
    """
    uv run z_test/deal_ths_data/get_trade_date.py
    """
    start_date = "2000-01-01"
    end_date = "2025-12-26"
    trading_days = get_trading_days(start_date, end_date)
    print(trading_days)

    date_list = trading_days.sort_values().strftime("%Y-%m-%d").tolist()
    logger.info(
        colored("%s:%s|%s", "green"), type(date_list), date_list[:2], date_list[-2:]
    )
