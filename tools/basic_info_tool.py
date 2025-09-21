import akshare as ak
import pandas as pd
from termcolor import colored
import os
import sys
from dotenv import load_dotenv

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)

from z_utils.logging_config import get_logger
from z_utils.db_cache import cache_to_duckdb

load_dotenv()
logger = get_logger(__name__)


@cache_to_duckdb()
def get_trading_days(start_date, end_date):
    """获取指定日期范围内的交易日列表"""
    try:
        start_dt = pd.to_datetime(start_date, format="%Y-%m-%d")
        end_dt = pd.to_datetime(end_date, format="%Y-%m-%d")
    except ValueError:
        raise ValueError("日期格式错误，请使用 'YYYY-MM-DD' 格式，如 '2025-07-30'")

    calendar = ak.stock_zh_index_daily(symbol="sh000001")
    calendar["date"] = pd.to_datetime(calendar["date"])

    # 筛选指定日期范围
    mask = (calendar["date"] >= start_dt) & (calendar["date"] <= end_dt)
    trading_days = calendar.loc[mask, "date"]

    # 转换为 'YYYY-MM-DD' 字符串格式的列表
    return trading_days.dt.strftime("%Y-%m-%d").tolist()


@cache_to_duckdb()
def get_a_share_stock_list(input_date):
    """
    获取当前A股所有股票的代码、名称和最新价

    Returns:
        pandas.DataFrame: 包含 '代码-Symbol', '名称-Name', '最新价-Latest_Price' 列的 DataFrame
    """
    try:
        # 获取A股实时行情数据
        stock_df = ak.stock_zh_a_spot_em()
        # 选择需要的列
        result_df = stock_df[["代码", "名称", "最新价"]].copy()
        result_df = result_df[result_df["最新价"].notna()]
        result_df = result_df.rename(
            columns={"代码": "Symbol", "名称": "Name", "最新价": "Latest_Price"}
        )
        # 补齐股票代码到6位
        result_df["Symbol"] = result_df["Symbol"].astype(str).str.zfill(6)

        return result_df
    except Exception as e:
        print(f"获取数据时发生错误: {e}")
        raise


if __name__ == "__main__":
    """
    uv run tools/basic_info_tool.py
    """
    start_date = "2025-07-31"
    end_date = "2025-09-30"
    input_date = "2025-09-21"
    re_run = False  # True,False
    trade_date = get_trading_days(start_date, end_date)
    logger.info(colored("%s", "green"), trade_date)
    all_code = get_a_share_stock_list(input_date, _re_run=re_run)
    logger.info(colored("\n%s", "green"), all_code)

    for code in all_code["Symbol"]:
        logger.info(colored("%s", "green"), code)
        break
    for code_row in all_code[["Symbol", "Name"]].values:
        logger.info(colored("%s|%s", "green"), code_row[0], code_row[1])
        break
