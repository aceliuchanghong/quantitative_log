import os
import sys
import pandas as pd
from iFinDPy import *
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


def thslogindemo():
    """登录函数"""
    thsLogin = THS_iFinDLogin(
        os.getenv("tonghuashun_acc"), os.getenv("tonghuashun_pwd")
    )
    print(colored(f"iFinDLogin code:{thsLogin}", "light_yellow"))
    if thsLogin in {0, -201}:
        print("iFinDLogin 登录成功")
    else:
        print("登录失败")


@cache_to_duckdb(db_name="stock_data.duckdb", debug=False)
def get_stock_intraday_data(stock_code: str, target_date: str) -> pd.DataFrame:
    """
    获取指定股票在特定日期的当日高频细节数据 (1分钟粒度)

    :param stock_code: 股票代码, 如 "603678.SH"
    :param target_date: 目标日期, 格式为 "YYYY-MM-DD"
    :return: 包含当日交易数据的 pandas DataFrame
    """
    # 构造查询的时间范围
    start_time = f"{target_date} 09:15:00"
    end_time = f"{target_date} 15:15:00"

    # 定义需要的指标
    indicators = "open;high;low;close;volume;avgPrice;turnoverRatio;changeRatio"
    params = "CPS:forward1,Fill:Original"  # 前复权，保留原始值填充

    # 调用高频序列接口
    result = THS_HF(stock_code, indicators, params, start_time, end_time)

    if result.errorcode != 0:
        print(f"数据获取失败，代码: {result.errorcode}, 信息: {result.errormsg}")
        return pd.DataFrame()

    # 返回 DataFrame 数据部分
    return result.data


if __name__ == "__main__":
    """
    ### OHLCV

    1. **开盘价（Open）**：交易时段开始时的第一笔成交价格
    2. **收盘价（Close）**：交易时段结束时的最后一笔成交价格
    3. **最高价（High）**：该交易时段内达到的最高成交价格
    4. **最低价（Low）**：该交易时段内达到的最低成交价格
    5. **成交量（Volume）**：该交易时段内成交的股票数量

    - avgPrice 平均成交价格
    - turnoverRatio 换手率
    - changeRatio 涨跌幅

    复权方式（CPS） forward1：精确前复权
    缺失值填充策略（Fill） Original 不进行任何填充

    uv run z_test/deal_ths_data/ohlcv_data_core.py
    """
    thslogindemo()
    code = "603678.SH"
    this_date = "2025-12-03"

    try:
        df = get_stock_intraday_data(code, this_date)
        if not df.empty:
            print(f"成功获取 {code} 在 {this_date} 的数据，共 {len(df)} 行：")
            print(colored(f"{df.head()}", "light_yellow"))
        else:
            print("未能获取到数据，请检查日期是否为交易日或代码是否正确。")
    except Exception as e:
        print(f"程序运行出错: {e}")
