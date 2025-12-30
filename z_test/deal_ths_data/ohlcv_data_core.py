import os
import sys
import pandas as pd
import asyncio
from iFinDPy import *
from dotenv import load_dotenv
from termcolor import colored
import time

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
)
from z_utils.logging_config import get_logger
from z_utils.db_cache import cache_to_duckdb, cache_to_duckdb_async

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


@cache_to_duckdb_async(db_name="stock_data.duckdb", debug=False)
async def get_stock_intraday_data_async(
    stock_code: str, target_date: str
) -> pd.DataFrame:
    """
    异步获取指定股票在特定日期的当日高频细节数据
    """
    # 1. 构造参数
    start_time = f"{target_date} 09:15:00"
    end_time = f"{target_date} 15:15:00"
    indicators = "open;high;low;close;volume;avgPrice;turnoverRatio;changeRatio"
    params = "CPS:forward1,Fill:Original"

    # 2. 使用 to_thread 将阻塞的 SDK 调用放入线程池执行
    # 这样不会阻塞 asyncio 的事件循环
    try:
        result = await asyncio.to_thread(
            THS_HF, stock_code, indicators, params, start_time, end_time
        )

        if result.errorcode != 0:
            print(f"数据获取失败 [{stock_code}]: {result.errmsg}")
            return pd.DataFrame()

        return result.data
    except Exception as e:
        print(f"异步请求发生异常 [{stock_code}]: {e}")
        return pd.DataFrame()


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
    # print(colored(f"result:{result}", "light_yellow"))

    if result.errorcode != 0:
        print(f"数据获取失败，代码: {result.errorcode}, 信息: {result.errmsg}")
        return pd.DataFrame()

    # 返回 DataFrame 数据部分
    return result.data


@cache_to_duckdb_async(db_name="stock_data_range.duckdb")
async def get_stock_range_data_async(
    stock_code: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    异步获取指定股票在 start_date 与 end_date 之中的1分钟细节数据
    """
    # 1. 构造查询的时间范围
    start_time = f"{start_date} 09:15:00"
    end_time = f"{end_date} 15:15:00"

    # 2. 定义需要的指标
    indicators = "open;high;low;close;volume;avgPrice;turnoverRatio;changeRatio"
    params = "CPS:forward1,Fill:Original"  # 前复权，保留原始值填充

    # 3. 异步调用
    try:
        result = await asyncio.to_thread(
            THS_HF, stock_code, indicators, params, start_time, end_time
        )

        if result.errorcode != 0:
            print(f"数据获取失败 [{stock_code}]: {result.errmsg}")
            return pd.DataFrame()

        return result.data

    except Exception as e:
        print(f"异步请求发生异常 [{stock_code}]: {e}")
        return pd.DataFrame()


@cache_to_duckdb(db_name="stock_data_range.duckdb", debug=False)
def get_stock_range_data(
    stock_code: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    获取指定股票在 start_date 与 end_date  之中的1分钟细节数据

    :param stock_code: 股票代码, 如 "603678.SH"
    :param start_date: 开始日期, 格式为 "YYYY-MM-DD"
    :param end_date: 结束日期, 格式为 "YYYY-MM-DD"
    :return: 交易数据的 pandas DataFrame
    """
    # 构造查询的时间范围
    start_time = f"{start_date} 09:15:00"
    end_time = f"{end_date} 15:15:00"

    # 定义需要的指标
    indicators = "open;high;low;close;volume;avgPrice;turnoverRatio;changeRatio"
    params = "CPS:forward1,Fill:Original"  # 前复权，保留原始值填充

    # 调用高频序列接口
    result = THS_HF(stock_code, indicators, params, start_time, end_time)
    # print(colored(f"result:{result}", "light_yellow"))

    if result.errorcode != 0:
        print(f"数据获取失败，代码: {result.errorcode}, 信息: {result.errmsg}")
        return pd.DataFrame()

    # 返回 DataFrame 数据部分
    return result.data


async def test_single_and_multiple_stocks():
    target_date = "2023-12-22"
    stocks = ["603678.SH", "000001.SZ", "600519.SH"]

    print(f"--- 开始测试异步获取 {len(stocks)} 只股票数据 ---")
    start_time = time.perf_counter()

    # 1. 创建任务列表
    tasks = [get_stock_intraday_data_async(code, target_date) for code in stocks]

    # 2. 并发执行并等待结果
    results = await asyncio.gather(*tasks)

    end_time = time.perf_counter()
    print(f"--- 测试完成，耗时: {end_time - start_time:.2f} 秒 ---\n")

    # 3. 验证结果
    for i, df in enumerate(results):
        stock_code = stocks[i]
        if not df.empty:
            print(f"✅ {stock_code}: 获取成功，行数: {len(df)}")
            print(df.head(2))  # 查看前两行
        else:
            print(f"❌ {stock_code}: 获取失败或无数据")


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

    # code = "603678.SH"
    # this_date = "2025-12-03"
    # try:
    #     df = get_stock_intraday_data(code, this_date)
    #     if not df.empty:
    #         print(f"成功获取 {code} 在 {this_date} 的数据，共 {len(df)} 行：")
    #         print(colored(f"{df.head()}", "light_yellow"))
    #     else:
    #         print("未能获取到数据，请检查日期是否为交易日或代码是否正确。")
    # except Exception as e:
    #     print(f"程序运行出错: {e}")

    # try:
    #     asyncio.run(test_single_and_multiple_stocks())
    # except KeyboardInterrupt:
    #     pass

    start_date = "2016-01-11"
    end_date = "2016-12-31"
    code = "603678.SH"
    file_path = "no_git_oic/test/00.csv"
    df = get_stock_range_data(code, start_date, end_date)
    df.to_csv(file_path, index=False, encoding="utf-8")
