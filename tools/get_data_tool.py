import os
import sys
from dotenv import load_dotenv
from termcolor import colored
import pandas as pd
from datetime import datetime

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)

from z_utils.logging_config import get_logger
from z_utils.db_cache import cache_to_duckdb
from z_utils.get_proxy import set_proxy
from tools.fix_code_tool import get_ticker

load_dotenv()
logger = get_logger(__name__)


@cache_to_duckdb()
def get_intraday_data_ak(symbol, date, period="1", adjust="qfq"):
    """
    获取指定股票在指定日期的分钟级数据

    参数:
        symbol: 股票代码，如 "000001"（平安银行）
        date: 日期字符串，如 "2024-05-10"
        period: 分钟周期，支持 "1", "5", "15", "30", "60"
        adjust: 复权类型，""（不复权），"qfq"（前复权），"hfq"（后复权）

    返回:
        pandas.DataFrame 包含分钟级数据
    """
    import akshare as ak

    # 验证和格式化日期
    request_dt = pd.to_datetime(date)
    days_diff = (datetime.now().date() - request_dt.date()).days

    if period == "1" and days_diff > 6:
        logger.warning(
            colored(
                "分钟级别数据('1m')通常只能获取最近 6 天。请求的日期 %s 可能超出范围。",
                "yellow",
            ),
            request_dt.strftime("%Y-%m-%d"),
        )

    start_time = f"{date} 09:30:00"
    end_time = f"{date} 15:00:00"

    df = ak.stock_zh_a_hist_min_em(
        symbol=symbol,
        period=period,
        adjust=adjust,
        start_date=start_time,
        end_date=end_time,
    )

    # 列重命名映射
    column_mapping = {
        "时间": "datetime",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        # '均价': 可选，这里移除
    }

    # 选择并重命名列
    df = df.rename(columns=column_mapping)
    df = df[["datetime", "open", "high", "low", "close", "volume", "amount"]]

    # 添加股票代码列
    df["stock_code"] = symbol

    # 调整列顺序为期望格式
    df = df[
        ["datetime", "stock_code", "open", "high", "low", "close", "volume", "amount"]
    ]

    return df


@cache_to_duckdb()
@set_proxy()
def get_intraday_data_yf(
    stock_code: str, input_date: str, interval: str = "1m"
) -> pd.DataFrame:
    """
    使用yfinance获取单只A股在指定某一天的分钟级别行情数据

    open	开盘价	在这一分钟的第一笔成交价格。
    close	收盘价	在这一分钟的最后一笔成交价格。这也是绘制分时走势线时通常使用的价格点
    volume	成交量	在这一分钟内，总共买卖的股票数量。单位是“股”
    amount	成交额	在这一分钟内，总共买卖的股票金额。单位是“元”。它约等于 价格 * 成交量

    Args:
        stock_code (str): 6位数字的A股股票代码。
        input_date (str): 希望获取数据的日期, 格式 'YYYY-MM-DD' 或 'YYYYMMDD'。
        interval (str): 数据间隔, e.g., '1m', '5m', '15m', '30m', '60m'。

    Returns:
        pd.DataFrame: 包含格式化后核心分钟数据的DataFrame 列包括：
                      ['datetime', 'stock_code', 'open', 'high', 'low', 'close', 'volume', 'amount']
                      如果获取失败则返回空的DataFrame。
    """
    import yfinance as yf

    # 验证和格式化日期
    request_dt = pd.to_datetime(input_date)
    days_diff = (datetime.now(request_dt.tz) - request_dt).days

    # 检查日期是否符合 yfinance 的限制
    if interval == "1m" and days_diff > 7:
        logger.warning(
            colored(
                "分钟级别数据('1m')通常只能获取最近 7 天。请求的日期 %s 可能超出范围。",
                "yellow",
            ),
            request_dt.strftime("%Y-%m-%d"),
        )

    # 准备API调用的开始和结束日期
    start_date_fmt = request_dt.strftime("%Y-%m-%d")
    end_date_fmt = (request_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # 获取适用于yfinance的ticker
    ticker = get_ticker(stock_code)

    # 从yfinance下载分钟级别数据
    df = yf.download(
        tickers=ticker,
        start=start_date_fmt,
        end=end_date_fmt,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )
    # 压平 MultiIndex 列名
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    logger.debug(colored("\n%s", "blue"), df)

    if df.empty:
        logger.warning(
            colored("未获取到股票 %s:%s 在 %s 的分钟数据", "yellow"),
            stock_code,
            ticker,
            start_date_fmt,
        )
        return pd.DataFrame()

    # 数据清洗和格式化
    df.reset_index(inplace=True)
    df.rename(
        columns={
            "Datetime": "datetime",
            "index": "datetime",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        },
        inplace=True,
    )

    # 仅保留所需的核心列
    core_cols = ["datetime", "open", "high", "low", "close", "volume"]
    df = df[core_cols]

    # 数据类型处理和计算成交金额
    price_cols = ["open", "high", "low", "close"]
    df[price_cols] = df[price_cols].round(2)  # A股价格通常保留两位小数
    df["volume"] = df["volume"].astype(int)

    # 计算估算成交金额 (Amount) = 收盘价 * 成交量
    df["amount"] = (df["close"] * df["volume"]).round(0).astype(int)

    # 添加股票代码并格式化时间
    df["stock_code"] = stock_code
    # 将datetime对象转换为标准字符串格式，便于查看或存储
    df["datetime"] = (df["datetime"] + pd.Timedelta(hours=8)).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # 调整列的顺序并返回
    output_columns = [
        "datetime",
        "stock_code",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    return df[output_columns]


if __name__ == "__main__":
    """
    uv run tools/get_data_tool.py
    """
    stock_code = "603678"
    input_date = "2025-09-19"
    re_run = False  # True,False
    yf_result = get_intraday_data_yf(stock_code, input_date, _re_run=re_run)
    logger.info(colored("\n%s", "green"), yf_result)

    input_date = "2025-09-15"
    ak_result = get_intraday_data_ak(stock_code, input_date, _re_run=re_run)
    logger.info(colored("\n%s", "green"), ak_result)
