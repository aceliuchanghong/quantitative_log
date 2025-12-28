import os
import sys
import baostock as bs
import pandas as pd
from dotenv import load_dotenv
from termcolor import colored

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
)
from z_utils.logging_config import get_logger
from z_utils.db_cache import cache_to_duckdb
from z_test.deal_ths_data.get_trade_date import get_trading_days

load_dotenv()
logger = get_logger(__name__)


@cache_to_duckdb(debug=False)
def get_historical_codes_baostock_core(target_date: str) -> list:
    """
    通过 baostock 获取指定日期的 A 股截面全量代码
    """
    rs = bs.query_all_stock(day=target_date)
    stock_list = []
    while (rs.error_code == "0") & rs.next():
        stock_list.append(rs.get_row_data()[0])

    clean_codes = [
        code.split(".")[-1] for code in stock_list if code.startswith(("sh", "sz"))
    ]
    return clean_codes


def save_to_csv(start_date, end_date, save_dir) -> bool:
    os.makedirs(save_dir, exist_ok=True)

    trading_days = get_trading_days(start_date, end_date)
    date_list = trading_days.sort_values().strftime("%Y-%m-%d").tolist()

    # 统一登录
    lg = bs.login()
    if lg.error_code != "0":
        logger.error(f"baostock login failed: {lg.error_msg}")
        return False

    try:
        logger.info(f"开始批量获取数据，共 {len(date_list)} 天...")

        for date_str in date_list:
            codes = get_historical_codes_baostock_core(date_str)
            logger.info(
                colored("日期: %s | 数量: %d | 样本: %s", "green"),
                date_str,
                len(codes),
                codes[:2],
            )
            if codes:
                file_path = os.path.join(save_dir, f"{date_str}.csv")
                df = pd.DataFrame(codes, columns=["code"])
                df.to_csv(file_path, index=False, encoding="utf-8")

                # logger.info(f"已保存: {file_path}")
            else:
                logger.warning(f"日期 {date_str} 数据为空，跳过保存")
        return True

    except Exception as e:
        logger.error(f"循环处理时发生异常: {e}")
        return False
    finally:
        bs.logout()
        logger.info("BaoStock 会话已关闭")


if __name__ == "__main__":
    """
    uv pip install baostock

    uv run z_test/deal_ths_data/get_trade_code.py
    """
    # try:
    #     lg = bs.login()
    #     if lg.error_code != "0":
    #         logger.error(f"baostock login failed: {lg.error_msg}")

    #     trade_date = "2000-01-05"
    #     codes = get_historical_codes_baostock_core(trade_date)
    #     logger.info(colored("%s:%s", "green"), len(codes), codes[:5])

    # except Exception as e:
    #     logger.error(f"登录发生异常: {e}")
    # finally:
    #     bs.logout()

    start_date = "2000-01-01"
    end_date = "2025-12-26"
    save_dir = "no_git_oic/historical_codes"
    save_to_csv(start_date, end_date, save_dir)
