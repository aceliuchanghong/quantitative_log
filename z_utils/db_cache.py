import os
import duckdb
import hashlib
import json
import pandas as pd
import functools
import inspect
import sys
import threading
from dotenv import load_dotenv
from io import StringIO
from pathlib import Path
from datetime import datetime, date
from termcolor import colored
from typing import Any, NamedTuple, Optional, Tuple, Dict


sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)

from z_utils.logging_config import get_logger

load_dotenv()
logger = get_logger(__name__)

# --- 模块1: 序列化与反序列化 ---
# 所有数据格式转换的逻辑都封装在这里。


class CustomJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，用于处理特殊类型（日期、路径等）。"""

    def default(self, obj):
        if isinstance(obj, (datetime, date, pd.Timestamp)):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)  # 兜底转换为字符串


class Serializer:
    """负责将Python对象序列化为可存储的格式，以及反序列化。"""

    @staticmethod
    def serialize(data: Any) -> Tuple[str, str, int]:
        """
        序列化数据。

        Returns:
            Tuple[str, str, int]: (序列化后的字符串, 数据类型, 数据量)
        """
        if isinstance(data, pd.DataFrame):
            return (
                data.to_json(orient="split", date_format="iso"),
                "pd_dataframe",
                len(data),
            )

        elif isinstance(data, pd.Series):
            # Series 转换为 JSON 并在反序列化时还原
            return (
                data.to_json(orient="split", date_format="iso"),
                "pd_series",
                len(data),
            )

        elif isinstance(data, pd.Index):
            # 处理 DatetimeIndex 等索引类型
            # 先转成 Series 以复用 split 格式，保留名称和类型
            s = pd.Series(data)
            return s.to_json(orient="split", date_format="iso"), "pd_index", len(data)

        else:
            # 通用 JSON 类型
            try:
                serialized = json.dumps(data, default=Serializer._custom_encoder)
                count = len(data) if isinstance(data, (list, dict, str)) else 1
                return serialized, "json", count
            except Exception as e:
                # 兜底：如果 JSON 失败，强转字符串
                return str(data), "string", 1

    @staticmethod
    def deserialize(serialized_data: str, data_type: str) -> Any:
        """根据记录的类型还原对象。"""
        if not serialized_data:
            return None

        if data_type == "pd_dataframe":
            return pd.read_json(StringIO(serialized_data), orient="split")

        elif data_type == "pd_series":
            return pd.read_json(StringIO(serialized_data), orient="split", typ="series")

        elif data_type == "pd_index":
            # 还原为 Index
            s = pd.read_json(StringIO(serialized_data), orient="split", typ="series")
            return pd.Index(s)

        elif data_type == "json":
            return json.loads(serialized_data)

        return serialized_data

    @staticmethod
    def _custom_encoder(obj):
        """处理 JSON 无法识别的日期等类型。"""
        if isinstance(obj, (datetime, date, pd.Timestamp)):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# --- 模块2: DuckDB 数据库管理 ---
# 所有与数据库的交互都封装在此类中。


class CacheResult(NamedTuple):
    """用于封装从缓存中查询到的结果的结构体。"""

    data: Any
    count: int


class DuckDBCacheManager:
    """封装所有与DuckDB缓存相关的操作。"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._initialize_schema()

    def _get_connection(self):
        """获取数据库连接。"""
        try:
            return duckdb.connect(database=self.db_path, read_only=False)
        except Exception as e:
            logger.error(
                colored("%s", "red"), f"连接DuckDB数据库 {self.db_path} 失败: {e}"
            )
            raise

    def _initialize_schema(self):
        """初始化数据库表结构。如果表不存在，则创建。"""
        with self._get_connection() as conn:
            conn.execute("CREATE SEQUENCE IF NOT EXISTS cache_id_seq;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS function_cache (
                    id UBIGINT PRIMARY KEY DEFAULT nextval('cache_id_seq'),
                    function_name VARCHAR NOT NULL,
                    params_hash VARCHAR NOT NULL,
                    result_data VARCHAR NOT NULL,
                    data_count UINTEGER NOT NULL,
                    data_type VARCHAR NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    UNIQUE(function_name, params_hash)
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_func_params 
                ON function_cache (function_name, params_hash);
                """
            )

    def query(
        self, function_name: str, params_hash: str, debug: bool = False
    ) -> Optional[CacheResult]:
        """
        从缓存中查询数据。

        Returns:
            Optional[CacheResult]: 如果命中缓存则返回CacheResult，否则返回None。
        """
        with self._get_connection() as conn:
            result = conn.execute(
                """
                SELECT result_data, data_count, data_type FROM function_cache
                WHERE function_name = ? AND params_hash = ?
                """,
                [function_name, params_hash],
            ).fetchone()

        if result:
            if debug:
                logger.info(colored("%s", "blue"), f"函数 {function_name} 缓存命中。")
            result_data_str, data_count, data_type = result
            deserialized_data = Serializer.deserialize(result_data_str, data_type)
            return CacheResult(data=deserialized_data, count=data_count)

        if debug:
            logger.info(colored("%s", "blue"), f"函数 {function_name} 缓存未命中。")
        return None

    def save(
        self,
        function_name: str,
        params_hash: str,
        result_data: Any,
        debug: bool = False,
    ):
        """
        将函数执行结果保存到缓存。
        """
        serialized_data, data_type, data_count = Serializer.serialize(result_data)
        with self._lock:  # 添加锁
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO function_cache (function_name, params_hash, result_data, data_count, data_type, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (function_name, params_hash) 
                    DO UPDATE SET
                        result_data = EXCLUDED.result_data,
                        data_count = EXCLUDED.data_count,
                        data_type = EXCLUDED.data_type,
                        created_at = EXCLUDED.created_at;
                    """,
                    (
                        function_name,
                        params_hash,
                        serialized_data,
                        data_count,
                        data_type,
                        datetime.now(),
                    ),
                )
        logger.debug(
            colored("函数 %s 执行记录已保存到数据库", "light_yellow"), function_name
        )


# --- 模块3: 装饰器工厂与辅助函数 ---
# 缓存的核心控制逻辑，如参数处理、重跑判断等。


def _get_params_hash(params: Dict[str, Any]) -> str:
    """根据参数字典生成唯一的MD5哈希值。"""
    params_str = json.dumps(params, sort_keys=True, cls=CustomJSONEncoder)
    return hashlib.md5(params_str.encode()).hexdigest()


def _prepare_cache_key_and_rerun_flag(
    func: callable, args: Tuple, kwargs: Dict, re_run_decorator: bool
) -> Tuple[str, str, Dict, bool, str]:
    """
    解析参数、生成缓存键，并判断是否需要强制重跑。

    Returns:
        Tuple: (函数名, 参数哈希, 清理后的参数字典, 是否强制重跑, 重跑原因)
    """
    function_name = func.__name__

    # 提取并移除 _re_run 控制参数
    force_rerun_arg = kwargs.pop("_re_run", False)

    # 解析 args 和 kwargs，合并为统一的 params 字典
    params = kwargs.copy()
    sig = inspect.signature(func)
    func_param_names = list(sig.parameters.keys())

    args_to_process = list(args)
    if args_to_process and func_param_names:
        first_param_name = func_param_names[0]
        if first_param_name in ("self", "cls"):
            args_to_process.pop(0)
            data_arg_names = func_param_names[1 : len(args_to_process) + 1]
        else:
            data_arg_names = func_param_names[: len(args_to_process)]
        params.update(dict(zip(data_arg_names, args_to_process)))

    # 判断是否需要强制重跑
    rerun_reason = ""
    if force_rerun_arg:
        skip_cache = True
        rerun_reason = "调用时传入 _re_run=True"
    elif re_run_decorator:
        skip_cache = True
        rerun_reason = "装饰器 re_run=True"
    else:
        skip_cache = False

    params_hash = _get_params_hash(params)

    return function_name, params_hash, params, skip_cache, rerun_reason


def _is_result_empty(result: Any) -> bool:
    """判断结果是否为空, None, 空DataFrame, 空集合等"""
    if result is None:
        return True
    if isinstance(result, pd.DataFrame) and result.empty:
        return True
    if isinstance(result, (list, dict, str)) and not result:
        return True
    return False


def create_cache_decorator(is_async: bool):
    """
    装饰器工厂，根据 is_async 参数创建同步或异步的缓存装饰器。
    将所有模块组合起来。
    """

    def decorator(
        db_name="cache.duckdb", default_return=None, debug=False, re_run=False
    ):

        cache_manager = DuckDBCacheManager(db_name)

        def outer_wrapper(func):
            if is_async:
                # --- 异步版本 ---
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    try:
                        name, p_hash, params, skip, reason = (
                            _prepare_cache_key_and_rerun_flag(
                                func, args, kwargs, re_run
                            )
                        )
                        if not skip:
                            cached = cache_manager.query(name, p_hash, debug)
                            if cached:
                                return cached.data
                        elif debug:
                            logger.info(
                                colored("%s", "blue"),
                                f"跳过缓存，强制重跑 {name} ({reason})",
                            )

                        result = await func(*args, **kwargs)

                        if _is_result_empty(result):
                            if debug:
                                logger.info(
                                    colored("%s", "blue"),
                                    f"{name} 返回空结果，不缓存。",
                                )
                            return result

                        cache_manager.save(name, p_hash, result, debug)
                        return result
                    except Exception as e:
                        logger.error(
                            colored("函数 %s 执行失败: %s", "red"), func.__name__, e
                        )
                        raise

                return async_wrapper
            else:
                # --- 同步版本 ---
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    try:
                        name, p_hash, params, skip, reason = (
                            _prepare_cache_key_and_rerun_flag(
                                func, args, kwargs, re_run
                            )
                        )
                        if not skip:
                            cached = cache_manager.query(name, p_hash, debug)
                            if cached:
                                return cached.data
                        elif debug:
                            logger.info(
                                colored("%s", "blue"),
                                f"跳过缓存，强制重跑 {name} ({reason})",
                            )

                        result = func(*args, **kwargs)

                        if _is_result_empty(result):
                            if debug:
                                logger.info(
                                    colored("%s", "blue"),
                                    f"{name} 返回空结果，不缓存。",
                                )
                            return result

                        cache_manager.save(name, p_hash, result, debug)
                        return result
                    except Exception as e:
                        logger.error(
                            colored("函数 %s 执行失败: %s", "red"), func.__name__, e
                        )
                        raise

                return sync_wrapper

        return outer_wrapper

    return decorator


# --- 最终导出的接口 ---
# 通过工厂模式生成我们需要的两个装饰器。
cache_to_duckdb = create_cache_decorator(is_async=False)
cache_to_duckdb_async = create_cache_decorator(is_async=True)

if __name__ == "__main__":

    @cache_to_duckdb(debug=True)
    def get_test_data(stock_code, start_date, end_date, adjust="qfq"):

        print(f"no cache")
        return {
            "stock_code": stock_code,
            "start_date": start_date,
            "end_date": end_date,
            "adjust": adjust,
        }

    @cache_to_duckdb_async()
    async def get_test_data_async(stock_code, start_date, end_date, adjust="qfq"):
        print(f"no async cache")
        return {
            "stock_code": stock_code,
            "start_date": start_date,
            "end_date": end_date,
            "adjust": adjust,
        }

    stock_code = "603678"
    start_data, end_data = "2025-07-23", "2025-07-23"
    adjust = "qfq"
    adjust2 = "hfq"
    re_run = True

    import time
    import asyncio

    start_time = time.time()

    df = get_test_data(
        stock_code,
        start_data.replace("-", ""),
        end_data.replace("-", ""),
        adjust,
        _re_run=re_run,
    )
    df2 = get_test_data(
        stock_code, start_data.replace("-", ""), end_data.replace("-", ""), adjust2
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(colored(f"耗时: {elapsed_time:.2f}秒", "light_yellow"))
    print(colored(f"{df}", "light_yellow"))
    print(colored(f"{df2}", "light_yellow"))

    # 测试同步

    start_time = time.time()

    df = asyncio.run(
        get_test_data_async(
            stock_code,
            start_data.replace("-", ""),
            end_data.replace("-", ""),
            adjust,
            _re_run=re_run,
        )
    )
    df2 = asyncio.run(
        get_test_data_async(
            stock_code, start_data.replace("-", ""), end_data.replace("-", ""), adjust2
        )
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(colored(f"耗时: {elapsed_time:.2f}秒", "light_yellow"))
    print(colored(f"{df}", "light_yellow"))
    print(colored(f"{df2}", "light_yellow"))

    # 测试多并发场景
    @cache_to_duckdb(debug=True, re_run=True)
    def add(a, b):
        time.sleep(1)  # 增加 sleep，模拟更长的计算，放大并发冲突概率
        print(f"Executing add({a}, {b}) in thread {threading.current_thread().name}")
        return a + b

    def worker(thread_id):
        try:
            result = add(1, 2)  # 相同参数，易触发冲突
            print(f"Thread {thread_id}: result = {result}")
        except Exception as e:
            print(f"Thread {thread_id} error: {e}")

    threads = [
        threading.Thread(target=worker, args=(i,), name=f"T{i}") for i in range(20)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 测试错误函数
    @cache_to_duckdb()
    def test_error_fun(name: str) -> str:
        try:
            names = name + 1
            return "hello " + names
        except Exception as e:
            logger.error(colored("%s", "red"), e)
            raise

    print(f"{test_error_fun('llch', _re_run=True)}")

    """
    uv run z_utils/db_cache.py
    """
