import duckdb
import os
from dotenv import load_dotenv
from termcolor import colored

load_dotenv()

DB_PATH = os.getenv("DUCKDB_PATH")
_duckdb_conn_instance = None


def _get_duckdb_conn(db_path):
    """内部函数：创建并返回一个新的 DuckDB 连接"""
    return duckdb.connect(db_path)


def get_shared_duckdb_conn(db_path=DB_PATH):
    """获取全局共享的 DuckDB 连接实例"""
    global _duckdb_conn_instance
    if _duckdb_conn_instance is None:
        _duckdb_conn_instance = _get_duckdb_conn(db_path)
    return _duckdb_conn_instance


def close_shared_duckdb_conn():
    global _duckdb_conn_instance
    if _duckdb_conn_instance is not None:
        _duckdb_conn_instance.close()
        _duckdb_conn_instance = None


if __name__ == "__main__":
    """
    uv run tools/duckdb_client.py
    """
    conn = get_shared_duckdb_conn()

    result = conn.execute("SHOW TABLES").fetchall()

    print(f"表数量: {len(result)}")
    if result:
        print(f"表名示例: {[t[0] for t in result]}")

    if result:
        first_table_name = result[0][0]  # 获取第一个表名
        print(colored(f"\n正在查看表结构: {first_table_name}", "cyan"))

        # 1. 查看表结构
        schema_result = conn.execute(f"DESCRIBE {first_table_name}").fetchdf()
        print(colored("\n📊 表结构:", "green"))
        print(schema_result)

        # 2. 查询前5条数据
        print(colored(f"\n🔍 查询 {first_table_name} 前5条数据:", "yellow"))
        data_df = conn.execute(f"SELECT * FROM {first_table_name} LIMIT 5").fetchdf()
        print(data_df)

    else:
        print(colored("⚠️  没有找到任何表", "red"))
