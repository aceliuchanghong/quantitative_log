import logging
import os
import re
import sys
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from threading import Lock

try:
    from concurrent_log_handler import ConcurrentRotatingFileHandler
except ImportError:
    ConcurrentRotatingFileHandler = None
    print(
        "警告: 'concurrent-log-handler' 未安装, 日志轮转在多进程环境下可能不安全。"
        "请运行 'pip install concurrent-log-handler' 进行安装。",
        file=sys.stderr,
    )


load_dotenv()


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = (
    "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s"
)
LOG_FILE = os.getenv("LOG_FILE")

LOG_MAX_SIZE = int(os.getenv("LOG_MAX_SIZE", 1)) * 1024 * 1024  # 默认 1MB
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", 50))

# 用于确保日志初始化线程安全的锁
_lock = Lock()
# 记录已配置的 logger 名称，避免重复配置
_configured_loggers = {}


class NoColorFormatter(logging.Formatter):
    """移除 ANSI 颜色码，确保日志文件内容干净"""

    ANSI_ESCAPE = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")

    def format(self, record):
        # 创建记录的副本，以避免影响其他处理器
        record_copy = logging.makeLogRecord(record.__dict__)
        # 调用父类的 format 方法获取原始格式化后的消息
        message = super().format(record_copy)
        # 移除 ANSI 颜色码
        return self.ANSI_ESCAPE.sub("", message)


def _setup_logger(name: str):
    """
    创建并配置一个 logger 实例。
    此函数是线程安全的。
    """
    if name in _configured_loggers:
        return _configured_loggers[name]

    with _lock:
        # 双重检查锁定，防止多线程重复配置
        if name in _configured_loggers:
            return _configured_loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        # 设置 propagate 为 False，避免日志向根 logger 传播导致重复输出
        logger.propagate = False

        # --- 格式化器 ---
        console_formatter = logging.Formatter(LOG_FORMAT)
        file_formatter = NoColorFormatter(LOG_FORMAT)

        # --- 控制台处理器 (StreamHandler) ---
        console_handler = logging.StreamHandler(sys.stdout)
        # 控制台可以显示更详细的日志，例如 DEBUG 级别
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # --- 文件处理器 (ConcurrentRotatingFileHandler) ---
        if LOG_FILE:
            log_dir = os.path.dirname(LOG_FILE)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            # 优先使用进程安全的处理器
            HandlerClass = (
                ConcurrentRotatingFileHandler
                if ConcurrentRotatingFileHandler
                else RotatingFileHandler
            )

            file_handler = HandlerClass(
                LOG_FILE,
                maxBytes=LOG_MAX_SIZE,
                backupCount=LOG_BACKUP_COUNT,
                encoding="utf-8",
                # 对于 ConcurrentRotatingFileHandler, delay 不是必需的，但保留也无妨
                delay=True,
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)

            # 修正 handleError 的签名
            def custom_error_handler(record):
                """处理日志写入期间的错误"""
                # 提供更有用的错误信息
                logging.error(f"日志文件写入失败！记录: {record.msg}", exc_info=True)

            file_handler.handleError = custom_error_handler

            logger.addHandler(file_handler)
        else:
            logger.warning("环境变量 'LOG_FILE' 未设置，日志将不会写入文件。")

        _configured_loggers[name] = logger
        return logger


def get_logger(name: str = None):
    """
    获取一个命名的 logger。
    """
    if name is None:
        import inspect

        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        name = module.__name__ if module else "__main__"

    return _setup_logger(name)


if __name__ == "__main__":
    """
    uv run z_utils/logging_config.py
    """
    logger = get_logger(__name__)
    logger.info("打印 INFO 日志")
    logger.debug("打印 DEBUG 日志")

    def test(name="qaq"):
        logger.debug(name)

    test()
