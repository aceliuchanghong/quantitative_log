import logging
import os
import re
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

load_dotenv()

# 加载环境变量
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = (
    "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s"
)
LOG_FILE = os.getenv("LOG_FILE")

LOG_MAX_SIZE = int(os.getenv("LOG_MAX_SIZE")) * 1024 * 1024
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT"))


class NoColorFormatter(logging.Formatter):
    """移除 ANSI 颜色码"""

    ANSI_ESCAPE = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")

    def format(self, record):
        original = super().format(record)
        return self.ANSI_ESCAPE.sub("", original)


def _setup_logger(name: str = __name__):
    """创建并配置一个 logger 实例"""
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL))

    # 格式化器
    formatter = logging.Formatter(LOG_FORMAT)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # 文件处理器
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=LOG_MAX_SIZE, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler.setFormatter(NoColorFormatter(LOG_FORMAT))

    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_logger(name=None):
    """获取一个命名的 logger"""
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
