import os
from functools import wraps


def set_proxy(ip: str = "127.0.0.1", port: int = 10808):
    """
    设置和清理代理的装饰器工厂

    直接使用 @set_proxy() 来应用默认代理，
    或者使用 @set_proxy(ip="ip", port=port) 来指定自定义代理。

    被装饰的函数支持传入 `no_proxy=True` 来临时禁用代理。

    Args:
        ip (str, optional): 代理服务器的 IP 地址。默认为 "127.0.0.1"
        port (int, optional): 代理服务器的端口号。默认为 10808
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 检查是否传入了 no_proxy=True
            no_proxy = kwargs.pop("no_proxy", False)  # 默认不禁用代理

            if no_proxy:
                # 不设置代理，直接执行函数
                return func(*args, **kwargs)

            # 否则设置代理
            proxy_url = f"http://{ip}:{port}"
            original_http_proxy = os.environ.get("HTTP_PROXY")
            original_https_proxy = os.environ.get("HTTPS_PROXY")

            os.environ["HTTP_PROXY"] = proxy_url
            os.environ["HTTPS_PROXY"] = proxy_url

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # 恢复原始环境变量
                if original_http_proxy is not None:
                    os.environ["HTTP_PROXY"] = original_http_proxy
                else:
                    os.environ.pop("HTTP_PROXY", None)  # 安全删除

                if original_https_proxy is not None:
                    os.environ["HTTPS_PROXY"] = original_https_proxy
                else:
                    os.environ.pop("HTTPS_PROXY", None)

        return wrapper

    return decorator
