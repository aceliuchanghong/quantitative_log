import paramiko
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)
from z_utils.logging_config import get_logger

logger = get_logger(__name__)


def ssh_execute_command(
    hostname,
    username,
    password=None,
    key_filename=None,
    port=22,
    command="",
    timeout=10,
):
    """
    通过 SSH 执行远程命令

    :param hostname: 远程主机 IP 或域名
    :param username: 登录用户名
    :param password: 密码
    :param key_filename: 私钥文件路径
    :param port: SSH 端口，默认 22
    :param command: 要执行的命令（字符串）
    :param timeout: 连接和命令执行超时时间（秒）
    :return: (stdout_str, stderr_str, exit_status)
             stdout_str: 命令标准输出（字符串）
             stderr_str: 命令标准错误（字符串）
             exit_status: 命令退出状态码 0 表示成功
    """
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(
            hostname=hostname,
            port=port,
            username=username,
            password=password,
            key_filename=key_filename,
            timeout=timeout,
        )
        logger.info(f"Connected to {hostname}")

        stdin, stdout, stderr = client.exec_command(command, timeout=timeout)

        stdout_str = stdout.read().decode("utf-8").strip()
        stderr_str = stderr.read().decode("utf-8").strip()
        exit_status = stdout.channel.recv_exit_status()

        logger.info(f"Command executed with exit status: {exit_status}")
        return stdout_str, stderr_str, exit_status

    except Exception as e:
        logger.error(f"SSH 执行命令失败: {e}")
        return "", str(e), -1

    finally:
        client.close()
        logger.info("SSH connection closed.")


if __name__ == "__main__":
    host = "192.168.1.100"
    user = "username"
    pwd = "password"

    cmd = "uname -a && whoami"

    stdout, stderr, exit_code = ssh_execute_command(
        hostname=host, username=user, password=pwd, command=cmd
    )

    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    print("EXIT CODE:", exit_code)
