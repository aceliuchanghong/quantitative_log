from iFinDPy import *


def thslogindemo():
    # 输入用户的帐号和密码
    print(thsLogin)
    if thsLogin in {0, -201}:
        print("登录成功")
    else:
        print("登录失败")


def run():
    # 历史行情-收盘价;最低价;最高价;开盘价;成交额-iFinD数据接口
    result = THS_HD(
        "000001.SH", "close;low;high;open;amt", "CPS:2", "2024-12-05", "2025-12-05"
    )
    print(f"{type(result)},{result}")
    x = THS_HF(
        "603678.SH",
        "open;high;low;close;volume;avgPrice;turnoverRatio;changeRatio",
        "CPS:forward1,Fill:Original",
        "2025-12-03 09:15:00",
        "2025-12-05 15:15:00",
    )
    print(f"{type(x)}")
    print(f"{x}")


if __name__ == "__main__":
    """
    uv run z_test/ths02.py
    """
    thslogindemo()

    run()
