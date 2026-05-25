"""
数据工具函数

提供合约代码映射、日期处理等公共功能。
"""
from datetime import date


def get_contract_code(symbol: str, target_date: date = None) -> str:
    """
    根据品种和日期获取主力合约代码。

    中国期货市场主力合约月份规则：1月、5月、9月（多数品种）。

    Args:
        symbol: 品种代码，如 'RB', 'I', 'CU'
        target_date: 目标日期，默认今天

    Returns:
        合约代码，如 'RB2605', 'I2609'
    """
    if target_date is None:
        target_date = date.today()

    month = target_date.month
    year = target_date.year % 100

    # 主力合约月映射
    if month <= 1:
        main_month = "01"
    elif month <= 5:
        main_month = "05"
    elif month <= 9:
        main_month = "09"
    else:
        main_month = "01"
        year = (year + 1) % 100

    return f"{symbol}{year:02d}{main_month}"


def get_contract_code_with_skip(
    symbol: str,
    target_date: date = None,
    skip_01_symbols: set = None,
) -> str:
    """
    获取主力合约代码，支持特定品种跳过01月合约。

    部分品种（如豆粕 M、菜粕 RM 等）主力合约跳过1月，
    在10-12月期间直接映射到次年5月。

    Args:
        symbol: 品种代码
        target_date: 目标日期
        skip_01_symbols: 需要跳过01月的品种集合

    Returns:
        合约代码
    """
    code = get_contract_code(symbol, target_date)

    if skip_01_symbols and symbol in skip_01_symbols:
        # 如果生成的是01月合约且当前在10月之后，跳转到05月
        if code[-2:] == "01":
            year = int(code[-4:-2])
            year = (year + 1) % 100
            return f"{symbol}{year:02d}05"

    return code
