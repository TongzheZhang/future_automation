"""
数据采集模块 - AKShare 适配器
获取中国期货市场行情数据
"""
import akshare as ak
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

# ---- 缓存路径 ----
CACHE_DIR = Path(__file__).parent.parent / "storage"


def get_futures_daily(symbol: str, start_date: str = None, end_date: str = None, refresh_if_older_than: int = 24) -> pd.DataFrame:
    """
    获取期货主力合约日线数据
    
    Args:
        symbol: 合约代码, 如 'RB0', 'I0', 'CU0'
        start_date: 开始日期 'YYYYMMDD'
        end_date: 结束日期 'YYYYMMDD'
        refresh_if_older_than: 缓存最大允许年龄（小时），超过则重新获取
    
    Returns:
        DataFrame with columns: date, open, high, low, close, volume, open_interest
    """
    cache_file = CACHE_DIR / f"{symbol}_daily.parquet"
    
    # 尝试读缓存，但检查时效性
    if cache_file.exists():
        cache_age_hours = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() / 3600
        if cache_age_hours < refresh_if_older_than:
            df = pd.read_parquet(cache_file)
            df['date'] = pd.to_datetime(df['date'])
            # 应用日期过滤
            if start_date:
                df = df[df['date'] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df['date'] <= pd.Timestamp(end_date)]
            return df
        logger.info(f"  {symbol} 缓存已过期 ({cache_age_hours:.1f}h > {refresh_if_older_than}h), 重新获取...")
    
    try:
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        
        logger.info(f"  获取 {symbol} 日线数据...")
        df = ak.futures_main_sina(symbol=symbol)
        
        if df is None or df.empty:
            logger.warning(f"  {symbol} 数据为空")
            return pd.DataFrame()
        
        # AKShare 实际返回的列名映射
        col_mapping = {}
        for col in df.columns:
            if '日期' in str(col):
                col_mapping[col] = 'date'
            elif '开盘' in str(col):
                col_mapping[col] = 'open'
            elif '最高' in str(col):
                col_mapping[col] = 'high'
            elif '最低' in str(col):
                col_mapping[col] = 'low'
            elif '收盘' in str(col):
                col_mapping[col] = 'close'
            elif '成交' in str(col):
                col_mapping[col] = 'volume'
            elif '持仓' in str(col):
                col_mapping[col] = 'open_interest'
        
        df = df.rename(columns=col_mapping)
        
        # 确保列存在
        for col in ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest']:
            if col not in df.columns:
                df[col] = None
        
        df['date'] = pd.to_datetime(df['date'])
        df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest']].copy()
        df = df.dropna(subset=['close'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # 缓存
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file, index=False)
        
        logger.info(f"  {symbol}: {len(df)} 条日线数据")
        return df
        
    except Exception as e:
        logger.error(f"  获取 {symbol} 失败: {e}")
        return pd.DataFrame()


def get_futures_minute(symbol: str, period: str = "5") -> pd.DataFrame:
    """
    获取期货分钟线数据
    
    Args:
        symbol: 合约代码
        period: 周期 '1', '5', '15', '30', '60'
    """
    cache_file = CACHE_DIR / f"{symbol}_{period}min.parquet"
    
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    
    try:
        logger.info(f"  获取 {symbol} {period}分钟数据...")
        df = ak.futures_main_sina_minute(symbol=symbol, period=period)
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        # 根据实际返回的列名做映射
        col_map = {
            '日期': 'datetime', '时间': 'datetime',
            '开盘价': 'open', '开盘': 'open',
            '最高价': 'high', '最高': 'high',
            '最低价': 'low', '最低': 'low',
            '收盘价': 'close', '收盘': 'close',
            '成交量': 'volume', '成交': 'volume',
            '持仓量': 'open_interest', '持仓': 'open_interest',
        }
        
        # 智能映射
        rename_map = {}
        for old_col in df.columns:
            for pattern, new_col in col_map.items():
                if pattern in str(old_col):
                    rename_map[old_col] = new_col
                    break
        
        df = df.rename(columns=rename_map)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        df.to_parquet(cache_file, index=False)
        logger.info(f"  {symbol}: {len(df)} 条{period}分钟数据")
        return df
        
    except Exception as e:
        logger.error(f"  获取 {symbol} 分钟数据失败: {e}")
        return pd.DataFrame()


def get_all_contracts_data(contracts: list, use_cache: bool = True) -> dict:
    """
    批量获取所有合约日线数据
    
    Returns:
        {symbol: DataFrame}
    """
    if not use_cache:
        for c in contracts:
            cache_file = CACHE_DIR / f"{c['akshare_symbol']}_daily.parquet"
            if cache_file.exists():
                cache_file.unlink()
    
    result = {}
    for c in contracts:
        df = get_futures_daily(c['akshare_symbol'])
        if not df.empty:
            result[c['symbol']] = df
    return result