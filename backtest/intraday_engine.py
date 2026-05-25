"""
日内分钟级回测引擎

使用真实的5分钟K线数据做精确回测。
不再用日线近似模拟，直接验证"开盘买入→N分钟后卖出"的收益。
"""
import pandas as pd
import numpy as np
from datetime import datetime, time
from pathlib import Path
from loguru import logger

# 期货交易时间(简化版)
TRADING_SESSIONS = [
    (time(9, 0), time(10, 15)),    # 早盘第一节
    (time(10, 30), time(11, 30)),  # 早盘第二节
    (time(13, 30), time(15, 0)),   # 下午盘
    # 夜盘(21:00-23:00或凌晨1:00/2:30) - 暂不处理,先做日盘信号
]


def load_intraday_data(symbol: str, contract_code: str, period: str = '5') -> pd.DataFrame:
    """
    加载日内分钟数据
    
    Args:
        symbol: 品种代码 (RB, I, CU...)
        contract_code: 具体合约 (RB2605, I2605...)
        period: 周期 '5', '15', '30', '60'
    """
    import akshare as ak
    
    cache_file = Path(__file__).parent.parent / "data" / "minute_storage" / f"{contract_code}_{period}min.parquet"
    
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    
    try:
        logger.info(f"  下载 {contract_code} {period}分钟数据...")
        df = ak.futures_zh_minute_sina(symbol=contract_code, period=period)
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        # 列名标准化
        df = df.rename(columns={
            'datetime': 'datetime',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'hold': 'open_interest',
        })
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file, index=False)
        
        logger.info(f"  {contract_code}: {len(df)} 条{period}分钟数据 ({df['datetime'].min()} ~ {df['datetime'].max()})")
        return df
        
    except Exception as e:
        logger.error(f"  下载 {contract_code} 失败: {e}")
        return pd.DataFrame()


def run_intraday_backtest(
    df: pd.DataFrame,
    contract_name: str,
    symbol: str = None,
    hold_bars: int = 6,          # 持有K线数 (6*5=30分钟)
    entry_bar: int = 0,           # 从第几根K线入场 (0=第一根5分钟K线, 即9:00-9:05)
    direction: str = 'auto',      # 'long', 'short', 'auto'
    gap_threshold: float = 0.003, # 缺口阈值(日线级别,用于auto模式)
    stop_loss_pct: float = 0.01,
    capital: float = 1000000.0,
    position_pct: float = 0.3,
) -> dict:
    """
    日内分钟级回测
    
    对每一天的交易时段，执行: 入场 → 持有N根K线 → 出场
    
    Returns:
        dict with trades list and summary stats
    """
    if df.empty:
        return {'trades': [], 'total_pnl': 0, 'sharpe': 0}
    
    # 只取日盘数据 (9:00-15:00)
    df = df.copy()
    df['date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.time
    
    # 每天按时间排，取每天的前N根K线
    daily_groups = df.groupby('date')
    
    # 获取前一日的日线收盘价(用于计算缺口)
    daily_closes = {}
    for date, group in daily_groups:
        daily_closes[date] = group['close'].iloc[-1]
    
    dates = sorted(daily_closes.keys())
    prev_close = None
    
    trades = []
    
    for i, date in enumerate(dates):
        day_data = df[df['date'] == date].reset_index(drop=True)
        
        if len(day_data) < entry_bar + hold_bars + 1:
            continue
        
        # 入场K线
        entry = day_data.iloc[entry_bar]
        entry_price = entry['open']
        entry_time = entry['datetime']
        
        # 出场K线
        exit_idx = entry_bar + hold_bars
        if exit_idx >= len(day_data):
            continue
        exit_bar = day_data.iloc[exit_idx]
        exit_price = exit_bar['open']  # 用下一根K线的开盘价出场
        
        # 止损检查: 在持有期间遍历
        stop_triggered = False
        stop_price = None
        sl_long = entry_price * (1 - stop_loss_pct)
        sl_short = entry_price * (1 + stop_loss_pct)
        
        for j in range(entry_bar + 1, exit_idx + 1):
            bar = day_data.iloc[j]
            if direction == 'long' and bar['low'] <= sl_long:
                stop_triggered = True
                stop_price = sl_long
                break
            elif direction == 'short' and bar['high'] >= sl_short:
                stop_triggered = True
                stop_price = sl_short
                break
        
        # 确定交易方向
        if direction == 'auto':
            if prev_close is not None:
                gap = (entry_price - prev_close) / prev_close
                if abs(gap) < gap_threshold:
                    prev_close = daily_closes.get(date)
                    continue  # 缺口太小不交易
                trade_dir = 'short' if gap > 0 else 'long'
            else:
                prev_close = daily_closes.get(date)
                continue
        else:
            trade_dir = direction
        
        # 计算PNL
        if stop_triggered:
            final_exit = stop_price
            exit_reason = 'SL'
        else:
            final_exit = exit_price
            exit_reason = 'Exit'
        
        if trade_dir == 'long':
            pnl_pct = (final_exit - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - final_exit) / entry_price
        
        # 计算实际金额（按标准期货盈亏公式：手数 × 合约乘数 × 价格变动）
        if symbol:
            from backtest.engine_v2 import get_spec
            spec = get_spec(symbol)
            multiplier = spec["multiplier"]
        else:
            multiplier = 10
        
        position_value = capital * position_pct
        lots = max(1, int(position_value / (entry_price * multiplier)))
        price_change = final_exit - entry_price if trade_dir == 'long' else entry_price - final_exit
        pnl_amount = price_change * lots * multiplier
        
        trades.append({
            'date': date,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_price': final_exit,
            'direction': trade_dir,
            'pnl_pct': pnl_pct,
            'pnl_amount': pnl_amount,
            'exit_reason': exit_reason,
            'prev_close': prev_close,
        })
        
        capital += pnl_amount
        prev_close = daily_closes.get(date)
    
    if not trades:
        return {
            'trades': [],
            'total_pnl': 0,
            'total_pnl_pct': 0,
            'sharpe': 0,
            'win_rate': 0,
            'num_trades': 0,
            'max_dd': 0,
        }
    
    # 统计
    pnls = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    
    total_pnl_pct = np.prod([1 + p for p in pnls]) - 1
    sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252) if len(pnls) > 1 and np.std(pnls) > 0 else 0
    win_rate = len(wins) / len(pnls)
    
    # 资金曲线最大回撤
    cum_pnl = np.cumprod([1 + p for p in pnls])
    peak = np.maximum.accumulate(cum_pnl)
    dd = (cum_pnl - peak) / peak
    max_dd = np.min(dd)
    
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    
    return {
        'contract': contract_name,
        'num_trades': len(trades),
        'total_pnl_pct': total_pnl_pct,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float('inf'),
        'trades': trades,
    }


def scan_all_intraday(
    contracts: list,
    hold_minutes_list: list = [30, 45, 60, 90],
    gap_threshold_list: list = [0.003, 0.005, 0.008],
) -> pd.DataFrame:
    """
    扫描所有品种+参数组合,找出最佳日内策略
    
    Returns:
        DataFrame: 所有参数组合的回测结果排名
    """
    results = []
    
    for c in contracts:
        symbol = c['symbol']
        # 构建合约代码
        from data.utils import get_contract_code_with_skip
        skip_symbols = {'M', 'RM', 'A', 'B', 'Y', 'P', 'OI', 'C', 'CS', 'JD', 'L', 'PP', 'V'}
        contract_code = get_contract_code_with_skip(symbol, skip_01_symbols=skip_symbols)
        
        # 加载数据
        df = load_intraday_data(symbol, contract_code, '5')
        if df.empty:
            logger.warning(f"  {contract_code} 无数据,跳过")
            continue
        
        for hold_min in hold_minutes_list:
            hold_bars = hold_min // 5
            
            for gap in gap_threshold_list:
                direction_modes = ['auto']  # 用缺口自动判断方向
                
                for dir_mode in direction_modes:
                    result = run_intraday_backtest(
                        df, 
                        contract_name=f"{symbol}({contract_code})",
                        symbol=symbol,
                        hold_bars=hold_bars,
                        direction=dir_mode,
                        gap_threshold=gap,
                        stop_loss_pct=0.01,
                    )
                    
                    if result['num_trades'] == 0:
                        continue
                    
                    results.append({
                        'contract': symbol,
                        'code': contract_code,
                        'direction': dir_mode,
                        'hold_min': hold_min,
                        'gap_thr': gap,
                        'num_trades': result['num_trades'],
                        'total_pnl': f"{result['total_pnl_pct']:+.2%}",
                        'sharpe': round(result['sharpe'], 2),
                        'win_rate': f"{result['win_rate']:.1%}",
                        'max_dd': f"{result['max_dd']:.2%}",
                        'profit_factor': round(result['profit_factor'], 2),
                        'avg_win': f"{result['avg_win']:+.2%}",
                        'avg_loss': f"{result['avg_loss']:+.2%}",
                        'total_pnl_val': result['total_pnl_pct'],
                    })
    
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values('sharpe', ascending=False)
    
    return df_results