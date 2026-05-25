"""
分钟级数据下载 + 日内精确回测

目的: 验证 "开盘入场 → 持有N分钟出场" 在真实分钟K线上的表现
     与日线 OHLC 近似模拟做对比

数据源: akshare.futures_zh_minute_sina
限制: API 只返回近 2-3 周的分钟数据，需要周期性累积

用法:
  python intraday_bt_v3.py           # 下载数据 + 回测
  python intraday_bt_v3.py --download # 仅下载(用于定期累积)
"""
import sys
from pathlib import Path
from datetime import datetime, date, time, timedelta
from collections import defaultdict

import akshare as ak
import pandas as pd
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from signals.factors import create_v3_scorer
from data.collectors.akshare_adapter import get_futures_daily

# ---- 配置 ----
MINUTE_DIR = Path(__file__).parent / "data" / "minute_storage"
MINUTE_DIR.mkdir(parents=True, exist_ok=True)

# 精选品种（v3 最佳配置）
SELECTED_SYMBOLS = ["CU", "TA", "P", "RB", "AG", "SC", "I", "MA", "SA"]

# 止损/止盈参数（与 v3 回测一致）
ATR_STOP_MULT = 2.0
ATR_TP_MULT = 3.0
HOLD_MINUTES = 45
HOLD_BARS = HOLD_MINUTES // 5  # 5分钟K线, 持有9根

# 每日开盘时间
MORNING_OPEN = time(9, 0)
AFTERNOON_OPEN = time(13, 30)
NIGHT_OPEN = time(21, 0)


from data.utils import get_contract_code


def download_minute_data(symbols: list[str] = None) -> dict:
    """下载所有品种的5分钟K线数据"""
    if symbols is None:
        symbols = SELECTED_SYMBOLS
    
    data = {}
    today = date.today()
    
    for sym in symbols:
        contract = get_contract_code(sym, today)
        cache_file = MINUTE_DIR / f"{sym}_5min.parquet"
        
        # 尝试读缓存
        cached = None
        if cache_file.exists():
            cached = pd.read_parquet(cache_file)
            cached['datetime'] = pd.to_datetime(cached['datetime'])
            logger.info(f"  {sym}({contract}): 缓存 {len(cached)} 条")
        
        # 下载新数据
        new_data = None
        for attempt in range(3):
            try:
                df = ak.futures_zh_minute_sina(symbol=contract, period='5')
                if df is not None and not df.empty:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    # 标准化列名
                    col_map = {}
                    for col in df.columns:
                        cl = col.lower()
                        if 'datetime' in cl:
                            col_map[col] = 'datetime'
                        elif 'open' in cl:
                            col_map[col] = 'open'
                        elif 'high' in cl:
                            col_map[col] = 'high'
                        elif 'low' in cl:
                            col_map[col] = 'low'
                        elif 'close' in cl:
                            col_map[col] = 'close'
                        elif 'volume' in cl or 'vol' in cl:
                            col_map[col] = 'volume'
                        elif 'hold' in cl:
                            col_map[col] = 'open_interest'
                    df = df.rename(columns=col_map)
                    new_data = df
                    break
            except Exception as e:
                logger.warning(f"  {sym}({contract}) 下载尝试 {attempt+1}/3 失败: {e}")
                import time as _time
                _time.sleep(1)
        
        # 合并缓存和新数据
        if cached is not None and new_data is not None:
            combined = pd.concat([cached, new_data]).drop_duplicates(subset=['datetime']).sort_values('datetime')
        elif new_data is not None:
            combined = new_data
        elif cached is not None:
            combined = cached
        else:
            logger.warning(f"  {sym}({contract}): 无数据")
            continue
        
        combined.to_parquet(cache_file, index=False)
        data[sym] = combined
        logger.info(f"  {sym}({contract}): {len(combined)} 条 | "
                     f"{combined['datetime'].min()} ~ {combined['datetime'].max()}")
    
    return data


def compute_atr_intraday(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """在分钟数据上计算 ATR"""
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def run_intraday_backtest(
    minute_data: dict[str, pd.DataFrame],
    daily_data: dict[str, pd.DataFrame],
) -> list[dict]:
    """
    真正的日内回测:
      - 每天开盘: 计算昨日收盘→今日开盘缺口
      - 如果缺口达到阈值 → 入场 (第1根5min K线的开盘价)
      - 持有 N 根K线 → 出场
      - 期间检查止损/止盈
    
    与日线模拟对比: 记录每日的 "真实出场价 vs 日线近似出场价"
    """
    trades = []
    scorer = create_v3_scorer()
    
    for sym, df_min in minute_data.items():
        if df_min.empty or sym not in daily_data:
            continue
        
        df_daily = daily_data[sym].copy()
        df_daily = df_daily.sort_values('date').reset_index(drop=True)
        
        if df_daily.empty or len(df_daily) < 14:
            continue
        
        # 分钟数据按天分组
        df_min = df_min.copy()
        df_min['date'] = df_min['datetime'].dt.date
        df_min['time'] = df_min['datetime'].dt.time
        
        dates = sorted(df_min['date'].unique())
        
        prev_close = None
        
        for d in dates:
            day_bars = df_min[df_min['date'] == d].reset_index(drop=True)
            if len(day_bars) < HOLD_BARS + 2:
                continue
            
            # 找日线数据中前一天的收盘价
            daily_row = df_daily[df_daily['date'] == pd.Timestamp(d)]
            if daily_row.empty:
                continue
            idx = daily_row.index[0]
            if idx == 0:
                continue
            prev_row = df_daily.iloc[idx - 1]
            prev_close = prev_row['close']
            today_open_daily = daily_row['open'].iloc[0]
            
            if pd.isna(prev_close) or pd.isna(today_open_daily) or prev_close <= 0:
                continue
            
            gap_pct = (today_open_daily - prev_close) / prev_close
            
            # 多因子评分
            direction = "short" if gap_pct > 0 else "long"
            
            confidence, factor_scores = scorer.score(
                df=df_daily, index=idx, prev_close=prev_close,
                gap_pct=gap_pct, direction=direction, symbol=sym,
            )
            
            if confidence < 0.70:
                continue
            
            # ---- 入场: 第一根5分钟K线的开盘价 (模拟 9:00 抢开盘) ----
            entry_bar = day_bars.iloc[0]
            entry_price = entry_bar['open']
            
            # ---- ATR (based on daily data) ----
            atr_val = (today_open_daily * 0.015)  # 简化: 日ATR的1.5%
            
            # 止损/止盈
            stop_dist = ATR_STOP_MULT * atr_val
            tp_dist = ATR_TP_MULT * atr_val
            
            if direction == "long":
                sl = entry_price - stop_dist
                tp = entry_price + tp_dist
            else:
                sl = entry_price + stop_dist
                tp = entry_price - tp_dist
            
            # ---- 持有期: 遍历随后的 K 线 ----
            exit_price = None
            exit_reason = "exit"
            exit_bar_idx = 0
            
            for j in range(1, min(HOLD_BARS + 1, len(day_bars))):
                bar = day_bars.iloc[j]
                
                if direction == "long":
                    if bar['low'] <= sl:
                        exit_price = sl
                        exit_reason = "stop_loss"
                        exit_bar_idx = j
                        break
                    if bar['high'] >= tp:
                        exit_price = tp
                        exit_reason = "take_profit"
                        exit_bar_idx = j
                        break
                else:
                    if bar['high'] >= sl:
                        exit_price = sl
                        exit_reason = "stop_loss"
                        exit_bar_idx = j
                        break
                    if bar['low'] <= tp:
                        exit_price = tp
                        exit_reason = "take_profit"
                        exit_bar_idx = j
                        break
            
            # 未触发止盈/止损 → 按持有期满出场
            if exit_price is None:
                exit_idx = min(HOLD_BARS, len(day_bars) - 1)
                exit_bar = day_bars.iloc[exit_idx]
                exit_price = exit_bar['open']  # 用该K线开盘价出场
                exit_reason = "exit"
                exit_bar_idx = exit_idx
            
            # ---- PnL ----
            if direction == "long":
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price
            
            # 日线近似对比: 用日线 OHLC 模拟的结果
            # 与 v2 回测引擎一致: 用当日收盘出场
            today_close_val = daily_row['close'].iloc[0]
            today_high_val = daily_row['high'].iloc[0]
            today_low_val = daily_row['low'].iloc[0]
            
            # 日线近似 PnL
            if direction == "long":
                if today_low_val <= sl:
                    daily_exit = sl
                elif today_high_val >= tp:
                    daily_exit = tp
                else:
                    daily_exit = today_close_val
                daily_pnl_pct = (daily_exit - today_open_daily) / today_open_daily
            else:
                if today_high_val >= sl:
                    daily_exit = sl
                elif today_low_val <= tp:
                    daily_exit = tp
                else:
                    daily_exit = today_close_val
                daily_pnl_pct = (today_open_daily - daily_exit) / today_open_daily
            
            trades.append({
                'symbol': sym,
                'date': d,
                'direction': direction,
                'gap_pct': gap_pct,
                'confidence': confidence,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'hold_bars': exit_bar_idx,
                'pnl_pct_intraday': pnl_pct,
                'pnl_pct_daily_approx': daily_pnl_pct,
                'pnl_diff': pnl_pct - daily_pnl_pct,
            })
    
    return trades


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    
    logger.info("=" * 60)
    logger.info("  ⏱️ 分钟级日内回测 v3")
    logger.info("=" * 60)
    
    # 1. 下载分钟数据
    if mode in ("full", "download"):
        logger.info("📡 下载分钟数据...")
        minute_data = download_minute_data()
    else:
        # 仅从缓存加载
        minute_data = {}
        for sym in SELECTED_SYMBOLS:
            cache_file = MINUTE_DIR / f"{sym}_5min.parquet"
            if cache_file.exists():
                df = pd.read_parquet(cache_file)
                df['datetime'] = pd.to_datetime(df['datetime'])
                minute_data[sym] = df
        logger.info(f"从缓存加载 {len(minute_data)} 个品种的分钟数据")
    
    if not minute_data:
        logger.error("无分钟数据")
        return
    
    # 2. 加载日线数据(用于因子评分 + 对比)
    from data.collectors.akshare_adapter import get_all_contracts_data
    config_dir = PROJECT_ROOT / "config"
    import yaml
    with open(config_dir / "contracts.yaml") as f:
        contracts = yaml.safe_load(f)['contracts']
    daily_data = get_all_contracts_data(contracts)
    
    # 3. 运行日内回测
    logger.info("\n📊 运行日内回测...")
    trades = run_intraday_backtest(minute_data, daily_data)
    
    if not trades:
        logger.warning("无交易信号 (分钟数据范围太短)")
        return
    
    df_trades = pd.DataFrame(trades)
    
    # 4. 结果分析
    logger.info(f"\n{'='*50}")
    logger.info(f"  日内回测结果")
    logger.info(f"{'='*50}")
    logger.info(f"  交易数: {len(df_trades)}")
    logger.info(f"  日期范围: {df_trades['date'].min()} ~ {df_trades['date'].max()}")
    
    # 分方向统计
    total_pnl = df_trades['pnl_pct_intraday'].sum()
    win_rate = (df_trades['pnl_pct_intraday'] > 0).mean()
    avg_pnl = df_trades['pnl_pct_intraday'].mean()
    
    logger.info(f"  总 PnL (日内): {total_pnl:+.2%}")
    logger.info(f"  胜率: {win_rate:.1%}")
    logger.info(f"  平均单笔: {avg_pnl:+.2%}")
    
    # 出场原因
    logger.info(f"\n  出场原因:")
    for reason, count in df_trades['exit_reason'].value_counts().items():
        logger.info(f"    {reason}: {count} 笔")
    
    # 日内 vs 日线对比
    logger.info(f"\n{'='*50}")
    logger.info(f"  日内精确 vs 日线近似 对比")
    logger.info(f"{'='*50}")
    
    intraday_pnl_sum = df_trades['pnl_pct_intraday'].sum()
    daily_pnl_sum = df_trades['pnl_pct_daily_approx'].sum()
    mae = abs(df_trades['pnl_diff']).mean()
    correlation = df_trades['pnl_pct_intraday'].corr(df_trades['pnl_pct_daily_approx'])
    
    logger.info(f"  日内总 PnL: {intraday_pnl_sum:+.2%}")
    logger.info(f"  日线近似 PnL: {daily_pnl_sum:+.2%}")
    logger.info(f"  平均误差 (MAE): {mae:.2%}")
    logger.info(f"  相关性: {correlation:.3f}")
    
    # 日内更好还是更差？
    if intraday_pnl_sum > daily_pnl_sum:
        logger.info(f"  ✅ 日内精确回测结果优于日线近似 (+{intraday_pnl_sum - daily_pnl_sum:+.2%})")
    else:
        logger.info(f"  ⚠️ 日内精确回测结果差于日线近似 ({intraday_pnl_sum - daily_pnl_sum:+.2%})")
    
    # 按品种汇总
    logger.info(f"\n  按品种:")
    by_sym = df_trades.groupby('symbol').agg(
        交易数=('pnl_pct_intraday', 'count'),
        日内PnL=('pnl_pct_intraday', 'sum'),
        日线近似PnL=('pnl_pct_daily_approx', 'sum'),
        误差=('pnl_diff', 'mean'),
        胜率=('pnl_pct_intraday', lambda x: (x > 0).mean()),
    ).round(4)
    logger.info(f"\n{by_sym.to_string()}")
    
    # 保存
    out = PROJECT_ROOT / "backtest" / "reports" / f"intraday_bt_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df_trades.to_csv(out, index=False, encoding="utf-8-sig")
    logger.info(f"\n  结果已保存: {out}")
    
    return df_trades


if __name__ == "__main__":
    main()
