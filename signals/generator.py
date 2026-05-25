"""
每日信号生成器

基于日内回测结果，为下一个交易日生成交易信号
"""
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date, timedelta
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.collectors.akshare_adapter import get_futures_daily
from backtest.intraday_engine import load_intraday_data, run_intraday_backtest


# 最优参数配置 (来自回测扫描结果)
OPTIMAL_CONFIGS = {
    'TA': {'gap_threshold': 0.005, 'hold_minutes': 45, 'stop_loss': 0.01, 'confidence': 0.75},
    'SC': {'gap_threshold': 0.003, 'hold_minutes': 45, 'stop_loss': 0.01, 'confidence': 0.70},
    'LH': {'gap_threshold': 0.003, 'hold_minutes': 90, 'stop_loss': 0.015, 'confidence': 0.65},
    'CU': {'gap_threshold': 0.005, 'hold_minutes': 30, 'stop_loss': 0.01, 'confidence': 0.60},
    'AG': {'gap_threshold': 0.005, 'hold_minutes': 30, 'stop_loss': 0.01, 'confidence': 0.55},
    'RB': {'gap_threshold': 0.008, 'hold_minutes': 30, 'stop_loss': 0.015, 'confidence': 0.55},
    'I':  {'gap_threshold': 0.008, 'hold_minutes': 30, 'stop_loss': 0.015, 'confidence': 0.55},
    'SA': {'gap_threshold': 0.005, 'hold_minutes': 45, 'stop_loss': 0.015, 'confidence': 0.55},
}


from data.utils import get_contract_code


def generate_daily_signals(contracts_config: list) -> list:
    """
    为下一个交易日生成信号
    
    检查每个品种的最新日线数据，如果有符合条件的缺口，生成信号
    """
    signals = []
    today = date.today()
    
    for c in contracts_config:
        symbol = c['symbol']
        akshare_sym = c['akshare_symbol']
        
        # 只处理有最优参数的品种
        if symbol not in OPTIMAL_CONFIGS:
            continue
        
        opt = OPTIMAL_CONFIGS[symbol]
        
        # 获取日线数据
        df_daily = get_futures_daily(akshare_sym)
        if df_daily.empty or len(df_daily) < 2:
            logger.warning(f"  {symbol} 日线数据不足")
            continue
        
        # 计算最新缺口
        prev = df_daily.iloc[-2]  # 上一个交易日
        today_row = df_daily.iloc[-1]  # 可能是今天(如果已收盘)
        
        prev_close = prev['close']
        today_close = today_row['close']  # 或今日最新价
        
        if pd.isna(prev_close) or pd.isna(today_close) or prev_close == 0:
            continue
        
        gap_pct = (today_close - prev_close) / prev_close
        
        # 检查缺口是否达到阈值
        if abs(gap_pct) < opt['gap_threshold']:
            continue
        
        # 生成信号
        direction = 'short' if gap_pct > 0 else 'long'
        
        entry_price = today_close  # 基于收盘价预估明日开盘位置
        
        if direction == 'long':
            stop_loss = round(entry_price * (1 - opt['stop_loss']), 2)
            take_profit = round(entry_price * (1 + opt['stop_loss'] * 1.5), 2)
            direction_cn = '做多'
        else:
            stop_loss = round(entry_price * (1 + opt['stop_loss']), 2)
            take_profit = round(entry_price * (1 - opt['stop_loss'] * 1.5), 2)
            direction_cn = '做空'
        
        exit_time_str = f"入场后{opt['hold_minutes']}分钟"
        
        signals.append({
            'symbol': symbol,
            'name': c['name'],
            'direction': direction_cn,
            'direction_code': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': opt['confidence'],
            'hold_minutes': opt['hold_minutes'],
            'exit_time': exit_time_str,
            'gap_pct': f"{gap_pct:+.2%}",
            'reason': f"收盘价{today_close} vs 前收{prev_close}，缺口{gap_pct:+.2%}，预计回补",
        })
    
    return signals


def format_signal_report(signals: list) -> str:
    """格式化信号报告(Markdown)"""
    now = datetime.now()
    
    if not signals:
        return f"""# 🔔 期货日内信号日报
**{now.strftime('%Y-%m-%d')}**  
    
暂无符合条件的交易信号。

> 系统将持续监控，下一个交易日盘前再次扫描。
"""
    
    # 按置信度排序
    signals = sorted(signals, key=lambda x: x['confidence'], reverse=True)
    
    lines = [
        f"# 🔔 期货日内信号日报",
        f"**{now.strftime('%Y-%m-%d %H:%M')}**  |  {len(signals)} 个信号",
        "",
        "---",
        "",
    ]
    
    for i, s in enumerate(signals):
        conf_emoji = '🔥' if s['confidence'] >= 0.7 else '⭐' if s['confidence'] >= 0.6 else '💡'
        
        lines.extend([
            f"### {conf_emoji} 信号 {i+1}: {s['name']}({s['symbol']})",
            "",
            f"| 项目 | 内容 |",
            f"|------|------|",
            f"| **操作** | **{s['direction']}** |",
            f"| **参考入场价** | {s['entry_price']} |",
            f"| **止损价** | {s['stop_loss']} |",
            f"| **止盈价** | {s['take_profit']} |",
            f"| **持有时间** | {s['exit_time']} |",
            f"| **置信度** | {s['confidence']:.0%} |",
            f"| **当前缺口** | {s['gap_pct']} |",
            f"| **逻辑** | {s['reason']} |",
            "",
            "---",
            "",
        ])
    
    lines.extend([
        "## ⚠️ 风险提示",
        "",
        "- 本信号为 AI 策略生成，仅供参考",
        "- 请结合自身判断决定是否执行",
        "- 严格执行止损，控制单笔风险",
        "",
        "*Spark 自进化投研系统 · 自动生成*",
    ])
    
    return '\n'.join(lines)


if __name__ == "__main__":
    # 测试
    with open(Path(__file__).parent.parent / "config" / "contracts.yaml") as f:
        contracts = yaml.safe_load(f)['contracts']
    
    signals = generate_daily_signals(contracts)
    report = format_signal_report(signals)
    print(report)