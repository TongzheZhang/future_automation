"""
进化引擎 - 策略自我进化模块

机制:
1. 定时分析策略表现
2. 当策略失效时，自动搜索新的参数组合或策略变体
3. 通过回测验证新策略
4. 优胜劣汰，更新策略库
"""
import sys
import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date, timedelta
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.intraday_engine import load_intraday_data, run_intraday_backtest, scan_all_intraday
from data.collectors.akshare_adapter import get_futures_daily


EVOLUTION_DIR = Path(__file__).parent
STATE_FILE = EVOLUTION_DIR / "evolution_state.json"


def load_evolution_state() -> dict:
    """加载进化状态"""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        'version': 1,
        'last_evolution': None,
        'active_strategies': {},
        'archive': [],
        'performance_log': [],
    }


def save_evolution_state(state: dict):
    """保存进化状态"""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def analyze_strategy_performance(symbol: str, contract_code: str, config: dict) -> dict:
    """分析单个策略的近期表现"""
    df = load_intraday_data(symbol, contract_code, '5')
    if df.empty:
        return {'status': 'no_data'}
    
    result = run_intraday_backtest(
        df, contract_name=f"{symbol}({contract_code})",
        hold_bars=config.get('hold_minutes', 45) // 5,
        direction='auto',
        gap_threshold=config.get('gap_threshold', 0.005),
        stop_loss_pct=config.get('stop_loss', 0.01),
    )
    
    return {
        'status': 'ok',
        'num_trades': result['num_trades'],
        'total_pnl': result['total_pnl_pct'],
        'sharpe': result['sharpe'],
        'win_rate': result['win_rate'],
        'profit_factor': result['profit_factor'],
    }


def evolve_strategy(symbol: str, contracts_config: list) -> dict:
    """
    进化策略: 扫描参数空间，找到更优配置
    
    Returns:
        新的最优配置
    """
    logger.info(f"🧬 进化 {symbol} 策略...")
    
    c_info = next((c for c in contracts_config if c['symbol'] == symbol), None)
    if c_info is None:
        return None
    
    # 大范围扫描参数
    df = scan_all_intraday(
        [c_info],
        hold_minutes_list=[15, 30, 45, 60, 90, 120],
        gap_threshold_list=[0.002, 0.003, 0.005, 0.008, 0.01, 0.015],
    )
    
    if df.empty:
        logger.warning(f"  {symbol} 无有效回测结果")
        return None
    
    # 转换 win_rate 为数值（兼容字符串百分比和数值形式）
    if df['win_rate'].dtype == object:
        df['win_rate_num'] = df['win_rate'].str.rstrip('%').astype(float) / 100
    else:
        df['win_rate_num'] = df['win_rate']
    
    # 找最优 (综合排序: Sharpe > 收益 > 胜率)
    df['score'] = df['sharpe'] * 0.5 + df['total_pnl_val'] * 10 * 0.3 + df['win_rate_num'] * 0.2
    best = df.iloc[0]
    
    new_config = {
        'gap_threshold': best['gap_thr'],
        'hold_minutes': best['hold_min'],
        'stop_loss': 0.01,
        'confidence': min(best['win_rate_num'], 0.85),
        'sharpe': best['sharpe'],
        'total_pnl': best['total_pnl'],
        'num_trades': best['num_trades'],
    }
    
    logger.info(f"  ✅ 新配置: gap≥{new_config['gap_threshold']}, hold={new_config['hold_minutes']}min, "
                f"Sharpe={new_config['sharpe']}, 收益={new_config['total_pnl']}")
    
    return new_config


def run_evolution_cycle(contracts_config: list):
    """
    执行一次完整的进化周期
    
    1. 分析所有活跃策略表现
    2. 对表现不佳的策略尝试进化
    3. 更新策略库
    """
    state = load_evolution_state()
    now = datetime.now().isoformat()
    
    logger.info("=" * 50)
    logger.info("🧬 进化引擎启动")
    logger.info(f"  时间: {now}")
    logger.info("=" * 50)
    
    from signals.generator import OPTIMAL_CONFIGS
    
    changes = []
    
    for symbol, config in list(OPTIMAL_CONFIGS.items()):
        c_info = next((c for c in contracts_config if c['symbol'] == symbol), None)
        if c_info is None:
            continue
        
        contract_code = f"{symbol}{datetime.now().year % 100:02d}05"
        
        perf = analyze_strategy_performance(symbol, contract_code, config)
        
        state['performance_log'].append({
            'time': now,
            'symbol': symbol,
            'sharpe': perf.get('sharpe', 0),
            'win_rate': perf.get('win_rate', 0),
        })
        
        # 保留最近100条记录
        if len(state['performance_log']) > 500:
            state['performance_log'] = state['performance_log'][-500:]
        
        # 判断是否需要进化: Sharpe < 0 或 近期无交易
        if perf.get('sharpe', 0) < 0 or perf.get('num_trades', 0) == 0:
            logger.info(f"  🔄 {symbol} 策略需要进化 (Sharpe={perf.get('sharpe', 0):.2f})")
            
            # 存档旧策略
            state['archive'].append({
                'symbol': symbol,
                'config': config,
                'replaced_at': now,
                'last_perf': perf,
            })
            
            # 进化
            new_config = evolve_strategy(symbol, contracts_config)
            if new_config:
                OPTIMAL_CONFIGS[symbol] = {
                    'gap_threshold': new_config['gap_threshold'],
                    'hold_minutes': new_config['hold_minutes'],
                    'stop_loss': 0.01,
                    'confidence': new_config['confidence'],
                }
                changes.append(symbol)
    
    state['last_evolution'] = now
    state['version'] += 1
    
    save_evolution_state(state)
    
    if changes:
        logger.info(f"\n✅ 进化完成: 更新了 {len(changes)} 个策略: {', '.join(changes)}")
    else:
        logger.info("\n✅ 所有策略表现良好，无需进化")
    
    return changes


if __name__ == "__main__":
    with open(Path(__file__).parent.parent / "config" / "contracts.yaml") as f:
        contracts = yaml.safe_load(f)['contracts']
    run_evolution_cycle(contracts)