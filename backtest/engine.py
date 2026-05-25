"""
回测引擎 - 批量运行策略并生成报告
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from loguru import logger
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from strategies.base import StrategyResult

REPORT_DIR = Path(__file__).parent / "reports"


def run_backtest(strategy, data: dict, name: str = None) -> StrategyResult:
    """运行单个策略回测"""
    name = name or strategy.name
    logger.info(f"\n{'='*50}")
    logger.info(f"  回测策略: {name}")
    logger.info(f"  描述: {strategy.description}")
    logger.info(f"{'='*50}")
    
    strategy.load_data(data)
    result = strategy.backtest()
    
    logger.info(f"\n  📊 回测结果:\n{result.summary()}")
    return result


def run_all_backtests(strategies: list, data: dict) -> dict:
    """批量运行所有策略回测"""
    results = {}
    for strategy in strategies:
        result = run_backtest(strategy, data)
        results[strategy.name] = result
    return results


def plot_equity_curves(results: dict, save_path: str = None):
    """绘制资金曲线对比"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for name, result in results.items():
        if result.equity_curve is not None and len(result.equity_curve) > 1:
            ax.plot(result.equity_curve.values, label=f"{name} (Sharpe={result.sharpe_ratio:.2f})", alpha=0.8)
    
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('策略资金曲线对比', fontsize=14)
    ax.set_xlabel('交易日')
    ax.set_ylabel('净值')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"  图表已保存: {save_path}")
    else:
        path = REPORT_DIR / f"equity_curves_{datetime.now().strftime('%Y%m%d')}.png"
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        logger.info(f"  图表已保存: {path}")
    
    plt.close()


def generate_report(results: dict, output_path: str = None) -> str:
    """生成回测报告"""
    lines = [
        f"# 期货策略回测报告",
        f"## 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "| 策略 | 总收益 | 年化收益 | 夏普 | 最大回撤 | 胜率 | 交易次数 | 盈亏比 |",
        "|------|--------|----------|------|----------|------|----------|--------|",
    ]
    
    for name, r in sorted(results.items(), key=lambda x: x[1].sharpe_ratio, reverse=True):
        lines.append(
            f"| {name} | {r.total_return:+.2%} | {r.annual_return:+.2%} | "
            f"{r.sharpe_ratio:.2f} | {r.max_drawdown:.2%} | {r.win_rate:.1%} | "
            f"{r.total_trades} | {r.profit_factor:.2f} |"
        )
    
    lines.extend([
        "",
        "---",
        "*Spark 自进化投研系统 · 自动生成*",
    ])
    
    report = '\n'.join(lines)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report)
    else:
        p = REPORT_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        p.write_text(report)
        output_path = str(p)
    
    logger.info(f"  报告已生成: {output_path}")
    return report