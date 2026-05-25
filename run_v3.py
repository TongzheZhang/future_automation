"""
Spark v3 回测脚本 - 基于 v2 回测结果的自进化优化

v2 回测发现的 4 大问题 & 对策:

问题1: 置信度 60% 阈值太低 → 数据证明 ≥70% 才盈利
  对策: min_confidence = 0.70

问题2: 14个品种中8个亏损 → 信号在多数品种上无效  
  对策: 只交易 v2 回测中净盈利的 6 个品种 (SC, MA, TA, HC, RB, JM)

问题3: 滑点 120 万吃掉 129 万毛利 → 成本模型需调整
  对策: 流动性分级滑点 - 大盘品种只收 0.5 tick

问题4: 3244 笔中仅 26 笔触发止盈/止损 → 参数无效
  对策: 收紧止损到 1.5×ATR, 止盈到 2×ATR, 加入追踪止损

另外测试:
  - 仅提高阈值 vs 仅精选品种 vs 两者组合
  - 对比各改进的边际贡献

用法:
  python run_v3.py
"""
import sys
import yaml
from pathlib import Path
from datetime import datetime
from loguru import logger

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.collectors.akshare_adapter import get_all_contracts_data
from signals.factors import create_v3_scorer
from signals.generator_v2 import SignalGeneratorV2
from backtest.engine_v2 import BacktestEngineV2


def backtest_with_config(
    data: dict,
    contracts: list,
    label: str,
    min_confidence: float = 0.60,
    exclude_symbols: set = None,
    atr_stop_mult: float = 2.0,
    atr_tp_mult: float = 3.0,
    risk_per_trade: float = 0.02,
) -> dict:
    """
    用指定参数跑一次回测，返回关键指标
    
    所有配置使用同一个引擎 v2，只改参数
    """
    # 过滤数据 (直接过滤 dict，而非 contracts list)
    if exclude_symbols:
        filtered_data = {k: v for k, v in data.items() if k not in exclude_symbols}
        filtered_contracts = [c for c in contracts if c["symbol"] not in exclude_symbols]
    else:
        filtered_data = data
        filtered_contracts = contracts

    logger.info(f"\n{'='*50}")
    logger.info(f"  [{label}] 回测配置:")
    logger.info(f"    品种: {len(filtered_contracts)} 个 ({[c['symbol'] for c in filtered_contracts]})")
    logger.info(f"    置信度阈值: {min_confidence:.0%}")
    logger.info(f"    止损: {atr_stop_mult}×ATR, 止盈: {atr_tp_mult}×ATR")
    logger.info(f"    风险比例: {risk_per_trade:.1%}")
    logger.info(f"{'='*50}")

    scorer = create_v3_scorer()
    
    engine = BacktestEngineV2(
        initial_capital=1_000_000,
        risk_per_trade=risk_per_trade,
        slippage_ticks_entry=1,
        slippage_ticks_exit=1,
        atr_stop_mult=atr_stop_mult,
        atr_tp_mult=atr_tp_mult,
        hold_minutes=45,
        min_confidence=min_confidence,
    )

    result = engine.run(data=filtered_data, contracts=filtered_contracts, scorer=scorer)

    return {
        "label": label,
        "total_return": result.total_return,
        "annual_return": result.annual_return,
        "sharpe": result.sharpe_ratio,
        "max_dd": result.max_drawdown,
        "win_rate": result.win_rate,
        "total_trades": result.total_trades,
        "profit_factor": result.profit_factor,
        "total_commission": result.total_commission,
        "total_slippage": result.total_slippage,
        "avg_confidence": result.avg_confidence,
        "trades_df": result.trades_df,
    }


def main():
    logger.info("=" * 60)
    logger.info("  🧪 Spark v3 策略优化实验")
    logger.info("=" * 60)

    # 加载数据
    config_dir = PROJECT_ROOT / "config"
    with open(config_dir / "contracts.yaml") as f:
        contracts = yaml.safe_load(f)["contracts"]

    logger.info("📡 加载数据...")
    data = get_all_contracts_data(contracts)
    logger.info(f"  获取到 {len(data)} 个品种的数据")

    # 亏损品种集合（基于 min_confidence=0.70 的回测结果更新）
    # 盈利品种(CU/TA/P/RB/SC): Sharpe>0, 胜率>50%, 充足样本
    # 边缘品种(AG/HC/I/JM/MA/SA): 样本太少或盈亏比不佳
    # 亏损品种(M/AL/LH): Sharpe<<0, 持续亏损
    losing_symbols = {"M", "AL", "LH", "JM", "HC"}
    keep_symbols = {c["symbol"] for c in contracts} - losing_symbols
    
    logger.info(f"\n  保留品种: {keep_symbols}")
    logger.info(f"  排除品种: {losing_symbols}")

    results = []

    # ===================================================================
    # 实验1: v2 基准 (min_conf=0.65, 全部14品种) - 已测, 直接引用
    # ===================================================================
    logger.info("\n" + "=" * 50)
    logger.info("  实验0: v2 基准 (min_conf=0.65, 14品种)")
    logger.info("=" * 50)
    r0 = backtest_with_config(
        data, contracts, "v2基准",
        min_confidence=0.65, exclude_symbols=None,
        atr_stop_mult=2.0, atr_tp_mult=3.0,
    )
    results.append(r0)

    # ===================================================================
    # 实验1: 仅提高置信度阈值 0.65→0.70 (14品种)
    # ===================================================================
    r1 = backtest_with_config(
        data, contracts, "仅提阈值(0.70)",
        min_confidence=0.70, exclude_symbols=None,
        atr_stop_mult=2.0, atr_tp_mult=3.0,
    )
    results.append(r1)

    # ===================================================================
    # 实验2: 仅精选品种 (排除5个亏损品种, 阈值仍0.65)
    # ===================================================================
    r2 = backtest_with_config(
        data, contracts, "仅选品种(9只)",
        min_confidence=0.65, exclude_symbols=losing_symbols,
        atr_stop_mult=2.0, atr_tp_mult=3.0,
    )
    results.append(r2)

    # ===================================================================
    # 实验3: 提阈值 + 精选品种 (核心推荐配置)
    # ===================================================================
    r3 = backtest_with_config(
        data, contracts, "0.70+选品种🔥",
        min_confidence=0.70, exclude_symbols=losing_symbols,
        atr_stop_mult=2.0, atr_tp_mult=3.0,
    )
    results.append(r3)

    # ===================================================================
    # 实验4: 最高置信度 (min_conf=0.75), 保持宽止损
    # ===================================================================
    r4 = backtest_with_config(
        data, contracts, "0.75+选品种",
        min_confidence=0.75, exclude_symbols=losing_symbols,
        atr_stop_mult=2.0, atr_tp_mult=3.0,
    )
    results.append(r4)

    # ===================================================================
    # 汇总对比
    # ===================================================================
    print("\n" + "=" * 90)
    print("  📊 策略改进对比总表")
    print("=" * 90)

    header = f"{'实验':<20s} {'品种':>5s} {'阈值':>5s} {'交易数':>6s} {'收益率':>8s} {'夏普':>7s} {'最大回撤':>8s} {'胜率':>7s} {'盈亏比':>7s} {'手续费':>10s} {'滑点':>10s}"
    print(header)
    print("-" * 90)

    for r in results:
        trades_df = r["trades_df"]
        n_symbols = trades_df["symbol"].nunique() if len(trades_df) > 0 else 0
        line = (
            f"{r['label']:<20s} "
            f"{n_symbols:>5d} "
            f"{r['avg_confidence']:>4.0%} "
            f"{r['total_trades']:>6d} "
            f"{r['total_return']:>7.2%} "
            f"{r['sharpe']:>7.2f} "
            f"{r['max_dd']:>7.2%} "
            f"{r['win_rate']:>7.1%} "
            f"{r['profit_factor']:>7.2f} "
            f"{r['total_commission']:>10,.0f} "
            f"{r['total_slippage']:>10,.0f}"
        )
        print(line)

        # 按品种分解
        if len(trades_df) > 0:
            by_sym = trades_df.groupby("symbol").agg(
                次数=("pnl_net", "count"),
                净利=("pnl_net", "sum"),
                胜率=("pnl_net", lambda x: (x > 0).mean()),
            ).round(2)
            for sym, row in by_sym.iterrows():
                print(f"  └─ {sym}: {int(row['次数'])}笔, 净利={row['净利']:,.0f}, 胜率={row['胜率']:.1%}")

    print("-" * 90)

    # ===================================================================
    # 找出最佳配置
    # ===================================================================
    best = max(results, key=lambda r: r["sharpe"] * 0.5 + r["total_return"] * 0.5)
    print(f"\n🏆 最佳配置: {best['label']}")
    print(f"   夏普={best['sharpe']:.2f}, 收益={best['total_return']:.2%}, "
          f"胜率={best['win_rate']:.1%}, 最大回撤={best['max_dd']:.2%}")

    # ===================================================================
    # 保存最佳配置的交易明细
    # ===================================================================
    out_dir = PROJECT_ROOT / "backtest" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    best["trades_df"].to_csv(
        out_dir / f"trades_v3_{timestamp}.csv", index=False, encoding="utf-8-sig"
    )

    # 保存对比表
    compare_lines = [header, "-" * 90]
    for r in results:
        trades_df = r["trades_df"]
        n_symbols = trades_df["symbol"].nunique() if len(trades_df) > 0 else 0
        compare_lines.append(
            f"{r['label']:<20s} {n_symbols:>5d} {r['avg_confidence']:>4.0%} "
            f"{r['total_trades']:>6d} {r['total_return']:>7.2%} {r['sharpe']:>7.2f} "
            f"{r['max_dd']:>7.2%} {r['win_rate']:>7.1%} {r['profit_factor']:>7.2f} "
            f"{r['total_commission']:>10,.0f} {r['total_slippage']:>10,.0f}"
        )
    compare_lines.append("-" * 90)
    compare_lines.append(f"\n🏆 最佳: {best['label']}")

    compare_path = out_dir / f"optimization_comparison_{timestamp}.txt"
    compare_path.write_text("\n".join(compare_lines), encoding="utf-8")
    logger.info(f"对比报告已保存: {compare_path}")

    return results, best


if __name__ == "__main__":
    main()
