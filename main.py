"""
Spark 自进化期货投研系统 - 主入口 (v2)

相比 v1 的改进：
  1. 集成多因子信号评分系统 (7因子)
  2. 使用专业回测引擎 (手续费/滑点/合约乘数/ATR仓位)
  3. 支持 Walk-forward 交叉验证
  4. 生成完整交易明细 DataFrame
  5. 参数稳定性评估
"""
import sys
import yaml
from pathlib import Path
from datetime import datetime
from loguru import logger

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---- 绘图中文字体 ----
import matplotlib
matplotlib.rcParams['font.sans-serif'] = [
    'Noto Serif CJK SC', 'AR PL UMing CN', 'WenQuanYi Micro Hei', 'DejaVu Sans'
]
matplotlib.rcParams['axes.unicode_minus'] = False

# ---- 新模块 ----
from data.collectors.akshare_adapter import get_all_contracts_data
from signals.factors import create_default_scorer, create_v3_scorer
from signals.generator_v2 import SignalGeneratorV2
from backtest.engine_v2 import BacktestEngineV2, BacktestResultV2
from backtest.cross_validation import (
    walk_forward_validate,
    evaluate_param_stability,
    format_cv_report,
)

# 保留旧引擎用于对比
from backtest.engine import plot_equity_curves, generate_report

# ---- 配置日志 ----
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M')}.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO",
)


def load_config():
    """加载配置"""
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    contracts_path = PROJECT_ROOT / "config" / "contracts.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)
    with open(contracts_path) as f:
        contracts_data = yaml.safe_load(f)

    return config, contracts_data['contracts']


def run_backtest_v2(
    data: dict,
    contracts: list,
    config: dict,
    scorer=None,
) -> BacktestResultV2:
    """
    使用 v2 回测引擎运行回测。

    Parameters
    ----------
    data : dict
        品种日线数据 {symbol: DataFrame}
    contracts : list
        品种配置
    config : dict
        全局配置（yaml）
    scorer : WeightedSignalScorer
        多因子评分器

    Returns
    -------
    BacktestResultV2
    """
    backtest_cfg = config.get('backtest', {})

    engine = BacktestEngineV2(
        initial_capital=backtest_cfg.get('initial_capital', 1_000_000),
        risk_per_trade=config.get('trading', {}).get('stop_loss_pct', 0.02),
        slippage_ticks_entry=1,
        slippage_ticks_exit=1,
        atr_stop_mult=2.0,
        atr_tp_mult=3.0,
        hold_minutes=45,
        bootstrap_samples=1000,
        min_confidence=0.60,
    )

    result = engine.run(data=data, contracts=contracts, scorer=scorer)
    return result


def run_cross_validation(
    data: dict,
    contracts: list,
    config: dict,
    scorer=None,
) -> dict:
    """
    运行 Walk-forward 交叉验证和参数稳28定性评估。

    Returns
    -------
    dict
        含 oos_results 和 stability
    """
    backtest_cfg = config.get('backtest', {})

    def bt_fn(data, contracts, start_date, end_date):
        engine = BacktestEngineV2(
            initial_capital=backtest_cfg.get('initial_capital', 1_000_000),
            risk_per_trade=0.02,
            slippage_ticks_entry=1,
            slippage_ticks_exit=1,
            min_confidence=0.60,
        )
        return engine.run(
            data=data, contracts=contracts,
            scorer=scorer,
            start_date=start_date, end_date=end_date,
        )

    oos_results = walk_forward_validate(
        data=data,
        contracts=contracts,
        backtest_fn=bt_fn,
        n_windows=8,
        train_ratio=0.7,
        min_trades_per_window=20,
    )

    stability = evaluate_param_stability(oos_results) if oos_results else {}
    return {"oos_results": oos_results, "stability": stability}


def run_pipeline():
    """
    执行完整流水线:
      1. 加载配置
      2. 采集数据
      3. v2 回测引擎回测
      4. 生成交易信号
      5. 交叉验证
      6. 生成报告
    """
    logger.info("=" * 60)
    logger.info("  🚀 Spark 自进化期货投研系统 v2.0")
    logger.info(f"  🕐 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # ==================================================================
    # 1. 加载配置
    # ==================================================================
    logger.info("\n📋 [1/6] 加载配置...")
    config, contracts = load_config()
    logger.info(f"  系统: {config['system']['name']} v{config['system']['version']}")
    logger.info(f"  加载 {len(contracts)} 个品种")

    # ==================================================================
    # 2. 采集数据
    # ==================================================================
    logger.info("\n📡 [2/6] 采集数据...")
    data = get_all_contracts_data(contracts)
    logger.info(f"  获取到 {len(data)} 个品种的数据")
    for symbol, df in data.items():
        if not df.empty:
            logger.info(
                f"    {symbol}: {len(df)} 条, "
                f"{df['date'].min().date()} ~ {df['date'].max().date()}"
            )

    if not data:
        logger.error("未获取到任何数据，流水线终止")
        return

    # ==================================================================
    # 3. 初始化评分器 + 信号生成器
    # ==================================================================
    logger.info("\n🧠 [3/6] 初始化多因子评分系统...")
    scorer = create_v3_scorer()  # 【v3】使用含ADX趋势过滤的评分器
    logger.info(f"  已初始化 {len(scorer.factors)} 个因子 (含ADX趋势强度)")

    signal_gen = SignalGeneratorV2(
        scorer=scorer,
        min_confidence=0.65,      # 【v3】提高门槛到65%
        min_gap_pct=0.005,        # 【v3】最小缺口提高到0.5%（过滤噪声）
        atr_stop_mult=2.0,
        atr_tp_mult=3.0,
        hold_minutes=45,
    )

    # ==================================================================
    # 4. v2 回测引擎
    # ==================================================================
    logger.info("\n📊 [4/6] 运行 v2 专业回测引擎...")
    result_v2 = run_backtest_v2(data, contracts, config, scorer=scorer)

    logger.info("\n" + result_v2.summary())

    # 输出交易明细
    trades_df = result_v2.trades_df
    if not trades_df.empty:
        logger.info(f"\n  交易明细 (共 {len(trades_df)} 笔):")
        # 按品种汇总展示
        by_symbol = trades_df.groupby("symbol").agg(
            交易次数=("pnl_net", "count"),
            净利润=("pnl_net", "sum"),
            毛利润=("pnl_gross", "sum"),
            总手续费=("commission_entry", lambda x: x.sum() + trades_df.loc[x.index, "commission_exit"].sum()),
            总滑点=("slippage_entry", lambda x: x.sum() + trades_df.loc[x.index, "slippage_exit"].sum()),
            胜率=("pnl_net", lambda x: (x > 0).mean()),
            平均置信度=("confidence", "mean"),
        ).round(2)
        logger.info(f"\n{by_symbol.to_string()}")

    # 生成交易明细 CSV
    trades_csv = PROJECT_ROOT / "backtest" / "reports" / f"trades_v2_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    trades_csv.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(trades_csv, index=False, encoding="utf-8-sig")
    logger.info(f"  交易明细已保存: {trades_csv}")

    # ==================================================================
    # 5. 生成每日信号
    # ==================================================================
    logger.info("\n🔔 [5/6] 生成明日交易信号...")
    signals = signal_gen.generate(data, contracts)

    if signals:
        signal_report = signal_gen.format_report(signals)
        signal_path = PROJECT_ROOT / "backtest" / "reports" / f"signals_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        signal_path.parent.mkdir(parents=True, exist_ok=True)
        signal_path.write_text(signal_report, encoding="utf-8")
        logger.info(f"  信号报告已保存: {signal_path}")

        logger.info("\n🏆 信号一览:")
        for i, s in enumerate(signals):
            logger.info(
                f"  {i+1}. {s.recommendation} {s.name}({s.symbol}) "
                f"{'做多' if s.direction == 'long' else '做空'} "
                f"conf={s.confidence:.1%} gap={s.gap_pct:+.2%}"
            )
    else:
        logger.info("  暂无符合条件的交易信号")

    # ==================================================================
    # 6. 交叉验证
    # ==================================================================
    logger.info("\n🔄 [6/6] Walk-forward 交叉验证...")
    cv_results = run_cross_validation(data, contracts, config, scorer=scorer)

    if cv_results["oos_results"]:
        cv_report = format_cv_report(cv_results["oos_results"], cv_results["stability"])
        cv_path = PROJECT_ROOT / "backtest" / "reports" / f"cv_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        cv_path.parent.mkdir(parents=True, exist_ok=True)
        cv_path.write_text(cv_report, encoding="utf-8")
        logger.info(f"  交叉验证报告已保存: {cv_path}")
    else:
        logger.warning("  交叉验证未产生有效结果（可能数据不足）")

    # ==================================================================
    # 最终汇总
    # ==================================================================
    logger.info("\n" + "=" * 60)
    logger.info("  ✅ 流水线完成!")
    logger.info("=" * 60)
    logger.info(f"  回测: {result_v2.total_trades} 笔交易, Sharpe={result_v2.sharpe_ratio:.2f}")
    logger.info(f"  当日信号: {len(signals)} 个")
    logger.info(f"  交叉验证窗口: {len(cv_results.get('oos_results', []))} 个")

    stability_grade = cv_results.get("stability", {}).get("stability_grade", "N/A")
    logger.info(f"  策略稳定性: {stability_grade}")

    return result_v2, signals, cv_results


if __name__ == "__main__":
    run_pipeline()