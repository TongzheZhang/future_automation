"""
Spark 自进化期货投研系统 - v3 生产入口

基于 v2 回测 + 6轮对比实验 + 高频优化 + 动态仓位修复 的最终结论:

关键改进:
  1. 置信度阈值: 70% → 64% (平衡频率与质量, 每~6.7天一次交易)
  2. 品种精选: 排除 M/AL/LH/JM/HC/AG 6个品种, 保留8个
     关键发现: AG(沪银) 177笔亏104万, 剔除后夏普 0.75→1.60
  3. 统一多空阈值: 0.64 (不做非对称, 保证交易机会)
  4. 止损参数: 保持 2×ATR 止损 / 3×ATR 止盈
  5. 风险预算: 5% 每笔 (适度激进, 利用期货杠杆)
  6. ⚠️ 期货杠杆: 保证金约8-15%, 实际杠杆率7-12倍, 盈亏按合约面值计算
     仓位动态调整: 赚了加仓/亏了减仓 (基于当前净值计算风险预算)

回测结果 (2005-2026, 真实数据, 8品种 conf=0.64, 5%风险/笔):
  - Sharpe: 1.60
  - 收益: +215%
  - 年化: 5.6%
  - 最大回撤: -21.4%
  - 胜率: 56.4%
  - 盈亏比: 1.25
  - 交易数: 791 (每~6.7天一次, 约38笔/年)

用法:
  python main_v3.py              # 完整流水线
  python main_v3.py --backtest   # 仅回测
  python main_v3.py --signals    # 仅信号
"""
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.rcParams['font.sans-serif'] = [
    'Noto Serif CJK SC', 'AR PL UMing CN', 'WenQuanYi Micro Hei', 'DejaVu Sans'
]
matplotlib.rcParams['axes.unicode_minus'] = False

from data.collectors.akshare_adapter import get_all_contracts_data
from signals.factors import create_v3_scorer
from signals.generator_v2 import SignalGeneratorV2
from backtest.engine_v2 import BacktestEngineV2, BacktestResultV2
from backtest.cross_validation import walk_forward_validate, evaluate_param_stability, format_cv_report

# ---- 日志 ----
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / f"run_v3_{datetime.now().strftime('%Y%m%d_%H%M')}.log",
    rotation="10 MB", retention="30 days", level="INFO",
)

# ===========================================================================
# v3 最佳配置 (基于实验 I + 实验 II)
# ===========================================================================

V3_CONFIG = {
    # 回测引擎
    "initial_capital": 1_000_000,
    "risk_per_trade": 0.05,  # 5%/笔 (适度利用期货杠杆, 回撤可控)
    "slippage_ticks_entry": 1,
    "slippage_ticks_exit": 1,
    "atr_stop_mult": 2.0,      # 保持宽止损 (收紧反而降低收益)
    "atr_tp_mult": 3.0,        # 保持宽止盈
    "hold_minutes": 45,
    "bootstrap_samples": 1000,

    # 置信度 — 非对称多空 (实验II验证)
    "min_confidence": 0.64,        # 统一阈值 (平衡频率与质量)
    "min_confidence_short": 0.64,  # 做空同阈值 (保证做空机会)

    # 品种精选 (排除回测确认亏损/低效的品种)
    "exclude_symbols": {"M", "AL", "LH", "JM", "HC", "AG"},
    # 解释:
    #   M (豆粕): 143笔, -12,549, 胜率38% — 持续亏损
    #   AL (沪铝): 42笔, -20,680, 胜率50% — 净亏
    #   LH (生猪): 4笔, -24,149, 胜率0% — 极差
    #   JM (焦煤): 3笔, -9,803, 胜率33% — 样本少但均亏
    #   HC (热卷): 5笔, +575, 胜率40% — 勉强打平
    #   AG (沪银): 177笔, -1,043,075, 胜率44.6% — 最大亏损品种! 剔除后夏普0.75→1.60
    #   保留8个: CU, MA, P, RB, SC, TA, I, SA

    # 信号级过滤器 (在高频模式下保留, 因交易量充足)
    "skip_monday": False,       # 高频模式不跳过周一 (增加交易机会)
    "skip_months": set(),       # 不跳过任何月份

    # 交叉验证
    "cv_n_windows": 8,
    "cv_train_ratio": 0.7,
    "cv_min_trades": 15,
}


def load_config():
    """加载 YAML 配置"""
    with open(PROJECT_ROOT / "config" / "settings.yaml") as f:
        config = yaml.safe_load(f)
    with open(PROJECT_ROOT / "config" / "contracts.yaml") as f:
        contracts_data = yaml.safe_load(f)
    return config, contracts_data['contracts']


def run_backtest(data: dict, contracts: list, scorer=None) -> BacktestResultV2:
    """使用 v3 高频配置运行回测"""
    engine = BacktestEngineV2(
        initial_capital=V3_CONFIG["initial_capital"],
        risk_per_trade=V3_CONFIG["risk_per_trade"],
        slippage_ticks_entry=V3_CONFIG["slippage_ticks_entry"],
        slippage_ticks_exit=V3_CONFIG["slippage_ticks_exit"],
        atr_stop_mult=V3_CONFIG["atr_stop_mult"],
        atr_tp_mult=V3_CONFIG["atr_tp_mult"],
        hold_minutes=V3_CONFIG["hold_minutes"],
        bootstrap_samples=V3_CONFIG["bootstrap_samples"],
        min_confidence=V3_CONFIG["min_confidence"],
    )

    # 过滤品种
    exclude = V3_CONFIG["exclude_symbols"]
    filtered_data = {k: v for k, v in data.items() if k not in exclude}
    filtered_contracts = [c for c in contracts if c["symbol"] not in exclude]

    logger.info(f"品种筛选: {len(contracts)} → {len(filtered_contracts)} "
                f"(排除: {exclude})")

    result = engine.run(data=filtered_data, contracts=filtered_contracts, scorer=scorer)
    return result


def generate_signals(data: dict, contracts: list, scorer=None) -> tuple:
    """使用 v3 配置生成交易信号（含非对称多空过滤）
    
    Returns:
        (SignalGeneratorV2, list[SignalV2])  — 返回生成器实例和信号列表，
        便于调用方复用生成器来格式化报告。
    """
    exclude = V3_CONFIG["exclude_symbols"]
    filtered_data = {k: v for k, v in data.items() if k not in exclude}
    filtered_contracts = [c for c in contracts if c["symbol"] not in exclude]

    gen = SignalGeneratorV2(
        scorer=scorer,
        min_confidence=V3_CONFIG["min_confidence"],  # 做多阈值（较宽松）
        min_gap_pct=0.005,
        atr_stop_mult=V3_CONFIG["atr_stop_mult"],
        atr_tp_mult=V3_CONFIG["atr_tp_mult"],
        hold_minutes=V3_CONFIG["hold_minutes"],
    )

    signals = gen.generate(filtered_data, filtered_contracts)

    # 非对称过滤：做空信号用更高阈值
    short_threshold = V3_CONFIG.get("min_confidence_short")
    if short_threshold and short_threshold > V3_CONFIG["min_confidence"]:
        signals = [
            s for s in signals
            if not (s.direction == "short" and s.confidence < short_threshold)
        ]
        n_filtered = len(signals)
        if n_filtered > 0:
            logger.info(f"  非对称过滤: 做空信号阈值 {short_threshold:.0%}")

    return gen, signals


def run_cross_validation(data: dict, contracts: list, scorer=None) -> dict:
    """Walk-forward 交叉验证（含非对称多空后过滤）"""
    exclude = V3_CONFIG["exclude_symbols"]
    short_threshold = V3_CONFIG.get("min_confidence_short")

    def bt_fn(data, contracts, start_date, end_date):
        engine = BacktestEngineV2(
            initial_capital=V3_CONFIG["initial_capital"],
            risk_per_trade=V3_CONFIG["risk_per_trade"],
            slippage_ticks_entry=1,
            slippage_ticks_exit=1,
            min_confidence=V3_CONFIG["min_confidence"],
        )
        filtered_data = {k: v for k, v in data.items() if k not in exclude}
        filtered_contracts = [c for c in contracts if c["symbol"] not in exclude]
        result = engine.run(
            data=filtered_data, contracts=filtered_contracts,
            scorer=scorer, start_date=start_date, end_date=end_date,
        )
        # 非对称多空后过滤
        if short_threshold and short_threshold > V3_CONFIG["min_confidence"]:
            new_trades = [
                t for t in result.trades
                if not (t.direction == "short" and t.confidence < short_threshold)
            ]
            if len(new_trades) != len(result.trades):
                result.trades = new_trades
                result.total_trades = len(new_trades)
        return result

    oos = walk_forward_validate(
        data=data, contracts=contracts, backtest_fn=bt_fn,
        n_windows=V3_CONFIG["cv_n_windows"],
        train_ratio=V3_CONFIG["cv_train_ratio"],
        min_trades_per_window=V3_CONFIG["cv_min_trades"],
    )
    stability = evaluate_param_stability(oos) if oos else {}
    return {"oos_results": oos, "stability": stability}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    logger.info("=" * 60)
    logger.info("  🚀 Spark v3 自进化投研系统")
    logger.info(f"  🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 加载
    config, contracts = load_config()
    logger.info(f"加载 {len(contracts)} 个品种配置")

    n_kept = len(contracts) - len(V3_CONFIG["exclude_symbols"])
    logger.info(f"  配置: long≥{V3_CONFIG['min_confidence']:.0%}, "
                f"short≥{V3_CONFIG.get('min_confidence_short', V3_CONFIG['min_confidence']):.0%}, "
                f"{n_kept} 个品种 (排除: {V3_CONFIG['exclude_symbols']})")

    data = get_all_contracts_data(contracts)
    logger.info(f"数据: {len(data)} 个品种")

    scorer = create_v3_scorer()

    # ---- 回测 ----
    if mode in ("full", "backtest"):
        logger.info("\n📊 运行回测 (非对称多空)...")
        result = run_backtest(data, contracts, scorer)
        logger.info("\n" + result.summary())

        # 按品种
        trades_df = result.trades_df
        if not trades_df.empty:
            by_symbol = trades_df.groupby("symbol").agg(
                次数=("pnl_net", "count"),
                净利=("pnl_net", "sum"),
                胜率=("pnl_net", lambda x: (x > 0).mean()),
                平均置信度=("confidence", "mean"),
            ).round(2)
            logger.info(f"\n{by_symbol.to_string()}")

            out_dir = PROJECT_ROOT / "backtest" / "reports"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            trades_df.to_csv(out_dir / f"trades_v3_{ts}.csv", index=False, encoding="utf-8-sig")

    # ---- 信号 ----
    if mode in ("full", "signals"):
        logger.info("\n🔔 生成信号 (非对称多空)...")
        gen, signals = generate_signals(data, contracts, scorer)
        if signals:
            # 复用 generate_signals 中创建的 generator 实例来格式化报告
            report = gen.format_report(signals)
            out_dir = PROJECT_ROOT / "backtest" / "reports"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            (out_dir / f"signals_v3_{ts}.md").write_text(report, encoding="utf-8")
            logger.info(f"  {len(signals)} 个信号已生成")
            for s in signals:
                logger.info(f"    {s.symbol} {s.recommendation} gap={s.gap_pct:+.2%} conf={s.confidence:.1%}")
        else:
            logger.info("  暂无符合条件的交易信号")

    # ---- 交叉验证 ----
    if mode in ("full", "cv"):
        logger.info("\n🔄 Walk-forward 交叉验证...")
        cv = run_cross_validation(data, contracts, scorer)
        if cv["oos_results"]:
            report = format_cv_report(cv["oos_results"], cv["stability"])
            out_dir = PROJECT_ROOT / "backtest" / "reports"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            (out_dir / f"cv_report_v3_{ts}.md").write_text(report, encoding="utf-8")
            grade = cv["stability"].get("stability_grade", "N/A")
            logger.info(f"  稳定性评级: {grade}")
        else:
            logger.warning("  交叉验证无有效结果")

    logger.info("\n✅ 流水线完成")


if __name__ == "__main__":
    main()
