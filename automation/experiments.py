"""Candidate experiment execution and acceptance evaluation."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

from automation.candidates import load_candidates, update_candidate_status
from automation.models import CandidatePlugin, ExperimentResult
from backtest.engine_v2 import BacktestEngineV2, BacktestResultV2

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from main_v3 import V3_CONFIG
from signals.factors import WeightedSignalScorer, create_v3_scorer


def create_experiment_scorer() -> WeightedSignalScorer:
    scorer = create_v3_scorer()
    for factor in scorer.factors:
        if factor.name == "historical_win_rate":
            factor.persist = False
    return scorer


def apply_factor_weight_overrides(scorer: WeightedSignalScorer, overrides: dict[str, float]) -> WeightedSignalScorer:
    if not overrides:
        return scorer
    for factor in scorer.factors:
        if factor.name in overrides:
            factor.weight = float(overrides[factor.name])
    return WeightedSignalScorer(scorer.factors)


def filter_data_and_contracts(
    data: dict[str, pd.DataFrame],
    contracts: list[dict],
    params: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    include = set(params.get("include_symbols") or data.keys())
    exclude = set(params.get("exclude_symbols") or set())
    symbols = include - exclude
    filtered_data = {symbol: df for symbol, df in data.items() if symbol in symbols}
    filtered_contracts = [c for c in contracts if c.get("symbol") in filtered_data]
    return filtered_data, filtered_contracts


def build_engine_from_params(params: dict[str, Any]) -> BacktestEngineV2:
    return BacktestEngineV2(
        initial_capital=V3_CONFIG["initial_capital"],
        risk_per_trade=V3_CONFIG["risk_per_trade"],
        slippage_ticks_entry=V3_CONFIG["slippage_ticks_entry"],
        slippage_ticks_exit=V3_CONFIG["slippage_ticks_exit"],
        atr_stop_mult=float(params.get("atr_stop_mult", V3_CONFIG["atr_stop_mult"])),
        atr_tp_mult=float(params.get("atr_tp_mult", V3_CONFIG["atr_tp_mult"])),
        hold_minutes=int(params.get("hold_minutes", V3_CONFIG["hold_minutes"])),
        bootstrap_samples=200,
        min_confidence=float(params.get("min_confidence", V3_CONFIG["min_confidence"])),
        min_gap_pct=float(params.get("min_gap_pct", 0.003)),
    )


def run_candidate_backtest(
    candidate: CandidatePlugin,
    data: dict[str, pd.DataFrame],
    contracts: list[dict],
) -> BacktestResultV2:
    scorer = apply_factor_weight_overrides(
        create_experiment_scorer(),
        candidate.params.get("factor_weight_overrides", {}),
    )
    filtered_data, filtered_contracts = filter_data_and_contracts(data, contracts, candidate.params)
    engine = build_engine_from_params(candidate.params)
    start = candidate.test_window.get("start_date")
    end = candidate.test_window.get("end_date")
    result = engine.run(
        data=filtered_data,
        contracts=filtered_contracts,
        scorer=scorer,
        start_date=start,
        end_date=end,
    )
    return apply_direction_bias(result, engine, candidate.params.get("direction_bias", "neutral"))


def apply_direction_bias(
    result: BacktestResultV2,
    engine: BacktestEngineV2,
    direction_bias: str,
) -> BacktestResultV2:
    if direction_bias not in {"long", "short"}:
        return result
    trades = [trade for trade in result.trades if trade.direction == direction_bias]
    return rebuild_result_from_trades(trades, engine)


def rebuild_result_from_trades(
    trades,
    engine: BacktestEngineV2,
) -> BacktestResultV2:
    if not trades:
        return BacktestResultV2()

    eq_series, daily_returns = engine._compute_daily_equity(trades)
    n_trades = len(trades)
    total_return = float((eq_series.iloc[-1] - engine.initial_capital) / engine.initial_capital)
    trade_dates = sorted([t.trade_date for t in trades])
    years = (trade_dates[-1] - trade_dates[0]).days / 365.25 if len(trade_dates) >= 2 else 1 / 252
    years = max(years, 0.1)
    annual_return = total_return / years if years > 0 else 0.0
    sharpe = (
        float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
        if len(daily_returns) > 1 and daily_returns.std() > 0
        else 0.0
    )
    peak = eq_series.expanding().max()
    drawdown = (eq_series - peak) / peak
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0
    wins = [t for t in trades if t.pnl_net > 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
    gross_profit = sum(t.pnl_net for t in wins)
    gross_loss = abs(sum(t.pnl_net for t in trades if t.pnl_net <= 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    total_commission = sum(t.commission_entry + t.commission_exit for t in trades)
    total_slippage = sum(t.slippage_entry + t.slippage_exit for t in trades)
    calmar = annual_return / abs(max_dd) if max_dd < 0 else 0.0
    avg_confidence = float(np.mean([t.confidence for t in trades]))
    sharpe_ci, win_rate_ci = engine._bootstrap_ci(trades)
    return BacktestResultV2(
        trades=trades,
        equity_curve=eq_series,
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        total_trades=n_trades,
        profit_factor=profit_factor,
        calmar_ratio=calmar,
        total_commission=total_commission,
        total_slippage=total_slippage,
        avg_hold_minutes=engine.hold_minutes,
        avg_confidence=avg_confidence,
        sharpe_ci_low=sharpe_ci[0],
        sharpe_ci_high=sharpe_ci[1],
        win_rate_ci_low=win_rate_ci[0],
        win_rate_ci_high=win_rate_ci[1],
    )


def run_baseline_backtest(
    candidate: CandidatePlugin,
    data: dict[str, pd.DataFrame],
    contracts: list[dict],
) -> BacktestResultV2:
    params = {
        "include_symbols": candidate.symbols,
        "exclude_symbols": [],
        "min_confidence": V3_CONFIG["min_confidence"],
        "min_gap_pct": 0.003,
        "hold_minutes": V3_CONFIG["hold_minutes"],
        "atr_stop_mult": V3_CONFIG["atr_stop_mult"],
        "atr_tp_mult": V3_CONFIG["atr_tp_mult"],
    }
    baseline = CandidatePlugin(
        id=f"{candidate.id}_baseline",
        idea_id=candidate.idea_id,
        created_at=candidate.created_at,
        title=f"{candidate.title} baseline",
        symbols=candidate.symbols,
        strategy_family=candidate.strategy_family,
        source_refs=candidate.source_refs,
        params=params,
        status="testing",
        test_window=candidate.test_window,
    )
    return run_candidate_backtest(baseline, data, contracts)


def metrics_from_result(result: BacktestResultV2) -> dict[str, Any]:
    def f(value: Any) -> float:
        return float(round(float(value), 6))

    return {
        "total_trades": int(result.total_trades),
        "total_return": f(result.total_return),
        "annual_return": f(result.annual_return),
        "sharpe": f(result.sharpe_ratio),
        "max_drawdown": f(result.max_drawdown),
        "win_rate": f(result.win_rate),
        "profit_factor": f(result.profit_factor) if np.isfinite(result.profit_factor) else "inf",
        "avg_confidence": f(result.avg_confidence),
    }


def evaluate_candidate(
    candidate: CandidatePlugin,
    result: BacktestResultV2,
    baseline: BacktestResultV2,
    thresholds: dict[str, Any],
) -> ExperimentResult:
    min_trades = int(thresholds.get("min_trades", 20))
    min_sharpe = float(thresholds.get("min_sharpe", 0.8))
    max_drawdown = float(thresholds.get("max_drawdown", 0.25))
    min_win_rate = float(thresholds.get("min_win_rate", 0.50))
    min_profit_factor = float(thresholds.get("min_profit_factor", 1.05))
    min_sharpe_improvement = float(thresholds.get("min_sharpe_improvement", 0.0))

    metrics = metrics_from_result(result)
    baseline_metrics = metrics_from_result(baseline)
    reasons: list[str] = []

    if result.total_trades < min_trades:
        reasons.append(f"trades<{min_trades}")
    if result.sharpe_ratio < min_sharpe:
        reasons.append(f"sharpe<{min_sharpe}")
    if abs(result.max_drawdown) > max_drawdown:
        reasons.append(f"drawdown>{max_drawdown}")
    if result.win_rate < min_win_rate:
        reasons.append(f"win_rate<{min_win_rate}")
    if result.profit_factor < min_profit_factor:
        reasons.append(f"profit_factor<{min_profit_factor}")
    if (result.sharpe_ratio - baseline.sharpe_ratio) < min_sharpe_improvement:
        reasons.append(f"sharpe_improvement<{min_sharpe_improvement}")

    if result.total_trades < min_trades:
        status = "needs_more_data"
    elif reasons:
        status = "rejected"
    else:
        status = "ready_for_review"

    return ExperimentResult(
        candidate_id=candidate.id,
        idea_id=candidate.idea_id,
        status=status,
        reason=", ".join(reasons) if reasons else "passed acceptance thresholds",
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        passed=status == "ready_for_review",
    )


def append_experiment_result(path: Path, result: ExperimentResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def run_candidate_experiments(
    candidates_dir: Path,
    experiments_path: Path,
    data: dict[str, pd.DataFrame],
    contracts: list[dict],
    thresholds: dict[str, Any],
    dry_run: bool = False,
) -> list[ExperimentResult]:
    candidates = [
        c for c in load_candidates(candidates_dir)
        if c.status in {"new", "testing", "needs_more_data"}
    ]
    if dry_run:
        candidates = candidates[:3]

    results: list[ExperimentResult] = []
    for candidate in candidates:
        try:
            candidate.status = "testing"
            result = run_candidate_backtest(candidate, data, contracts)
            baseline = run_baseline_backtest(candidate, data, contracts)
            experiment = evaluate_candidate(candidate, result, baseline, thresholds)
            validation = experiment.to_dict()
            update_candidate_status(candidate, candidates_dir, experiment.status, validation)
            append_experiment_result(experiments_path, experiment)
            results.append(experiment)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"candidate experiment failed for {candidate.id}: {exc}")
            experiment = ExperimentResult(
                candidate_id=candidate.id,
                idea_id=candidate.idea_id,
                status="rejected",
                reason=f"experiment_error: {exc}",
                metrics={},
                baseline_metrics={},
                passed=False,
            )
            update_candidate_status(candidate, candidates_dir, "rejected", experiment.to_dict())
            append_experiment_result(experiments_path, experiment)
            results.append(experiment)
    return results


def make_dry_run_data(contracts: list[dict], periods: int = 180) -> dict[str, pd.DataFrame]:
    rng = np.random.RandomState(7)
    data: dict[str, pd.DataFrame] = {}
    for idx, contract in enumerate(contracts):
        symbol = contract["symbol"]
        dates = pd.date_range("2024-01-01", periods=periods, freq="B")
        close = 1000 + idx * 100 + np.cumsum(rng.randn(periods) * 8)
        open_ = close * (1 + rng.randn(periods) * 0.004)
        if idx % 3 == 0:
            open_[30::25] = close[29::25][:len(open_[30::25])] * 1.012
        high = np.maximum(open_, close) + np.abs(rng.randn(periods) * 6)
        low = np.minimum(open_, close) - np.abs(rng.randn(periods) * 6)
        data[symbol] = pd.DataFrame(
            {
                "date": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": 10000 + np.abs(rng.randn(periods) * 1200),
                "open_interest": 50000 + np.cumsum(rng.randn(periods) * 100),
            }
        )
    return data
