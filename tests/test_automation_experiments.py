import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from automation.experiments import (
    apply_direction_bias,
    build_engine_from_params,
    evaluate_candidate,
    filter_data_and_contracts,
)
from automation.models import CandidatePlugin
from backtest.engine_v2 import BacktestEngineV2, BacktestResultV2, TradeRecord


def make_candidate():
    return CandidatePlugin(
        id="c1",
        idea_id="i1",
        created_at="2026-05-14T00:00:00Z",
        title="SA candidate",
        symbols=["SA"],
        strategy_family="gap_filter",
        source_refs=[],
        params={
            "include_symbols": ["SA"],
            "exclude_symbols": [],
            "min_gap_pct": 0.006,
            "min_confidence": 0.7,
            "hold_minutes": 30,
            "atr_stop_mult": 1.8,
            "atr_tp_mult": 2.6,
            "factor_weight_overrides": {},
            "direction_bias": "long",
        },
    )


def test_candidate_params_map_to_engine():
    engine = build_engine_from_params(make_candidate().params)
    assert engine.min_gap_pct == 0.006
    assert engine.min_confidence == 0.7
    assert engine.hold_minutes == 30
    assert engine.atr_stop_mult == 1.8


def test_filter_data_and_contracts_include_exclude():
    data = {
        "SA": pd.DataFrame({"date": []}),
        "SC": pd.DataFrame({"date": []}),
    }
    contracts = [{"symbol": "SA"}, {"symbol": "SC"}]
    filtered_data, filtered_contracts = filter_data_and_contracts(data, contracts, make_candidate().params)
    assert list(filtered_data) == ["SA"]
    assert filtered_contracts == [{"symbol": "SA"}]


def result(trades, sharpe, drawdown, win_rate, profit_factor):
    return BacktestResultV2(
        total_trades=trades,
        sharpe_ratio=sharpe,
        max_drawdown=drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
    )


def test_acceptance_ready_rejected_and_needs_more_data():
    candidate = make_candidate()
    thresholds = {
        "min_trades": 20,
        "min_sharpe": 0.8,
        "max_drawdown": 0.25,
        "min_win_rate": 0.5,
        "min_profit_factor": 1.05,
        "min_sharpe_improvement": 0.0,
    }
    baseline = result(30, 0.9, -0.1, 0.52, 1.1)
    ready = evaluate_candidate(candidate, result(30, 1.2, -0.1, 0.55, 1.2), baseline, thresholds)
    assert ready.status == "ready_for_review"
    rejected = evaluate_candidate(candidate, result(30, 0.2, -0.1, 0.55, 1.2), baseline, thresholds)
    assert rejected.status == "rejected"
    low_sample = evaluate_candidate(candidate, result(5, 2.0, -0.1, 0.8, 2.0), baseline, thresholds)
    assert low_sample.status == "needs_more_data"


def test_direction_bias_filters_result_trades():
    engine = BacktestEngineV2(initial_capital=1_000_000)
    trades = [
        TradeRecord(
            symbol="SA", name="纯碱", trade_date=pd.Timestamp("2024-01-02"),
            direction="long", entry_time=pd.Timestamp("2024-01-02"),
            entry_price=100, exit_time=pd.Timestamp("2024-01-02"),
            exit_price=110, exit_reason="exit", pnl_pct=0.1,
            pnl_gross=1000, commission_entry=1, commission_exit=1,
            slippage_entry=1, slippage_exit=1, pnl_net=996,
            position_size=1, notional=1000, gap_pct=0.01,
            atr_at_entry=2, confidence=0.7,
        ),
        TradeRecord(
            symbol="SA", name="纯碱", trade_date=pd.Timestamp("2024-01-03"),
            direction="short", entry_time=pd.Timestamp("2024-01-03"),
            entry_price=100, exit_time=pd.Timestamp("2024-01-03"),
            exit_price=90, exit_reason="exit", pnl_pct=0.1,
            pnl_gross=1000, commission_entry=1, commission_exit=1,
            slippage_entry=1, slippage_exit=1, pnl_net=996,
            position_size=1, notional=1000, gap_pct=-0.01,
            atr_at_entry=2, confidence=0.7,
        ),
    ]
    result_obj = BacktestResultV2(trades=trades, total_trades=2)
    filtered = apply_direction_bias(result_obj, engine, "long")
    assert filtered.total_trades == 1
    assert filtered.trades[0].direction == "long"
