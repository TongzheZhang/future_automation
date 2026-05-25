"""
测试信号模型与评估
"""

import pytest
from signals.models import TradingSignal, Direction, RiskLevel, PolicyImpact, FundamentalState
from signals.evaluator import SignalEvaluator


def test_trading_signal_creation():
    """测试交易信号创建"""
    signal = TradingSignal(
        id="SIG-TEST001",
        commodity_code="RB",
        commodity_name="螺纹钢",
        exchange="SHFE",
        direction=Direction.LONG,
        confidence=0.85,
        conviction_level="高",
        catalyst="政策限产",
        core_logic="环保限产导致供应收缩",
        stop_loss_logic="跌破前低止损",
        target_logic="基差修复",
        holding_period_days=14,
        risk_level=RiskLevel.MEDIUM,
        position_sizing="适中",
    )
    
    assert signal.direction == Direction.LONG
    assert signal.confidence == 0.85


def test_signal_evaluator():
    """测试信号评估"""
    evaluator = SignalEvaluator()
    
    signal = TradingSignal(
        id="SIG-TEST001",
        commodity_code="RB",
        commodity_name="螺纹钢",
        exchange="SHFE",
        direction=Direction.LONG,
        confidence=0.85,
        conviction_level="高",
        catalyst="政策限产",
        core_logic="环保限产导致供应收缩",
        stop_loss_logic="跌破前低止损",
        target_logic="基差修复",
        holding_period_days=14,
        risk_level=RiskLevel.MEDIUM,
        position_sizing="适中",
        policy_driver=PolicyImpact(
            policy_title="测试政策",
            policy_source="发改委",
            policy_level="部委",
            policy_type="环保",
            direction=Direction.LONG,
            mechanism="限产",
            strength="强",
            time_horizon="中期",
            confidence=0.8,
        ),
        fundamental_driver=FundamentalState(
            inventory_status="偏低",
            score=0.75,
        ),
    )
    
    result = evaluator.evaluate_signal(signal)
    assert result["passed"] is True
    assert result["evaluated_score"] > 0.5


def test_sector_exposure():
    """测试板块风控"""
    evaluator = SignalEvaluator()
    
    signals = [
        TradingSignal(
            id=f"SIG-{i}",
            commodity_code=code,
            commodity_name=name,
            exchange="SHFE",
            direction=Direction.LONG,
            confidence=0.8,
            conviction_level="高",
            catalyst="测试",
            core_logic="测试",
            stop_loss_logic="测试",
            target_logic="测试",
            holding_period_days=10,
            risk_level=RiskLevel.LOW,
            position_sizing="轻仓",
        )
        for i, (code, name) in enumerate([
            ("RB", "螺纹钢"), ("I", "铁矿石"), ("J", "焦炭"), ("CU", "铜")
        ])
    ]
    
    filtered = evaluator.check_sector_exposure(signals, max_per_sector=2)
    # 黑色系最多保留 2 个
    black_count = sum(1 for s in filtered if s.commodity_code in ["RB", "I", "J"])
    assert black_count <= 2
