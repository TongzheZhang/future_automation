"""
BacktestEngineV2 核心逻辑单元测试
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from backtest.engine_v2 import BacktestEngineV2, TradeRecord, get_spec


class TestPositionSize:
    """仓位计算测试"""

    def test_risk_based_position(self):
        """正常风险预算下的仓位计算"""
        engine = BacktestEngineV2(initial_capital=1_000_000, risk_per_trade=0.02)
        # RB: multiplier=10, 价格4000, ATR=80, stop_mult=2.0
        # 止损距离 = 160, 每手亏损 = 160 * 10 = 1600
        # 风险金额 = 1,000,000 * 0.02 = 20,000
        # 手数 = 20,000 / 1600 = 12
        lots = engine._calc_position_size("RB", 4000.0, 80.0, 1_000_000)
        assert lots == 12

    def test_zero_risk_returns_zero(self):
        """ATR为0或风险过大时应返回0（不交易）"""
        engine = BacktestEngineV2(initial_capital=1_000_000, risk_per_trade=0.02)
        lots = engine._calc_position_size("RB", 4000.0, 0.0, 1_000_000)
        assert lots == 0

    def test_margin_cap(self):
        """保证金约束应限制最大手数"""
        engine = BacktestEngineV2(initial_capital=1_000_000, risk_per_trade=0.50)
        # SC: multiplier=1000, 价格600, margin=0.15
        # 保证金约束: max_lots = 1,000,000 / (600 * 1000 * 0.15) ≈ 11
        lots = engine._calc_position_size("SC", 600.0, 1.0, 1_000_000)
        assert lots <= 11


class TestCommissionAndSlippage:
    """费用计算测试"""

    def test_pct_fee(self):
        """按比例收费品种"""
        engine = BacktestEngineV2()
        comm, notional = engine._calc_commission("RB", 4000.0, 10)
        assert notional == 4000.0 * 10 * 10  # price * multiplier * lots
        assert comm == notional * 0.0001

    def test_fixed_fee(self):
        """固定收费品种"""
        engine = BacktestEngineV2()
        comm, notional = engine._calc_commission("AU", 500.0, 5)
        assert comm == 10.0 * 5  # 10元/手 * 5手

    def test_slippage_cost(self):
        """滑点成本计算"""
        engine = BacktestEngineV2()
        cost = engine._calc_slippage_cost("RB", 2, 10)
        # RB: min_tick=1.0, multiplier=10
        assert cost == 2 * 1.0 * 10 * 10  # ticks * min_tick * multiplier * lots


class TestDailyEquity:
    """日频权益曲线测试"""

    def test_empty_trades(self):
        """无交易时返回初始资金"""
        engine = BacktestEngineV2(initial_capital=1_000_000)
        eq, rets = engine._compute_daily_equity([])
        assert len(eq) == 1
        assert eq.iloc[0] == 1_000_000
        assert len(rets) == 0

    def test_single_trade(self):
        """单笔交易的权益曲线"""
        engine = BacktestEngineV2(initial_capital=1_000_000)
        trade = TradeRecord(
            symbol="RB", name="螺纹钢", trade_date=pd.Timestamp("2024-01-15"),
            direction="long", entry_time=pd.Timestamp("2024-01-15"),
            entry_price=4000, exit_time=pd.Timestamp("2024-01-15"),
            exit_price=4100, exit_reason="exit", pnl_pct=0.025,
            pnl_gross=10000, commission_entry=40, commission_exit=41,
            slippage_entry=100, slippage_exit=100, pnl_net=9719,
            position_size=10, notional=400000, gap_pct=0.01,
            atr_at_entry=80, confidence=0.7,
        )
        eq, rets = engine._compute_daily_equity([trade])
        assert eq.iloc[-1] == 1_009_719
        assert len(rets) == 1

    def test_cross_symbol_chronological(self):
        """跨品种交易应按日期排序"""
        engine = BacktestEngineV2(initial_capital=1_000_000)
        trades = [
            TradeRecord(
                symbol="CU", name="沪铜", trade_date=pd.Timestamp("2024-01-16"),
                direction="short", entry_time=pd.Timestamp("2024-01-16"),
                entry_price=70000, exit_time=pd.Timestamp("2024-01-16"),
                exit_price=69500, exit_reason="exit", pnl_pct=0.0071,
                pnl_gross=5000, commission_entry=175, commission_exit=174,
                slippage_entry=250, slippage_exit=250, pnl_net=4151,
                position_size=1, notional=350000, gap_pct=0.005,
                atr_at_entry=500, confidence=0.65,
            ),
            TradeRecord(
                symbol="RB", name="螺纹钢", trade_date=pd.Timestamp("2024-01-15"),
                direction="long", entry_time=pd.Timestamp("2024-01-15"),
                entry_price=4000, exit_time=pd.Timestamp("2024-01-15"),
                exit_price=4100, exit_reason="exit", pnl_pct=0.025,
                pnl_gross=10000, commission_entry=40, commission_exit=41,
                slippage_entry=100, slippage_exit=100, pnl_net=9719,
                position_size=10, notional=400000, gap_pct=0.01,
                atr_at_entry=80, confidence=0.7,
            ),
        ]
        eq, rets = engine._compute_daily_equity(trades)
        # 尽管 trades 列表中 CU 在 RB 前面，但权益曲线应按日期排序
        # 1月15日: +9719, 1月16日: +4151
        assert eq.iloc[0] == 1_000_000
        assert eq.iloc[1] == 1_009_719
        assert eq.iloc[2] == 1_013_870


class TestRunSingleSymbol:
    """单品种回测测试"""

    def test_no_trade_when_gap_too_small(self):
        """缺口小于阈值时不应产生交易"""
        engine = BacktestEngineV2(min_confidence=0.0)
        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        df = pd.DataFrame({
            "date": dates,
            "open": [100.0] * 20,
            "high": [101.0] * 20,
            "low": [99.0] * 20,
            "close": [100.0] * 20,
            "volume": [1000] * 20,
            "open_interest": [5000] * 20,
        })
        trades, equity = engine._run_single_symbol("RB", df, "螺纹钢", None, None, None, 1_000_000)
        assert len(trades) == 0
        assert equity == 1_000_000

    def test_lots_zero_skips_trade(self):
        """仓位为0时应跳过交易"""
        engine = BacktestEngineV2(initial_capital=1000, risk_per_trade=0.02)
        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        df = pd.DataFrame({
            "date": dates,
            "open": [100000.0] * 5 + [110000.0] * 15,  # 大缺口
            "high": [101000.0] * 5 + [111000.0] * 15,
            "low": [99000.0] * 5 + [109000.0] * 15,
            "close": [100000.0] * 5 + [110000.0] * 15,
            "volume": [1000] * 20,
            "open_interest": [5000] * 20,
        })
        # 资金太小，ATR极大，仓位为0
        trades, equity = engine._run_single_symbol("RB", df, "螺纹钢", None, None, None, 1000)
        assert len(trades) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
