"""
测试日内T+0模块
"""

import pytest
from datetime import datetime

from data.collectors.market_data import MarketDataCollector, SINA_CODE_MAP
from intraday.models import (
    IntradaySignal, IntradayTrade, DailyReview,
    Direction, TradeStatus, MarketSnapshotData,
)
from intraday.record import (
    save_signals, load_signals, save_trades, load_trades,
    save_review, load_review,
)


def test_sina_code_map():
    """测试品种代码映射"""
    assert SINA_CODE_MAP["RB"] == "nf_RB0"
    assert SINA_CODE_MAP["M"] == "nf_M0"
    assert SINA_CODE_MAP["CU"] == "nf_CU0"


def test_market_snapshot_properties():
    """测试行情快照属性"""
    snap = MarketSnapshotData(
        commodity="RB",
        open=3200,
        prev_settle=3100,
        last=3250,
        high=3300,
        low=3150,
        gap_pct=3.23,
        change_pct=4.84,
        amplitude_pct=4.84,
    )
    assert snap.gap_pct == pytest.approx(3.23, rel=0.01)
    assert snap.change_pct == pytest.approx(4.84, rel=0.01)
    assert snap.amplitude_pct == pytest.approx(4.84, rel=0.01)


def test_intraday_signal_should_trade():
    """测试信号交易判断"""
    s1 = IntradaySignal(
        date="2026-05-25",
        commodity="RB",
        direction=Direction.LONG,
        confidence=8,
    )
    assert s1.should_trade() is True
    
    s2 = IntradaySignal(
        date="2026-05-25",
        commodity="M",
        direction=Direction.NO_TRADE,
        confidence=5,
    )
    assert s2.should_trade() is False
    
    s3 = IntradaySignal(
        date="2026-05-25",
        commodity="CU",
        direction=Direction.SHORT,
        confidence=6,
    )
    assert s3.should_trade() is False  # 置信度<7


def test_intraday_trade_calculate_pnl():
    """测试盈亏计算"""
    t1 = IntradayTrade(
        date="2026-05-25",
        commodity="RB",
        direction=Direction.LONG,
        actual_entry=3200,
        actual_exit=3250,
    )
    t1.calculate_pnl()
    assert t1.pnl == 500.0   # RB 合约乘数 10，(3250-3200)*10
    assert t1.status == TradeStatus.WIN
    
    t2 = IntradayTrade(
        date="2026-05-25",
        commodity="M",
        direction=Direction.SHORT,
        actual_entry=3000,
        actual_exit=3050,
    )
    t2.calculate_pnl()
    assert t2.pnl == -500.0  # M 合约乘数 10，(3000-3050)*10
    assert t2.status == TradeStatus.LOSS


def test_record_roundtrip():
    """测试记录读写"""
    date_str = "2026-05-25"
    
    # 信号
    signals = [
        IntradaySignal(date=date_str, commodity="RB", direction=Direction.LONG, confidence=8),
        IntradaySignal(date=date_str, commodity="M", direction=Direction.NO_TRADE, confidence=5),
    ]
    save_signals(date_str, signals)
    loaded_signals = load_signals(date_str)
    assert len(loaded_signals) == 2
    assert loaded_signals[0].commodity == "RB"
    assert loaded_signals[0].confidence == 8
    
    # 交易
    trades = [
        IntradayTrade(
            date=date_str, commodity="RB", direction=Direction.LONG,
            actual_entry=3200, actual_exit=3250,
        ),
    ]
    trades[0].calculate_pnl()
    save_trades(date_str, trades)
    loaded_trades = load_trades(date_str)
    assert len(loaded_trades) == 1
    assert loaded_trades[0].pnl == 500.0
    
    # 复盘
    review = DailyReview(
        date=date_str,
        signals=loaded_signals,
        trades=loaded_trades,
    )
    review.compute_stats()
    save_review(review)
    loaded_review = load_review(date_str)
    assert loaded_review is not None
    assert loaded_review.trade_count == 1
    assert loaded_review.accuracy == 100.0
    assert loaded_review.total_pnl == 500.0


def test_daily_review_stats():
    """测试复盘统计计算"""
    review = DailyReview(
        date="2026-05-25",
        trades=[
            IntradayTrade(date="2026-05-25", commodity="RB", direction=Direction.LONG, pnl=100, status=TradeStatus.WIN),
            IntradayTrade(date="2026-05-25", commodity="M", direction=Direction.SHORT, pnl=-50, status=TradeStatus.LOSS),
            IntradayTrade(date="2026-05-25", commodity="CU", direction=Direction.NO_TRADE, pnl=0, status=TradeStatus.NO_TRADE),
        ],
    )
    review.compute_stats()
    assert review.trade_count == 2
    assert review.win_count == 1
    assert review.loss_count == 1
    assert review.accuracy == 50.0
    assert review.total_pnl == 50.0
    assert review.avg_pnl == 25.0
