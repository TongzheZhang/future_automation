"""
日内T+0交易数据模型
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NO_TRADE = "NO_TRADE"


class TradeStatus(str, Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"
    NO_TRADE = "NO_TRADE"
    PENDING = "PENDING"


# 品种每手合约乘数（吨/手），用于盈亏计算
CONTRACT_SIZE: Dict[str, int] = {
    "AG": 15,
    "AL": 5,
    "AO": 20,
    "AP": 10,
    "AU": 1000,
    "BC": 5,
    "BR": 5,
    "BU": 10,
    "C": 10,
    "CF": 5,
    "CJ": 5,
    "CS": 10,
    "CU": 5,
    "EB": 5,
    "EG": 10,
    "FG": 20,
    "HC": 10,
    "I": 100,
    "J": 100,
    "JD": 10,
    "JM": 60,
    "L": 5,
    "LC": 1,
    "LH": 16,
    "LU": 10,
    "M": 10,
    "MA": 10,
    "NI": 1,
    "NR": 10,
    "OI": 10,
    "P": 10,
    "PB": 5,
    "PF": 5,
    "PG": 20,
    "PK": 5,
    "PP": 5,
    "PX": 5,
    "RB": 10,
    "RM": 10,
    "RU": 10,
    "SA": 20,
    "SC": 1000,
    "SF": 5,
    "SH": 30,
    "SI": 5,
    "SM": 5,
    "SN": 1,
    "SP": 10,
    "SR": 10,
    "SS": 5,
    "TA": 5,
    "UR": 20,
    "V": 5,
    "Y": 10,
    "ZN": 5,
}


class MarketSnapshotData(BaseModel):
    """开盘行情快照（简化版，用于序列化）"""
    commodity: str = ""
    name: str = ""
    time: str = ""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    last: float = 0.0
    prev_settle: float = 0.0
    settle: float = 0.0          # 当日结算价（盘中可能为0，盘后更新）
    bid: float = 0.0
    ask: float = 0.0
    open_interest: float = 0.0
    volume: float = 0.0
    date: str = ""
    gap_pct: float = 0.0
    change_pct: float = 0.0
    amplitude_pct: float = 0.0


class IntradaySignal(BaseModel):
    """日内交易信号"""
    
    date: str = Field(..., description="交易日 YYYY-MM-DD")
    commodity: str = Field(..., description="品种代码")
    commodity_name: str = Field(default="", description="品种名称")
    
    direction: Direction = Field(default=Direction.NO_TRADE)
    entry_price: float = Field(default=0.0, description="建议入场价")
    confidence: int = Field(default=0, ge=0, le=10, description="确定性评分 0-10")
    stop_loss_price: float = Field(default=0.0, description="日内止损价")
    target_price: float = Field(default=0.0, description="日内目标价")
    
    core_logic: str = Field(default="", description="核心交易逻辑")
    overnight_news: str = Field(default="", description="隔夜关键新闻摘要")
    market_snapshot: MarketSnapshotData = Field(default_factory=MarketSnapshotData)
    
    generated_at: datetime = Field(default_factory=datetime.now)
    
    # 是否建议交易
    def should_trade(self) -> bool:
        return self.direction != Direction.NO_TRADE and self.confidence >= 7


class IntradayTrade(BaseModel):
    """日内交易记录"""
    
    date: str
    commodity: str
    direction: Direction
    
    # 信号信息
    signal_entry: float = 0.0
    signal_stop: float = 0.0
    signal_target: float = 0.0
    confidence: int = 0
    core_logic: str = ""
    
    # 实际成交
    actual_entry: float = 0.0
    actual_exit: float = 0.0
    
    # 日内行情
    day_high: float = 0.0
    day_low: float = 0.0
    day_close: float = 0.0
    
    # 盈亏（按1手，简化计算）
    pnl: float = 0.0
    max_drawdown: float = 0.0
    status: TradeStatus = TradeStatus.PENDING
    
    # 复盘
    review_notes: str = ""
    lessons: List[str] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None
    
    def calculate_pnl(self, contract_size: Optional[float] = None) -> float:
        """计算盈亏（元），按合约乘数*价格差，不考虑手续费"""
        if contract_size is None:
            contract_size = CONTRACT_SIZE.get(self.commodity, 1)

        if self.direction == Direction.LONG:
            self.pnl = round((self.actual_exit - self.actual_entry) * contract_size, 2)
        elif self.direction == Direction.SHORT:
            self.pnl = round((self.actual_entry - self.actual_exit) * contract_size, 2)
        else:
            self.pnl = 0.0
        
        # 判断状态
        if self.pnl > 0:
            self.status = TradeStatus.WIN
        elif self.pnl < 0:
            self.status = TradeStatus.LOSS
        else:
            self.status = TradeStatus.BREAKEVEN
        
        return self.pnl


class DailyReview(BaseModel):
    """每日复盘"""
    
    date: str
    signals: List[IntradaySignal] = Field(default_factory=list)
    trades: List[IntradayTrade] = Field(default_factory=list)
    
    # 统计指标
    total_signals: int = 0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    accuracy: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    
    # LLM 复盘
    review_summary: str = ""
    lessons: List[str] = Field(default_factory=list)
    strategy_adjustments: List[str] = Field(default_factory=list)
    
    generated_at: datetime = Field(default_factory=datetime.now)
    
    def compute_stats(self):
        """计算统计指标"""
        self.total_signals = len(self.signals)
        self.trade_count = len([t for t in self.trades if t.status != TradeStatus.NO_TRADE])
        self.win_count = len([t for t in self.trades if t.status == TradeStatus.WIN])
        self.loss_count = len([t for t in self.trades if t.status == TradeStatus.LOSS])
        
        if self.trade_count > 0:
            self.accuracy = round(self.win_count / self.trade_count * 100, 2)
            self.total_pnl = round(sum(t.pnl for t in self.trades), 2)
            self.avg_pnl = round(self.total_pnl / self.trade_count, 2)
