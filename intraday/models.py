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
    "A": 10,
    "AG": 15,
    "AL": 5,
    "AO": 20,
    "AP": 10,
    "AU": 1000,
    "B": 10,
    "BC": 5,
    "BR": 5,
    "BU": 10,
    "C": 10,
    "CF": 5,
    "CJ": 5,
    "CS": 10,
    "CU": 5,
    "CY": 5,
    "EB": 5,
    "EG": 10,
    "FG": 20,
    "FU": 10,
    "HC": 10,
    "I": 100,
    "IC": 200,
    "IF": 300,
    "IH": 300,
    "IM": 200,
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
    "RR": 10,
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
    "T": 10000,
    "TA": 5,
    "TL": 10000,
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
    def should_trade(self, min_confidence: int = 7) -> bool:
        return self.direction != Direction.NO_TRADE and self.confidence >= min_confidence


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


class CognitionItem(BaseModel):
    """认知条目 — 从复盘中提取的教训"""
    
    id: str = Field(..., description="唯一ID")
    lesson: str = Field(..., description="核心教训")
    category: str = Field(..., description="类别: signal_filter / entry_timing / exit_timing / risk_control / market_regime / general")
    confidence: int = Field(default=5, ge=0, le=10, description="可靠度 0-10")
    verification_count: int = Field(default=0, description="验证次数")
    win_count: int = Field(default=0, description="验证中胜场")
    loss_count: int = Field(default=0, description="验证中败场")
    affected_commodities: List[str] = Field(default_factory=list)
    source_trade_date: str = Field(default="", description="来源交易日期")
    created_at: datetime = Field(default_factory=datetime.now)
    status: str = Field(default="pending", description="pending / validated / invalidated")
    
    # 防过拟合字段（新增）
    source_type: str = Field(default="own_trade", description="来源: own_trade / market_observation")
    cross_commodity_count: int = Field(default=1, ge=1, description="该认知在多少个品种上被观察到")
    cross_date_count: int = Field(default=1, ge=1, description="该认知在多少个交易日被观察到")
    uniqueness_score: int = Field(default=5, ge=0, le=10, description="独特性评分 0-10，越低越普适")
    
    def record_verification(self, is_win: bool):
        """记录一次验证结果"""
        self.verification_count += 1
        if is_win:
            self.win_count += 1
        else:
            self.loss_count += 1
        
        # 更新可靠度
        if self.verification_count >= 3:
            win_rate = self.win_count / self.verification_count
            if win_rate >= 0.6:
                self.status = "validated"
                self.confidence = min(10, self.confidence + 1)
            elif win_rate <= 0.3 and self.verification_count >= 5:
                self.status = "invalidated"
                self.confidence = max(0, self.confidence - 2)
    
    def to_prompt_rule(self) -> str:
        """转换为 prompt 经验规则格式"""
        status_emoji = {"pending": "⏳", "validated": "✅", "invalidated": "❌"}.get(self.status, "⏳")
        source_emoji = {"own_trade": "🎯", "market_observation": "👁"}.get(self.source_type, "🎯")
        return f"{status_emoji} {source_emoji} [{self.category}] {self.lesson} (可靠度:{self.confidence}/10,普适:{self.uniqueness_score}/10,品种:{self.cross_commodity_count},日期:{self.cross_date_count})"
    
    def is_generally_applicable(self) -> bool:
        """判断是否具有足够的普适性"""
        return self.cross_commodity_count >= 3 and self.cross_date_count >= 3 and self.uniqueness_score <= 5


class CognitionLibrary(BaseModel):
    """认知库 — 所有 lessons 的集合"""
    
    version: int = Field(default=1)
    updated_at: datetime = Field(default_factory=datetime.now)
    items: List[CognitionItem] = Field(default_factory=list)
    evolved_prompt_additions: str = Field(default="", description="累积追加到 strategy prompt 的经验规则文本")
    
    def get_validated_items(self, min_confidence: int = 7, require_generality: bool = False) -> List[CognitionItem]:
        """获取已验证的高可靠度条目
        
        Args:
            min_confidence: 最低可靠度
            require_generality: 是否要求具备跨品种/跨日期普适性（默认False保持向后兼容）
        """
        result = [
            item for item in self.items
            if item.confidence >= min_confidence and item.status in ["pending", "validated"]
        ]
        if require_generality:
            result = [item for item in result if item.is_generally_applicable()]
        return result
    
    def get_items_by_category(self, category: str) -> List[CognitionItem]:
        """按类别获取条目"""
        return [item for item in self.items if item.category == category]
    
    def rebuild_prompt_additions(self, max_length: int = 2000) -> str:
        """重新构建 prompt 追加文本（优先使用具备普适性的认知，冷启动期使用高置信度认知）"""
        validated = self.get_validated_items(require_generality=True)

        # 冷启动期：若无普适性认知，退而使用高置信度(≥8)认知作为"试用期规则"
        if not validated:
            validated = self.get_validated_items(min_confidence=8, require_generality=False)
            is_bootstrapping = True
        else:
            is_bootstrapping = False

        if not validated:
            return ""

        lines = ["\n【系统进化经验规则（基于历史复盘自动积累）】"]
        if is_bootstrapping:
            lines.append("（注：以下规则处于试用期，尚未经过充分跨日期验证，请结合自身判断参考使用）")

        for i, item in enumerate(validated, 1):
            rule = item.to_prompt_rule()
            if is_bootstrapping:
                rule = rule.replace("⏳", "⏳[试用期]", 1)
            lines.append(f"{i}. {rule}")

        text = "\n".join(lines)
        if len(text) > max_length:
            text = text[:max_length] + "\n...（规则过多，已截断）"
        return text


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
    
    # 进化相关
    extracted_cognitions: List[CognitionItem] = Field(default_factory=list)
    evolution_report: str = ""
    
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
