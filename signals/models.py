"""
信号数据模型 — 使用 Pydantic 定义交易信号的结构
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class PolicyImpact(BaseModel):
    """政策影响详情"""
    policy_title: str
    policy_source: str                          # 发布部委
    policy_level: str                           # 层级
    policy_type: str                            # 类型
    direction: Direction                        # 对品种的方向
    mechanism: str                              # 影响机制
    strength: str = Field(..., pattern="^(强|中|弱)$")
    time_horizon: str = Field(..., pattern="^(短期|中期|长期)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    key_quotes: List[str] = []
    is_direction_change: bool = False


class FundamentalState(BaseModel):
    """基本面状态"""
    inventory_status: str                       # 库存状态（高/中/低/去库中/累库中）
    supply_demand_gap: Optional[str] = None     # 供需缺口描述
    basis_structure: Optional[str] = None       # 基差结构（contango/backwardation）
    profit_distribution: Optional[str] = None   # 利润分布
    seasonal_factor: Optional[str] = None       # 季节性因素
    inventory_cycle_phase: Optional[str] = None # 库存周期阶段
    score: float = Field(..., ge=0.0, le=1.0)   # 基本面得分


class ChainTransmission(BaseModel):
    """产业链传导"""
    upstream: str                               # 上游状态
    midstream: str                              # 中游状态
    downstream: str                             # 下游状态
    transmission_path: List[str]                # 传导路径
    bottleneck: Optional[str] = None            # 瓶颈环节
    score: float = Field(..., ge=0.0, le=1.0)   # 产业链逻辑得分


class TradingSignal(BaseModel):
    """交易信号完整模型"""
    
    # 基本信息
    id: str = Field(..., description="信号唯一ID")
    created_at: datetime = Field(default_factory=datetime.now)
    commodity_code: str
    commodity_name: str
    exchange: str
    
    # 信号方向
    direction: Direction
    confidence: float = Field(..., ge=0.0, le=1.0, description="综合置信度")
    conviction_level: str = Field(..., pattern="^(高|中|低)$")
    
    # 驱动因素
    catalyst: str = Field(..., description="核心触发因素")
    policy_driver: Optional[PolicyImpact] = None
    fundamental_driver: Optional[FundamentalState] = None
    chain_driver: Optional[ChainTransmission] = None
    
    # 交易逻辑
    core_logic: str = Field(..., description="核心交易逻辑")
    entry_conditions: List[str] = []
    stop_loss_logic: str
    target_logic: str
    holding_period_days: int = Field(..., ge=1, le=90)
    risk_level: RiskLevel
    position_sizing: str = Field(..., pattern="^(轻仓|适中|重仓)$")
    
    # 风险与确认
    risk_factors: List[str] = []
    required_confirmations: List[str] = []
    
    # 元数据
    source_commodities: List[str] = []          # 相关品种（用于板块风控）
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # 执行状态（后续填充）
    status: str = "PENDING"                     # PENDING / ACTIVE / CLOSED / CANCELLED
    executed_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    pnl: Optional[float] = None
    review_notes: Optional[str] = None


class SignalBatch(BaseModel):
    """信号批次"""
    batch_id: str
    generated_at: datetime = Field(default_factory=datetime.now)
    signals: List[TradingSignal]
    market_summary: Optional[str] = None
    policy_summary: Optional[str] = None
