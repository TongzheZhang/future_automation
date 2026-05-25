"""
交易信号生成器
- 综合政策分析、基本面分析、产业链分析
- 生成结构化的交易信号
- 使用 LLM 进行综合评估
"""

import json
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from signals.models import (
    TradingSignal, Direction, RiskLevel,
    PolicyImpact, FundamentalState, ChainTransmission,
)
from research.llm_integration import LLMClient
from research.policy_analyzer import PolicyAnalyzer
from research.fundamental_analyzer import FundamentalAnalyzer
from research.chain_mapper import ChainMapper

logger = logging.getLogger(__name__)


class SignalGenerator:
    """交易信号生成器"""
    
    def __init__(self):
        self.llm_client: Optional[LLMClient] = None
    
    async def _get_llm(self) -> LLMClient:
        if self.llm_client is None:
            self.llm_client = LLMClient()
        return self.llm_client
    
    def _map_direction(self, direction_str: str) -> Direction:
        """映射方向字符串"""
        direction_str = direction_str.upper()
        if "LONG" in direction_str or "多" in direction_str:
            return Direction.LONG
        elif "SHORT" in direction_str or "空" in direction_str:
            return Direction.SHORT
        return Direction.NEUTRAL
    
    def _map_risk_level(self, risk_str: str) -> RiskLevel:
        """映射风险等级"""
        risk_str = risk_str.upper()
        if "HIGH" in risk_str or "高" in risk_str:
            return RiskLevel.HIGH
        elif "LOW" in risk_str or "低" in risk_str:
            return RiskLevel.LOW
        return RiskLevel.MEDIUM
    
    async def generate_signal(
        self,
        commodity_code: str,
        commodity_name: str,
        exchange: str,
        policy_analysis: Optional[Dict[str, Any]] = None,
        fundamental_analysis: Optional[Dict[str, Any]] = None,
        chain_analysis: Optional[Dict[str, Any]] = None,
    ) -> Optional[TradingSignal]:
        """
        生成单个品种的交易信号
        """
        if not policy_analysis and not fundamental_analysis and not chain_analysis:
            logger.warning(f"{commodity_code} 无任何分析输入，无法生成信号")
            return None
        
        llm = await self._get_llm()
        
        # 调用 LLM 综合评估
        try:
            synthesis = await llm.synthesize_signal(
                policy_analysis=policy_analysis or {},
                fundamental_analysis=fundamental_analysis or {},
                chain_analysis=chain_analysis or {},
                commodity=commodity_name,
            )
        except Exception as e:
            logger.error(f"信号综合评估失败 ({commodity_code}): {e}")
            return None
        
        direction = self._map_direction(synthesis.get("direction", "NEUTRAL"))
        confidence = synthesis.get("confidence", 0)
        
        # 过滤中性或低置信度信号
        if direction == Direction.NEUTRAL or confidence < 0.6:
            logger.info(f"{commodity_code} 信号被过滤: direction={direction.value}, confidence={confidence}")
            return None
        
        # 构建 PolicyImpact
        policy_driver = None
        if policy_analysis and "direct_impacts" in policy_analysis:
            for impact in policy_analysis.get("direct_impacts", []):
                if impact.get("commodity") == commodity_name or commodity_name in impact.get("commodity", ""):
                    policy_driver = PolicyImpact(
                        policy_title=policy_analysis.get("policy_title", ""),
                        policy_source=policy_analysis.get("ministry", ""),
                        policy_level=policy_analysis.get("policy_level", ""),
                        policy_type=policy_analysis.get("policy_type", ""),
                        direction=direction,
                        mechanism=impact.get("mechanism", ""),
                        strength=impact.get("strength", "中"),
                        time_horizon=impact.get("time_horizon", "中期"),
                        confidence=impact.get("confidence", 0.5),
                        key_quotes=policy_analysis.get("key_quotes", []),
                        is_direction_change=policy_analysis.get("is_direction_change", False),
                    )
                    break
        
        # 构建 FundamentalState
        fundamental_driver = None
        if fundamental_analysis:
            fundamental_driver = FundamentalState(
                inventory_status=fundamental_analysis.get("inventory_assessment", "未知"),
                supply_demand_gap=fundamental_analysis.get("supply_assessment", "") + " | " + fundamental_analysis.get("demand_assessment", ""),
                basis_structure=fundamental_analysis.get("basis_assessment", ""),
                profit_distribution=fundamental_analysis.get("profit_assessment", ""),
                seasonal_factor=fundamental_analysis.get("seasonal_factor", ""),
                inventory_cycle_phase=None,
                score=fundamental_analysis.get("overall_score", 0.5),
            )
        
        # 构建 ChainTransmission
        chain_driver = None
        if chain_analysis:
            chain_driver = ChainTransmission(
                upstream="",
                midstream="",
                downstream="",
                transmission_path=chain_analysis.get("transmission_path", []),
                bottleneck=chain_analysis.get("bottleneck"),
                score=chain_analysis.get("overall_score", 0.5),
            )
        
        signal = TradingSignal(
            id=f"SIG-{uuid.uuid4().hex[:8].upper()}",
            commodity_code=commodity_code,
            commodity_name=commodity_name,
            exchange=exchange,
            direction=direction,
            confidence=confidence,
            conviction_level=synthesis.get("conviction_level", "中"),
            catalyst=synthesis.get("core_logic", "")[:100],
            policy_driver=policy_driver,
            fundamental_driver=fundamental_driver,
            chain_driver=chain_driver,
            core_logic=synthesis.get("core_logic", ""),
            entry_conditions=synthesis.get("entry_conditions", []),
            stop_loss_logic=synthesis.get("stop_loss_logic", ""),
            target_logic=synthesis.get("target_logic", ""),
            holding_period_days=synthesis.get("holding_period_days", 14),
            risk_level=self._map_risk_level(synthesis.get("risk_level", "MEDIUM")),
            position_sizing=synthesis.get("position_sizing", "轻仓"),
            risk_factors=synthesis.get("risk_factors", []),
            required_confirmations=synthesis.get("required_confirmations", []),
        )
        
        logger.info(
            f"生成信号: {signal.id} | {commodity_code} | {direction.value} | "
            f"置信度={confidence:.2f}"
        )
        
        return signal
    
    async def generate_batch(
        self,
        analyses: List[Dict[str, Any]],
    ) -> List[TradingSignal]:
        """
        批量生成信号
        analyses 格式:
        [
            {
                "commodity_code": "...",
                "commodity_name": "...",
                "exchange": "...",
                "policy_analysis": {...},
                "fundamental_analysis": {...},
                "chain_analysis": {...},
            }
        ]
        """
        signals = []
        for analysis in analyses:
            signal = await self.generate_signal(
                commodity_code=analysis["commodity_code"],
                commodity_name=analysis["commodity_name"],
                exchange=analysis["exchange"],
                policy_analysis=analysis.get("policy_analysis"),
                fundamental_analysis=analysis.get("fundamental_analysis"),
                chain_analysis=analysis.get("chain_analysis"),
            )
            if signal:
                signals.append(signal)
        
        # 按置信度排序
        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals
    
    async def close(self):
        if self.llm_client:
            await self.llm_client.close()
