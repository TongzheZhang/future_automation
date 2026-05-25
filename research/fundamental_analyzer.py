"""
基本面分析引擎
- 整合基本面数据
- 评估供需状态、库存周期、基差结构
- 生成基本面评分
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

import yaml

from research.llm_integration import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class FundamentalData:
    """基本面数据点"""
    commodity: str
    inventory: Optional[float] = None           # 库存量
    inventory_yoy: Optional[float] = None       # 库存同比
    inventory_mom: Optional[float] = None       # 库存环比
    operating_rate: Optional[float] = None      # 开工率
    profit: Optional[float] = None              # 利润
    basis: Optional[float] = None               # 基差
    import_profit: Optional[float] = None       # 进口利润
    spot_price: Optional[float] = None          # 现货价
    futures_price: Optional[float] = None       # 期货价
    data_date: Optional[str] = None


class FundamentalAnalyzer:
    """基本面分析引擎"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.llm_client: Optional[LLMClient] = None
    
    async def _get_llm(self) -> LLMClient:
        if self.llm_client is None:
            self.llm_client = LLMClient()
        return self.llm_client
    
    def calculate_inventory_cycle(
        self,
        inventory_mom: float,
        demand_indicator: float,  # 正数表示需求改善，负数表示需求恶化
    ) -> str:
        """
        判断库存周期阶段
        """
        if inventory_mom > 0 and demand_indicator > 0:
            return "主动补库"
        elif inventory_mom > 0 and demand_indicator < 0:
            return "被动补库"
        elif inventory_mom < 0 and demand_indicator < 0:
            return "主动去库"
        else:
            return "被动去库"
    
    def score_fundamental(
        self,
        data: FundamentalData,
        direction_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        基于原始数据计算基本面评分
        这是一个启发式评分，后续应由 LLM 进行更 nuanced 的评估
        """
        score = 0.5  # 中性起点
        factors = []
        
        # 库存因素
        if data.inventory_yoy is not None:
            if data.inventory_yoy < -10:  # 库存大幅低于去年
                score += 0.15
                factors.append("库存同比大幅下降，供应偏紧")
            elif data.inventory_yoy > 20:  # 库存大幅高于去年
                score -= 0.15
                factors.append("库存同比大幅累积，供应宽松")
        
        if data.inventory_mom is not None:
            if data.inventory_mom < -5:
                score += 0.1
                factors.append("库存环比去库")
            elif data.inventory_mom > 5:
                score -= 0.1
                factors.append("库存环比累库")
        
        # 基差因素
        if data.basis is not None:
            if data.basis > 0:  # 现货升水
                score += 0.1
                factors.append("现货升水，现货紧张")
            else:
                score -= 0.05
                factors.append("现货贴水")
        
        # 利润因素
        if data.profit is not None:
            if data.profit < 0:  # 亏损
                score += 0.05  # 可能触发减产
                factors.append("行业亏损，存在减产预期")
            elif data.profit > 500:  # 高利润（假设单位）
                score -= 0.05
                factors.append("高利润刺激供应")
        
        # 限制分数范围
        score = max(0.0, min(1.0, score))
        
        return {
            "commodity": data.commodity,
            "score": score,
            "factors": factors,
            "inventory_status": self._inventory_status(data),
            "basis_structure": "backwardation" if (data.basis and data.basis > 0) else "contango",
            "raw_data": {
                "inventory": data.inventory,
                "inventory_yoy": data.inventory_yoy,
                "inventory_mom": data.inventory_mom,
                "operating_rate": data.operating_rate,
                "profit": data.profit,
                "basis": data.basis,
            },
        }
    
    def _inventory_status(self, data: FundamentalData) -> str:
        """判断库存状态"""
        if data.inventory_yoy is None:
            return "未知"
        if data.inventory_yoy < -15:
            return "极低"
        elif data.inventory_yoy < -5:
            return "偏低"
        elif data.inventory_yoy < 5:
            return "中性"
        elif data.inventory_yoy < 15:
            return "偏高"
        else:
            return "极高"
    
    async def deep_analyze_with_llm(
        self,
        commodity: str,
        fundamental_data: Dict[str, Any],
        alpha_pai_research: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        使用 LLM 进行深度基本面分析
        """
        llm = await self._get_llm()
        
        system_prompt = """你是一位资深的大宗商品基本面分析师。请基于提供的数据，对品种的基本面进行深度分析。
输出 JSON 格式。"""
        
        user_prompt = f"""品种：{commodity}

【基本面数据】
{json.dumps(fundamental_data, ensure_ascii=False, indent=2)}

"""
        if alpha_pai_research:
            user_prompt += f"""【Alpha 派投研信息】
{alpha_pai_research}

"""
        
        user_prompt += """请输出以下 JSON 格式：
{
    "supply_assessment": "供应端评估",
    "demand_assessment": "需求端评估",
    "inventory_assessment": "库存评估",
    "profit_assessment": "利润与开工评估",
    "basis_assessment": "基差与期限结构评估",
    "seasonal_factor": "季节性因素",
    "overall_score": 0.0-1.0,
    "direction_bias": "偏多/偏空/中性",
    "key_risks": ["风险因素"],
    "data_gaps": ["缺失的关键数据"]
}

确保 JSON 格式正确。"""
        
        response = await llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        
        from research.llm_integration import extract_json_from_text
        try:
            return extract_json_from_text(response.content)
        except json.JSONDecodeError:
            logger.error(f"基本面分析返回非 JSON: {response.content[:500]}")
            raise
    
    async def close(self):
        if self.llm_client:
            await self.llm_client.close()
