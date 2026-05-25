"""
产业链映射与传导分析
- 定义各品种的产业链关系
- 分析利润传导与价格传导
- 识别产业链套利机会
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from research.llm_integration import LLMClient

logger = logging.getLogger(__name__)


# 产业链定义
COMMODITY_CHAINS = {
    "black": {
        "name": "黑色系产业链",
        "nodes": {
            "iron_ore": {"name": "铁矿石", "commodities": ["I"]},
            "coke": {"name": "焦炭", "commodities": ["J", "JM"]},
            "steel": {"name": "钢材", "commodities": ["RB", "HC"]},
            "downstream": {"name": "地产/基建", "commodities": []},
        },
        "edges": [
            {"from": "iron_ore", "to": "steel", "relation": "原料"},
            {"from": "coke", "to": "steel", "relation": "原料"},
            {"from": "steel", "to": "downstream", "relation": "需求"},
        ],
    },
    "energy_chemical": {
        "name": "能源化工产业链",
        "nodes": {
            "crude": {"name": "原油", "commodities": ["SC"]},
            "px": {"name": "PX", "commodities": []},
            "pta": {"name": "PTA", "commodities": ["TA"]},
            "meg": {"name": "MEG", "commodities": ["EG"]},
            "polyester": {"name": "聚酯", "commodities": []},
            "textile": {"name": "纺织", "commodities": []},
        },
        "edges": [
            {"from": "crude", "to": "px", "relation": "上游"},
            {"from": "px", "to": "pta", "relation": "原料"},
            {"from": "pta", "to": "polyester", "relation": "原料"},
            {"from": "polyester", "to": "textile", "relation": "需求"},
        ],
    },
    "agriculture_soy": {
        "name": "大豆压榨产业链",
        "nodes": {
            "soybean": {"name": "大豆", "commodities": ["A"]},
            "soybean_meal": {"name": "豆粕", "commodities": ["M"]},
            "soybean_oil": {"name": "豆油", "commodities": ["Y"]},
            "feed": {"name": "饲料", "commodities": []},
            "hog": {"name": "生猪", "commodities": ["LH"]},
        },
        "edges": [
            {"from": "soybean", "to": "soybean_meal", "relation": "压榨产出"},
            {"from": "soybean", "to": "soybean_oil", "relation": "压榨产出"},
            {"from": "soybean_meal", "to": "feed", "relation": "原料"},
            {"from": "feed", "to": "hog", "relation": "养殖"},
        ],
    },
    "copper": {
        "name": "铜产业链",
        "nodes": {
            "copper_mine": {"name": "铜矿", "commodities": []},
            "copper_concentrate": {"name": "铜精矿", "commodities": []},
            "copper_smelting": {"name": "铜冶炼", "commodities": ["CU"]},
            "coprod_semis": {"name": "铜加工", "commodities": []},
            "power": {"name": "电力", "commodities": []},
            "ev": {"name": "新能源汽车", "commodities": []},
        },
        "edges": [
            {"from": "copper_mine", "to": "copper_concentrate", "relation": "采矿"},
            {"from": "copper_concentrate", "to": "copper_smelting", "relation": "冶炼原料"},
            {"from": "copper_smelting", "to": "coprod_semis", "relation": "精炼铜"},
            {"from": "coprod_semis", "to": "power", "relation": "下游需求"},
            {"from": "coprod_semis", "to": "ev", "relation": "下游需求"},
        ],
    },
    "aluminum": {
        "name": "铝产业链",
        "nodes": {
            "bauxite": {"name": "铝土矿", "commodities": []},
            "alumina": {"name": "氧化铝", "commodities": ["AO"]},
            "electrolytic_aluminum": {"name": "电解铝", "commodities": ["AL"]},
            "aluminum_semis": {"name": "铝加工", "commodities": []},
            "construction": {"name": "建筑", "commodities": []},
            "transportation": {"name": "交通", "commodities": []},
        },
        "edges": [
            {"from": "bauxite", "to": "alumina", "relation": "原料"},
            {"from": "alumina", "to": "electrolytic_aluminum", "relation": "原料"},
            {"from": "electrolytic_aluminum", "to": "aluminum_semis", "relation": "加工"},
            {"from": "aluminum_semis", "to": "construction", "relation": "需求"},
            {"from": "aluminum_semis", "to": "transportation", "relation": "需求"},
        ],
    },
}


class ChainMapper:
    """产业链映射器"""
    
    def __init__(self):
        self.chains = COMMODITY_CHAINS
        self.llm_client: Optional[LLMClient] = None
    
    async def _get_llm(self) -> LLMClient:
        if self.llm_client is None:
            self.llm_client = LLMClient()
        return self.llm_client
    
    def get_chain_by_commodity(self, commodity_code: str) -> Optional[str]:
        """根据品种代码找到所属产业链"""
        for chain_id, chain in self.chains.items():
            for node_id, node in chain["nodes"].items():
                if commodity_code in node.get("commodities", []):
                    return chain_id
        return None
    
    def get_related_commodities(self, commodity_code: str) -> List[str]:
        """获取同一产业链的相关品种"""
        chain_id = self.get_chain_by_commodity(commodity_code)
        if not chain_id:
            return []
        
        chain = self.chains[chain_id]
        related = []
        for node in chain["nodes"].values():
            related.extend(node.get("commodities", []))
        
        return list(set(related) - {commodity_code})
    
    def get_chain_description(self, chain_id: str) -> str:
        """获取产业链文字描述"""
        chain = self.chains.get(chain_id)
        if not chain:
            return ""
        
        lines = [f"## {chain['name']}", ""]
        for node_id, node in chain["nodes"].items():
            commodities = ", ".join(node.get("commodities", [])) or "无期货品种"
            lines.append(f"- **{node['name']}** ({commodities})")
        
        lines.append("")
        lines.append("**传导关系**:")
        for edge in chain["edges"]:
            from_node = chain["nodes"][edge["from"]]["name"]
            to_node = chain["nodes"][edge["to"]]["name"]
            lines.append(f"- {from_node} → [{edge['relation']}] → {to_node}")
        
        return "\n".join(lines)
    
    async def analyze_transmission(
        self,
        commodity: str,
        policy_impact: Optional[Dict[str, Any]] = None,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        分析产业链传导
        """
        chain_id = self.get_chain_by_commodity(commodity)
        if not chain_id:
            return {
                "commodity": commodity,
                "chain": None,
                "transmission_analysis": "该品种暂无产业链定义",
                "score": 0.5,
            }
        
        chain = self.chains[chain_id]
        chain_desc = self.get_chain_description(chain_id)
        
        llm = await self._get_llm()
        
        system_prompt = """你是一位产业链分析专家。请分析给定政策或市场变化在产业链中的传导路径。
输出 JSON 格式。"""
        
        user_prompt = f"""品种：{commodity}

【所属产业链】
{chain_desc}

"""
        if policy_impact:
            user_prompt += f"""【政策影响】
{json.dumps(policy_impact, ensure_ascii=False, indent=2)}

"""
        
        if market_data:
            user_prompt += f"""【市场数据】
{json.dumps(market_data, ensure_ascii=False, indent=2)}

"""
        
        user_prompt += """请输出以下 JSON 格式：
{
    "affected_nodes": ["受影响的产业链环节"],
    "transmission_path": ["传导路径的每一步，含时序"],
    "bottleneck": "瓶颈环节（如有）",
    "price_impact_sequence": [{"commodity": "品种", "direction": "涨/跌", "timing": "立即/短期/中期", "magnitude": "幅度估计"}],
    "spread_opportunity": "是否存在跨品种套利机会",
    "overall_score": 0.0-1.0,
    "confidence": 0.0-1.0,
    "key_assumptions": ["关键假设"]
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
            result = extract_json_from_text(response.content)
            result["chain_id"] = chain_id
            result["chain_name"] = chain["name"]
            return result
        except json.JSONDecodeError:
            logger.error(f"产业链分析返回非 JSON: {response.content[:500]}")
            raise
    
    async def close(self):
        if self.llm_client:
            await self.llm_client.close()
