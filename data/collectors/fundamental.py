"""
基本面数据采集器
- 通过 Brave Search 获取最新基本面数据和分析
- 通过 Alpha 派获取深度投研数据
- 整合多源信息生成基本面画像
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

from data.collectors.brave_search import BraveSearchCollector
from data.collectors.alpha_pai import AlphaPaiCollector

logger = logging.getLogger(__name__)


@dataclass
class FundamentalDataPoint:
    commodity: str
    indicator: str          # inventory / operating_rate / profit / basis / supply_demand
    value: Optional[float] = None
    description: str = ""
    unit: str = ""
    date: Optional[str] = None
    source: str = ""
    yoy_change: Optional[float] = None
    mom_change: Optional[float] = None


class FundamentalCollector:
    """基本面数据采集器"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.brave = BraveSearchCollector(config_path)
        self.alpha_pai = AlphaPaiCollector()
        self.data_cache: Dict[str, Dict[str, Any]] = {}
    
    async def fetch_inventory(self, commodity: str, keywords: List[str]) -> Optional[FundamentalDataPoint]:
        """获取库存数据"""
        query = f"{' '.join(keywords)} 库存 最新数据"
        results = await self.brave.search(query, count=5, freshness="pw")
        
        if results:
            return FundamentalDataPoint(
                commodity=commodity,
                indicator="inventory",
                description=results[0].description[:200],
                source=results[0].url,
            )
        return None
    
    async def fetch_supply_demand(self, commodity: str, keywords: List[str]) -> Optional[FundamentalDataPoint]:
        """获取供需分析"""
        query = f"{' '.join(keywords)} 供需 分析"
        results = await self.brave.search(query, count=5, freshness="pw")
        
        if results:
            return FundamentalDataPoint(
                commodity=commodity,
                indicator="supply_demand",
                description=results[0].description[:200],
                source=results[0].url,
            )
        return None
    
    async def fetch_profit(self, commodity: str, keywords: List[str]) -> Optional[FundamentalDataPoint]:
        """获取利润数据"""
        query = f"{' '.join(keywords)} 利润 生产利润"
        results = await self.brave.search(query, count=5, freshness="pw")
        
        if results:
            return FundamentalDataPoint(
                commodity=commodity,
                indicator="profit",
                description=results[0].description[:200],
                source=results[0].url,
            )
        return None
    
    async def fetch_basis(self, commodity: str, keywords: List[str]) -> Optional[FundamentalDataPoint]:
        """获取基差信息"""
        query = f"{' '.join(keywords)} 基差 现货期货"
        results = await self.brave.search(query, count=5, freshness="pw")
        
        if results:
            return FundamentalDataPoint(
                commodity=commodity,
                indicator="basis",
                description=results[0].description[:200],
                source=results[0].url,
            )
        return None
    
    async def fetch_alpha_pai_research(self, commodity: str, keywords: List[str]) -> Optional[str]:
        """通过 Alpha 派获取深度研究"""
        try:
            question = f"{'/'.join(keywords[:2])} 最新基本面和供需情况如何？"
            result = self.alpha_pai.qa(
                question=question,
                mode="Think",
            )
            return result.answer
        except Exception as e:
            logger.error(f"Alpha 派研究失败 {commodity}: {e}")
            return None
    
    async def fetch_all_for_commodity(self, commodity: str, keywords: List[str]) -> Dict[str, Any]:
        """获取某品种的所有基本面数据"""
        results = {
            "commodity": commodity,
            "timestamp": datetime.now().isoformat(),
            "indicators": {},
            "alpha_pai_research": None,
        }
        
        # 并行获取各类数据
        inventory_task = self.fetch_inventory(commodity, keywords)
        supply_demand_task = self.fetch_supply_demand(commodity, keywords)
        profit_task = self.fetch_profit(commodity, keywords)
        basis_task = self.fetch_basis(commodity, keywords)
        alpha_pai_task = self.fetch_alpha_pai_research(commodity, keywords)
        
        import asyncio
        gathered = await asyncio.gather(
            inventory_task, supply_demand_task, profit_task, basis_task, alpha_pai_task,
            return_exceptions=True,
        )
        
        if gathered[0] and not isinstance(gathered[0], Exception):
            results["indicators"]["inventory"] = gathered[0]
        if gathered[1] and not isinstance(gathered[1], Exception):
            results["indicators"]["supply_demand"] = gathered[1]
        if gathered[2] and not isinstance(gathered[2], Exception):
            results["indicators"]["profit"] = gathered[2]
        if gathered[3] and not isinstance(gathered[3], Exception):
            results["indicators"]["basis"] = gathered[3]
        if gathered[4] and not isinstance(gathered[4], Exception):
            results["alpha_pai_research"] = gathered[4]
        
        self.data_cache[commodity] = results
        return results
    
    async def close(self):
        await self.brave.close()
