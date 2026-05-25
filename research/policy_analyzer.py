"""
政策分析引擎
- 监控各部委政策发布
- 使用 LLM 分析政策文本
- 评估政策对期货品种的影响
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

import yaml
import aiohttp
from bs4 import BeautifulSoup

from research.llm_integration import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class PolicyNews:
    """政策新闻条目"""
    title: str
    source: str
    url: str
    publish_time: datetime
    content: str
    ministry: str
    priority: int


class PolicyAnalyzer:
    """政策分析引擎"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.llm_client: Optional[LLMClient] = None
        self.policy_sources = self.config.get("policy_sources", {})
        self.commodities = self.config.get("monitored_commodities", {})
    
    async def _get_llm(self) -> LLMClient:
        if self.llm_client is None:
            self.llm_client = LLMClient()
        return self.llm_client
    
    def get_commodities_by_ministry(self, ministry: str) -> List[Dict[str, str]]:
        """获取某部委相关的期货品种"""
        results = []
        for group, commodities in self.commodities.items():
            for comm in commodities:
                if ministry in comm.get("related_policies", []):
                    results.append(comm)
        return results
    
    async def fetch_policy_news(self, days_back: int = 1) -> List[PolicyNews]:
        """
        抓取政策新闻
        当前实现为简化版，后续可扩展为RSS/爬虫
        """
        # TODO: 实现各部委网站的实际抓取逻辑
        # 当前返回模拟结构，提示需要用户接入实际数据源
        logger.info("政策新闻抓取: 当前为框架实现，需要接入实际数据源")
        return []
    
    async def analyze_policy(
        self,
        policy_title: str,
        policy_content: str,
        ministry: str,
    ) -> Optional[Dict[str, Any]]:
        """
        分析单条政策，评估对期货品种的影响
        """
        related = self.get_commodities_by_ministry(ministry)
        if not related:
            logger.info(f"{ministry} 暂无监控品种映射")
            return None
        
        commodity_names = [c["name"] for c in related]
        llm = await self._get_llm()
        
        try:
            result = await llm.analyze_policy_text(
                policy_title=policy_title,
                policy_content=policy_content,
                related_commodities=commodity_names,
            )
            result["ministry"] = ministry
            result["policy_title"] = policy_title
            result["analyzed_at"] = datetime.now().isoformat()
            return result
        except Exception as e:
            logger.error(f"政策分析失败: {e}")
            return None
    
    async def batch_analyze(
        self,
        policies: List[PolicyNews],
    ) -> List[Dict[str, Any]]:
        """批量分析政策"""
        results = []
        for policy in policies:
            analysis = await self.analyze_policy(
                policy_title=policy.title,
                policy_content=policy.content,
                ministry=policy.ministry,
            )
            if analysis:
                results.append(analysis)
        return results
    
    def filter_high_impact(
        self,
        analyses: List[Dict[str, Any]],
        min_confidence: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """过滤高影响度政策"""
        high_impact = []
        for analysis in analyses:
            confidence = analysis.get("confidence", 0)
            if confidence >= min_confidence:
                # 检查是否有强影响的品种
                for impact in analysis.get("direct_impacts", []):
                    if impact.get("strength") in ["强", "中"]:
                        high_impact.append(analysis)
                        break
        return high_impact
    
    async def generate_policy_summary(
        self,
        analyses: List[Dict[str, Any]],
    ) -> str:
        """
        生成政策摘要报告
        """
        if not analyses:
            return "近期无重大政策变化。"
        
        llm = await self._get_llm()
        
        # 构建摘要 prompt
        analyses_text = json.dumps(analyses, ensure_ascii=False, indent=2)
        
        messages = [
            {
                "role": "system",
                "content": "你是一位期货宏观分析师，请根据政策分析结果，生成简洁的政策摘要。"
            },
            {
                "role": "user",
                "content": f"""以下是对近期政策的分析结果，请生成一份政策摘要：

{analyses_text}

要求：
1. 按品种分组，说明各品种面临的政策环境
2. 指出政策转向或新增的重大变化
3. 评估市场预期程度
4. 输出 markdown 格式，简洁明了
"""
            }
        ]
        
        response = await llm.chat(messages=messages, temperature=0.3)
        return response.content
    
    async def close(self):
        if self.llm_client:
            await self.llm_client.close()
