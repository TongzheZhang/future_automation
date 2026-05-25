"""
政策新闻采集器
- 基于 Brave Search 监控各部委政策发布
- 支持按品种关键词过滤
- 输出结构化政策新闻
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

import yaml

from data.collectors.brave_search import BraveSearchCollector

logger = logging.getLogger(__name__)


@dataclass
class PolicyNewsItem:
    title: str
    source: str
    url: str
    publish_time: Optional[datetime]
    summary: str
    ministry: str
    priority: int
    matched_keywords: List[str]


class PolicyNewsCollector:
    """政策新闻采集器"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.sources = self.config.get("policy_sources", {})
        self.commodities = self.config.get("monitored_commodities", {})
        self.brave = BraveSearchCollector(config_path)
    
    def get_commodities_by_ministry(self, ministry: str) -> List[Dict[str, Any]]:
        """获取某部委相关的期货品种"""
        results = []
        for group, commodities in self.commodities.items():
            for comm in commodities:
                if ministry in comm.get("related_policies", []):
                    results.append(comm)
        return results
    
    async def fetch_from_ministry(self, ministry_config: Dict[str, Any], days: int = 1) -> List[PolicyNewsItem]:
        """
        从单个部委源抓取政策（通过 Brave Search）
        """
        ministry_name = ministry_config["name"]
        priority = ministry_config.get("priority", 2)
        
        # 获取该部委相关的品种关键词
        related_comms = self.get_commodities_by_ministry(ministry_name)
        if not related_comms:
            return []
        
        # 合并关键词去重
        all_keywords = set()
        for comm in related_comms:
            all_keywords.update(comm.get("alpha_pai_keywords", [comm["name"]]))
        
        keywords_list = list(all_keywords)[:5]  # 限制关键词数量
        
        # 使用 Brave Search 搜索
        try:
            results = await self.brave.search_policy_news(
                ministry=ministry_name,
                commodity_keywords=keywords_list,
                days=days,
            )
        except Exception as e:
            logger.error(f"搜索 {ministry_name} 政策失败: {e}")
            return []
        
        news_items = []
        for result in results:
            # 检查是否真正相关
            matched = [kw for kw in keywords_list if kw in result.title or kw in result.description]
            if not matched:
                continue
            
            news_items.append(PolicyNewsItem(
                title=result.title,
                source=result.source or ministry_name,
                url=result.url,
                publish_time=None,  # Brave 不总是提供精确时间
                summary=result.description,
                ministry=ministry_name,
                priority=priority,
                matched_keywords=matched,
            ))
        
        logger.info(f"{ministry_name}: 找到 {len(news_items)} 条相关政策")
        return news_items
    
    async def fetch_all(self, days_back: int = 1) -> List[PolicyNewsItem]:
        """
        抓取所有配置源的政策新闻
        """
        all_news = []
        
        # 各部委
        ministries = self.sources.get("ministries", [])
        for ministry in ministries:
            try:
                news = await self.fetch_from_ministry(ministry, days=days_back)
                all_news.extend(news)
            except Exception as e:
                logger.error(f"抓取 {ministry['name']} 失败: {e}")
        
        # 按优先级排序
        all_news.sort(key=lambda x: x.priority)
        
        # 去重（基于 URL）
        seen_urls = set()
        unique_news = []
        for item in all_news:
            if item.url not in seen_urls:
                seen_urls.add(item.url)
                unique_news.append(item)
        
        logger.info(f"政策新闻采集完成: 共 {len(unique_news)} 条去重后政策")
        return unique_news
    
    async def close(self):
        await self.brave.close()
