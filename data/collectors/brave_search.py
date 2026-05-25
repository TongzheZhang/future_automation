"""
Brave Search 集成
- 用于搜索政策新闻、基本面信息、行业动态
- 替代传统爬虫，快速获取高质量信息
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

import aiohttp
import yaml

logger = logging.getLogger(__name__)


@dataclass
class BraveSearchResult:
    title: str
    url: str
    description: str
    age: Optional[str] = None
    source: Optional[str] = None


class BraveSearchCollector:
    """Brave Search 采集器"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        search_cfg = cfg.get("search", {})
        self.api_key = search_cfg.get("brave_api_key")
        self.base_url = search_cfg.get("brave_base_url", "https://api.search.brave.com/res/v1/web/search")
        
        if not self.api_key:
            raise ValueError("Brave API Key 未配置")
        
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "X-Subscription-Token": self.api_key,
                    "Accept": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session
    
    async def search(
        self,
        query: str,
        count: int = 10,
        offset: int = 0,
        search_lang: str = "zh-hans",
        freshness: Optional[str] = None,  # pd (past day), pw (past week), pm (past month)
    ) -> List[BraveSearchResult]:
        """
        执行 Brave Search
        
        Args:
            query: 搜索关键词
            count: 返回结果数量 (1-20)
            offset: 分页偏移
            search_lang: 搜索语言
            freshness: 时间过滤
        """
        session = await self._get_session()
        
        params = {
            "q": query,
            "count": min(count, 20),
            "offset": offset,
            "search_lang": search_lang,
        }
        if freshness:
            params["freshness"] = freshness
        
        try:
            async with session.get(self.base_url, params=params) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"Brave Search 错误 {resp.status}: {text}")
                    return []
                
                data = await resp.json()
                results = []
                
                for item in data.get("web", {}).get("results", []):
                    results.append(BraveSearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        description=item.get("description", ""),
                        age=item.get("age"),
                        source=item.get("meta", {}).get("url", ""),
                    ))
                
                logger.info(f"Brave Search '{query}' 返回 {len(results)} 条结果")
                return results
        
        except Exception as e:
            logger.error(f"Brave Search 失败: {e}")
            return []
    
    async def search_policy_news(
        self,
        ministry: str,
        commodity_keywords: List[str],
        days: int = 1,
    ) -> List[BraveSearchResult]:
        """
        搜索特定部委的政策新闻
        """
        freshness = "pd" if days <= 1 else "pw" if days <= 7 else "pm"
        keywords_str = " ".join(commodity_keywords[:3])
        # 使用多种查询策略以提高召回率
        queries = [
            f"{ministry} {keywords_str} 政策",
            f"{ministry} {keywords_str} 通知",
        ]
        
        all_results = []
        seen_urls = set()
        for query in queries:
            results = await self.search(query, count=5, freshness=freshness)
            for r in results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    all_results.append(r)
        
        return all_results
    
    async def search_fundamental(
        self,
        commodity: str,
        indicator: str,
        days: int = 7,
    ) -> List[BraveSearchResult]:
        """
        搜索基本面数据/分析
        """
        freshness = "pw" if days <= 7 else "pm"
        query = f"{commodity} {indicator} 最新数据 分析"
        
        return await self.search(query, count=10, freshness=freshness)
    
    async def search_industry_news(
        self,
        industry: str,
        days: int = 3,
    ) -> List[BraveSearchResult]:
        """
        搜索行业动态
        """
        freshness = "pd" if days <= 1 else "pw" if days <= 3 else "pm"
        query = f"{industry} 行业动态 最新消息"
        
        return await self.search(query, count=10, freshness=freshness)
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
