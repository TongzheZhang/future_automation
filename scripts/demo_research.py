"""
演示脚本 — 展示系统核心工作流程
- 使用 Brave Search 获取政策和基本面信息
- 生成基础日报（不依赖 OpenRouter/Alpha 派）
- 用于验证系统架构和数据流

用法:
    python scripts/demo_research.py
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.collectors.brave_search import BraveSearchCollector
from data.collectors.policy_news import PolicyNewsCollector
from research.chain_mapper import ChainMapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("demo")


async def demo_brave_search():
    """演示 Brave Search 能力"""
    print("\n" + "=" * 60)
    print("[Demo 1] Brave Search — 政策新闻搜索")
    print("=" * 60)
    
    async with BraveSearchCollector() as bs:
        # 搜索发改委政策
        results = await bs.search(
            query="site:gov.cn 发改委 钢铁 政策",
            count=5,
            freshness="pw",
        )
        print(f"\n找到 {len(results)} 条发改委政策相关结果:")
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r.title}")
            print(f"   {r.description[:120]}...")
            print(f"   URL: {r.url}")
        
        # 搜索基本面
        print("\n" + "-" * 60)
        results2 = await bs.search(
            query="豆粕 库存 供需 最新分析",
            count=5,
            freshness="pw",
        )
        print(f"\n找到 {len(results2)} 条豆粕基本面分析:")
        for i, r in enumerate(results2, 1):
            print(f"\n{i}. {r.title}")
            print(f"   {r.description[:120]}...")


async def demo_policy_collection():
    """演示政策新闻采集"""
    print("\n" + "=" * 60)
    print("[Demo 2] 政策新闻采集器")
    print("=" * 60)
    
    collector = PolicyNewsCollector()
    try:
        news = await collector.fetch_all(days_back=3)
        print(f"\n共采集到 {len(news)} 条政策新闻")
        
        for item in news[:5]:
            print(f"\n• [{item.ministry}] {item.title}")
            print(f"  关键词: {', '.join(item.matched_keywords)}")
            print(f"  摘要: {item.summary[:100]}...")
    finally:
        await collector.close()


async def demo_chain_analysis():
    """演示产业链分析"""
    print("\n" + "=" * 60)
    print("[Demo 3] 产业链映射")
    print("=" * 60)
    
    mapper = ChainMapper()
    
    commodities = ["RB", "I", "CU", "M", "SC"]
    for code in commodities:
        chain_id = mapper.get_chain_by_commodity(code)
        if chain_id:
            related = mapper.get_related_commodities(code)
            print(f"\n{code} → 产业链: {chain_id}")
            print(f"  相关品种: {', '.join(related) if related else '无'}")


async def generate_demo_report():
    """生成演示日报"""
    print("\n" + "=" * 60)
    print("[Demo 4] 生成演示日报")
    print("=" * 60)
    
    async with BraveSearchCollector() as bs:
        # 收集多品种信息
        commodities = [
            ("RB", "螺纹钢", ["螺纹钢", "钢铁", "地产政策"]),
            ("CU", "铜", ["铜", "铜矿", "新能源"]),
            ("M", "豆粕", ["豆粕", "大豆", "USDA"]),
        ]
        
        report_lines = [
            f"# 期货投研日报 (Demo) — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "## 政策动态",
            "",
        ]
        
        for code, name, keywords in commodities:
            query = f"{' '.join(keywords[:2])} 政策 最新"
            results = await bs.search(query, count=3, freshness="pw")
            
            report_lines.append(f"### {name} ({code})")
            if results:
                for r in results[:2]:
                    report_lines.append(f"- {r.title}: {r.description[:80]}...")
            else:
                report_lines.append("- 暂无重大政策动态")
            report_lines.append("")
        
        report_lines.extend([
            "## 基本面要点",
            "",
        ])
        
        for code, name, keywords in commodities:
            query = f"{' '.join(keywords[:2])} 库存 供需 分析"
            results = await bs.search(query, count=3, freshness="pw")
            
            report_lines.append(f"### {name} ({code})")
            if results:
                for r in results[:2]:
                    report_lines.append(f"- {r.description[:100]}...")
            else:
                report_lines.append("- 暂无基本面更新")
            report_lines.append("")
        
        report_lines.append("---")
        report_lines.append("*本报告为演示版本，仅使用 Brave Search 数据*")
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        report_dir = PROJECT_ROOT / "reports" / "daily"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"demo_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        print(f"\n演示日报已生成: {report_path}")
        print("\n" + "-" * 60)
        print(report_content[:1500])
        print("...")
        print("-" * 60)


async def main():
    print("\n" + "#" * 60)
    print("# 政策-基本面期货投研系统 — 功能演示")
    print("#" * 60)
    
    await demo_brave_search()
    await demo_policy_collection()
    await demo_chain_analysis()
    await generate_demo_report()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("""
说明：
- 本演示使用 Brave Search API 获取公开信息
- 完整功能需要配置有效的 OpenRouter API Key（用于 LLM 分析）
- 完整功能需要配置 Alpha 派 API Key（用于深度投研数据）
- 运行完整投研: python scripts/daily_research.py --focus RB CU M
""")


if __name__ == "__main__":
    asyncio.run(main())
