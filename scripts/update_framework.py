"""
投研框架迭代更新脚本
- 根据新的研究发现或交易复盘，更新 docs/ 中的认知框架
- 支持交互式更新和批量更新
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.llm_integration import LLMClient

DOCS_DIR = PROJECT_ROOT / "docs"
FRAMEWORK_FILE = DOCS_DIR / "research_framework.md"
POLICY_SOURCES_FILE = DOCS_DIR / "policy_sources.md"


def append_to_file(filepath: Path, content: str):
    """追加内容到文件"""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write("\n" + content + "\n")


def update_framework_with_research(new_insight: str, category: str = "general"):
    """
    将新的研究发现追加到框架文档
    """
    timestamp = datetime.now().strftime("%Y-%m-%d")
    
    entry = f"""
---

## 认知更新记录 [{timestamp}] — {category}

{new_insight}

*更新时间: {timestamp}*
"""
    
    append_to_file(FRAMEWORK_FILE, entry)
    print(f"已更新框架文档: {FRAMEWORK_FILE}")


def add_trade_case(
    commodity: str,
    direction: str,
    entry_price: float,
    exit_price: float,
    pnl: float,
    logic: str,
    review: str,
):
    """
    添加交易案例复盘
    """
    timestamp = datetime.now().strftime("%Y%m%d")
    case_file = DOCS_DIR / "trade_cases" / f"{commodity}_{direction}_{timestamp}.md"
    
    content = f"""# 交易复盘 — {commodity} {direction}

- **日期**: {datetime.now().strftime("%Y-%m-%d")}
- **品种**: {commodity}
- **方向**: {direction}
- **入场价**: {entry_price}
- **出场价**: {exit_price}
- **盈亏**: {pnl}

## 交易逻辑

{logic}

## 复盘

{review}

## 认知更新

- 
"""
    
    with open(case_file, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"交易复盘已保存: {case_file}")
    
    # 同时更新框架文档
    update_framework_with_research(
        new_insight=f"交易案例 ({commodity} {direction}): {review}",
        category="trade_review",
    )


async def ai_update_framework(topic: str, new_evidence: str):
    """
    使用 LLM 辅助更新框架
    """
    client = LLMClient()
    
    # 读取现有框架
    with open(FRAMEWORK_FILE, "r", encoding="utf-8") as f:
        current_framework = f.read()
    
    prompt = f"""你是一位资深的投研框架维护者。当前框架如下：

{current_framework[:3000]}...

【新的证据/发现】
主题: {topic}
内容: {new_evidence}

请输出：
1. 现有框架中需要修正或补充的部分
2. 建议添加的新认知
3. 更新后的相关章节内容（markdown 格式）

只输出需要修改的部分，不需要重复整个框架。"""
    
    response = await client.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    
    print("\n" + "=" * 60)
    print("LLM 建议的框架更新")
    print("=" * 60)
    print(response.content)
    
    # 保存建议
    suggestion_file = DOCS_DIR / f"framework_suggestion_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    with open(suggestion_file, "w", encoding="utf-8") as f:
        f.write(response.content)
    
    print(f"\n建议已保存: {suggestion_file}")
    await client.close()


def main():
    parser = argparse.ArgumentParser(description="更新投研框架")
    parser.add_argument("--insight", help="新的研究发现文本")
    parser.add_argument("--category", default="general", help="发现类别")
    parser.add_argument("--trade", action="store_true", help="添加交易复盘")
    parser.add_argument("--commodity", help="交易品种")
    parser.add_argument("--direction", help="交易方向")
    parser.add_argument("--entry", type=float, help="入场价")
    parser.add_argument("--exit", type=float, help="出场价")
    parser.add_argument("--pnl", type=float, help="盈亏")
    parser.add_argument("--logic", help="交易逻辑")
    parser.add_argument("--review", help="复盘内容")
    parser.add_argument("--ai-update", action="store_true", help="使用 AI 辅助更新")
    parser.add_argument("--topic", help="AI 更新主题")
    parser.add_argument("--evidence", help="AI 更新证据")
    
    args = parser.parse_args()
    
    if args.trade:
        if not all([args.commodity, args.direction, args.entry is not None, args.exit is not None, args.pnl is not None]):
            print("交易复盘需要 --commodity --direction --entry --exit --pnl")
            return
        add_trade_case(
            args.commodity, args.direction,
            args.entry, args.exit, args.pnl,
            args.logic or "", args.review or "",
        )
    elif args.ai_update:
        if not args.topic or not args.evidence:
            print("AI 更新需要 --topic 和 --evidence")
            return
        import asyncio
        asyncio.run(ai_update_framework(args.topic, args.evidence))
    elif args.insight:
        update_framework_with_research(args.insight, args.category)
    else:
        print("请提供 --insight, --trade, 或 --ai-update")


if __name__ == "__main__":
    main()
