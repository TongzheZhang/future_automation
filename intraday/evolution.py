"""
策略自我进化引擎
- 从复盘报告中提取 lessons
- 融合到认知库
- 动态修改 strategy prompt
- 更新框架文档
- 追踪改进效果
"""

import os
import sys
import json
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.llm_integration import LLMClient, extract_json_from_text
from data.collectors.alpha_pai import AlphaPaiCollector
from intraday.models import CognitionItem, CognitionLibrary, DailyReview, TradeStatus
from intraday.record import load_trades, RECORD_DIR

logger = logging.getLogger("evolution")

COGNITION_FILE = RECORD_DIR / "cognition_library.json"
FRAMEWORK_PATH = Path(__file__).parent.parent / "docs" / "research_framework.md"


def load_cognition_library() -> CognitionLibrary:
    """加载认知库"""
    if COGNITION_FILE.exists():
        with open(COGNITION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CognitionLibrary(**data)
    return CognitionLibrary()


def save_cognition_library(library: CognitionLibrary):
    """保存认知库"""
    library.updated_at = datetime.now()
    with open(COGNITION_FILE, "w", encoding="utf-8") as f:
        json.dump(library.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
    logger.info(f"认知库已保存: {COGNITION_FILE} | 共 {len(library.items)} 条认知")


async def extract_cognitions_from_review(review: DailyReview) -> List[CognitionItem]:
    """
    从复盘报告中提取结构化的 lessons
    """
    if not review.trades:
        return []
    
    # 构建复盘摘要
    trade_summary = []
    for t in review.trades:
        trade_summary.append(
            f"品种:{t.commodity} 方向:{t.direction.value} 盈亏:{t.pnl} "
            f"入场:{t.actual_entry} 平仓:{t.actual_exit} 最高:{t.day_high} 最低:{t.day_low} "
            f"逻辑:{t.core_logic}"
        )
    
    prompt = f"""你是一位期货策略进化专家。请从以下复盘内容中提取 3-5 条结构化的经验教训（lessons）。

【当日交易记录】
{chr(10).join(trade_summary)}

【复盘分析】
{review.review_summary[:2000]}

请输出 JSON 数组格式：
[
    {{
        "lesson": "核心教训（简洁、可执行，50字以内）",
        "category": "类别: signal_filter / entry_timing / exit_timing / risk_control / market_regime / general",
        "confidence": 1-10 的整数（基于当日表现判断该教训的可靠程度）,
        "affected_commodities": ["适用品种代码"]
    }}
]

提取原则：
- 只提取具体的、可操作的教训，不要泛泛而谈
- confidence 应基于当日表现的确定性：盈亏大且逻辑清晰 → 高 confidence；盈亏小或逻辑模糊 → 低 confidence
- 若当日无交易或无明显教训，返回空数组 []
- 避免提取与已有常识重复的内容（如"要严格止损"这种过于泛化的内容）
"""
    
    client = LLMClient()
    try:
        resp = await client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1500,
        )
        data = extract_json_from_text(resp.content)
        
        if not isinstance(data, list):
            logger.warning(f"提取的认知不是数组: {type(data)}")
            return []
        
        cognitions = []
        for item in data:
            if not isinstance(item, dict):
                continue
            lesson = item.get("lesson", "").strip()
            if not lesson or len(lesson) < 5:
                continue
            
            cog = CognitionItem(
                id=f"COG-{datetime.now().strftime('%Y%m%d')}-{len(cognitions)+1:02d}",
                lesson=lesson,
                category=item.get("category", "general"),
                confidence=min(10, max(0, int(item.get("confidence", 5)))),
                affected_commodities=item.get("affected_commodities", []),
                source_trade_date=review.date,
            )
            # 根据当日表现初始化验证计数
            for t in review.trades:
                if t.status == TradeStatus.WIN:
                    cog.win_count += 1
                    cog.verification_count += 1
                elif t.status == TradeStatus.LOSS:
                    cog.loss_count += 1
                    cog.verification_count += 1
            
            cognitions.append(cog)
        
        logger.info(f"从复盘提取了 {len(cognitions)} 条认知")
        return cognitions
    
    except Exception as e:
        logger.error(f"认知提取失败: {e}")
        return []
    finally:
        await client.close()


def merge_cognitions(library: CognitionLibrary, new_cogs: List[CognitionItem]) -> CognitionLibrary:
    """
    将新认知融合到认知库
    - 相似认知合并（更新可靠度、跨品种计数、跨日期计数）
    - 全新认知追加
    - 矛盾认知标记
    """
    for new_cog in new_cogs:
        # 查找相似认知（基于 lesson 文本相似度）
        merged = False
        for existing in library.items:
            if new_cog.lesson[:30] == existing.lesson[:30] or new_cog.lesson in existing.lesson or existing.lesson in new_cog.lesson:
                # 合并：取更高可靠度，累加验证次数
                existing.confidence = max(existing.confidence, new_cog.confidence)
                existing.verification_count += new_cog.verification_count
                existing.win_count += new_cog.win_count
                existing.loss_count += new_cog.loss_count
                existing.source_trade_date = new_cog.source_trade_date
                
                # 更新跨品种/跨日期计数（防过拟合关键）
                existing.cross_commodity_count = max(
                    existing.cross_commodity_count,
                    len(set(existing.affected_commodities + new_cog.affected_commodities))
                )
                existing.cross_date_count += 1
                existing.uniqueness_score = min(
                    existing.uniqueness_score,
                    new_cog.uniqueness_score
                )
                
                # 合并 affected_commodities
                existing.affected_commodities = list(set(
                    existing.affected_commodities + new_cog.affected_commodities
                ))
                
                merged = True
                logger.info(f"认知合并: {existing.lesson[:40]}... (验证次数+{new_cog.verification_count}, 跨品种:{existing.cross_commodity_count}, 跨日期:{existing.cross_date_count})")
                break
        
        if not merged:
            library.items.append(new_cog)
            logger.info(f"新增认知: {new_cog.lesson[:40]}... (来源:{new_cog.source_type})")
    
    # 按可靠度排序
    library.items.sort(key=lambda x: (x.confidence, x.verification_count), reverse=True)
    
    # 重新构建 prompt 追加文本（只使用具备普适性的认知）
    library.evolved_prompt_additions = library.rebuild_prompt_additions()
    
    return library


def get_evolved_system_prompt(base_prompt: str, prompt_additions: str = "") -> str:
    """
    将认知库中的经验规则拼接到 base system_prompt
    """
    if not prompt_additions:
        return base_prompt
    
    return base_prompt + prompt_additions


def get_dynamic_confidence_threshold(library: CognitionLibrary, default: int = 7) -> int:
    """
    根据近期历史表现动态调整置信度阈值
    - 若近期胜率持续>60%，可降低阈值（更多交易机会）
    - 若近期胜率持续<40%，可提高阈值（减少错误交易）
    """
    # 获取最近10次交易的认知验证数据
    recent_items = [item for item in library.items if item.verification_count > 0]
    if not recent_items:
        return default
    
    total_wins = sum(item.win_count for item in recent_items)
    total_losses = sum(item.loss_count for item in recent_items)
    total = total_wins + total_losses
    
    if total < 5:
        return default
    
    win_rate = total_wins / total
    if win_rate >= 0.6:
        return max(5, default - 1)  # 降低阈值
    elif win_rate <= 0.4:
        return min(9, default + 1)  # 提高阈值
    
    return default


async def update_framework_with_cognitions(library: CognitionLibrary, framework_path: Path = None):
    """
    将认知库更新到 docs/research_framework.md
    """
    if framework_path is None:
        framework_path = FRAMEWORK_PATH
    if not framework_path.exists():
        logger.warning("框架文档不存在，跳过更新")
        return
    
    with open(framework_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 构建认知积累章节
    sections = []
    categories = {
        "signal_filter": "信号筛选",
        "entry_timing": "入场时机",
        "exit_timing": "出场时机",
        "risk_control": "风险控制",
        "market_regime": "市场状态判断",
        "general": "通用原则",
    }
    
    for cat_key, cat_name in categories.items():
        items = library.get_items_by_category(cat_key)
        if not items:
            continue
        sections.append(f"### {cat_name}")
        for item in items:
            status = {"pending": "⏳待验证", "validated": "✅已验证", "invalidated": "❌已证伪"}.get(item.status, "⏳")
            sections.append(f"- {status} **{item.lesson}** (可靠度:{item.confidence}/10,验证:{item.verification_count}次)")
        sections.append("")
    
    if not sections:
        return
    
    cognition_section = "\n".join(sections)
    new_section = f"""## 八、日内交易认知积累（自动更新）

> 本章节由系统根据每日复盘自动更新，记录从实战中提炼的经验教训。
> 上次更新: {datetime.now().strftime('%Y-%m-%d %H:%M')}

{cognition_section}
"""
    
    # 替换或追加章节
    marker = "## 八、日内交易认知积累"
    if marker in content:
        # 替换现有章节
        import re
        pattern = re.compile(rf"{re.escape(marker)}.*?(?=\n## |\Z)", re.DOTALL)
        content = pattern.sub(new_section, content)
    else:
        # 追加到文件末尾
        content = content.rstrip() + "\n\n" + new_section
    
    with open(framework_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    logger.info(f"框架文档已更新: {framework_path}")


async def extract_cognitions_from_market_scan(market_scan) -> List[CognitionItem]:
    """
    从市场扫描报告中提取普适性 lessons
    """
    if not market_scan or not market_scan.extracted_lessons:
        return []
    
    cognitions = []
    for i, lesson in enumerate(market_scan.extracted_lessons):
        cog = CognitionItem(
            id=f"MKT-{datetime.now().strftime('%Y%m%d')}-{len(cognitions)+1:02d}",
            lesson=lesson.get("lesson", ""),
            category=lesson.get("category", "general"),
            confidence=min(10, max(0, int(lesson.get("universality_score", 5)))),
            source_type="market_observation",
            cross_commodity_count=len(lesson.get("commodities", [])),
            cross_date_count=1,
            uniqueness_score=10 - min(10, max(0, int(lesson.get("universality_score", 5)))),
            affected_commodities=lesson.get("commodities", []),
            source_trade_date=market_scan.date,
        )
        cognitions.append(cog)
    
    logger.info(f"从市场扫描提取了 {len(cognitions)} 条认知")
    return cognitions


async def run_evolution(review: DailyReview, market_scan=None) -> str:
    """
    执行完整的进化流程（支持从自己交易 + 市场观察中提取）
    返回进化报告文本
    """
    logger.info("=" * 60)
    logger.info("策略自我进化开始")
    logger.info("=" * 60)
    
    # 1. 加载认知库
    library = load_cognition_library()
    
    # 2. 从自己交易中提取认知
    trade_cogs = await extract_cognitions_from_review(review)
    review.extracted_cognitions = trade_cogs
    
    # 3. 从市场扫描中提取认知
    market_cogs = await extract_cognitions_from_market_scan(market_scan)
    
    # 4. 融合所有认知
    all_new_cogs = trade_cogs + market_cogs
    if all_new_cogs:
        library = merge_cognitions(library, all_new_cogs)
        save_cognition_library(library)
    
    # 5. 更新框架文档
    await update_framework_with_cognitions(library)
    
    # 6. 生成进化报告
    report_lines = [
        f"# 策略进化报告 ({review.date})",
        "",
        f"- 认知库总条目: {len(library.items)}",
        f"- 本次新增/更新: {len(all_new_cogs)}",
        f"  - 来自自己交易: {len(trade_cogs)}",
        f"  - 来自市场观察: {len(market_cogs)}",
        f"- 已验证高可靠度条目 (≥7, 含普适性): {len(library.get_validated_items(require_generality=True))}",
        f"- 动态置信度阈值: {get_dynamic_confidence_threshold(library)}",
        "",
    ]
    
    if trade_cogs:
        report_lines.append("## 本次从交易提取的认知")
        report_lines.append("")
        for cog in trade_cogs:
            report_lines.append(f"- **[{cog.category}]** {cog.lesson} (可靠度:{cog.confidence})")
        report_lines.append("")
    
    if market_cogs:
        report_lines.append("## 本次从市场观察提取的认知")
        report_lines.append("")
        for cog in market_cogs:
            report_lines.append(f"- **[{cog.category}]** {cog.lesson} (普适性:{cog.uniqueness_score}/10, 品种:{cog.cross_commodity_count})")
        report_lines.append("")
    
    if library.evolved_prompt_additions:
        report_lines.append("## 已融入策略 Prompt 的经验规则")
        report_lines.append("")
        report_lines.append("```")
        report_lines.append(library.evolved_prompt_additions)
        report_lines.append("```")
        report_lines.append("")
    
    report_lines.append("---")
    report_lines.append(f"*进化时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    report_content = "\n".join(report_lines)
    review.evolution_report = report_content
    
    # 保存报告
    report_dir = Path(__file__).parent.parent / "reports" / "intraday"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"evolution_{review.date}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    logger.info(f"进化报告已保存: {report_path}")
    
    print("\n" + "=" * 60)
    print(report_content)
    print("=" * 60)
    
    return report_content


def get_cognition_prompt_additions() -> str:
    """
    供 strategy.py 调用，获取当前应追加到 prompt 的经验规则
    """
    library = load_cognition_library()
    return library.evolved_prompt_additions


def get_current_confidence_threshold(default: int = 7) -> int:
    """
    供 strategy.py 调用，获取当前动态调整的置信度阈值
    """
    library = load_cognition_library()
    return get_dynamic_confidence_threshold(library, default)
