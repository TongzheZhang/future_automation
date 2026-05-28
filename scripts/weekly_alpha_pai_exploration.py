"""
每周 Alpha派 策略探索脚本
- 基于当周交易记录和市场热点，进行跨品种深度策略扫描
- 输出下周重点关注清单、套利机会、风险预警
- 保存到 docs/commodity_chains/weekly_outlook.md

调度：每周日晚 20:00 自动运行（见 scheduler/cron.py）
也可手动运行：python scripts/weekly_alpha_pai_exploration.py
"""

import os
import sys
import asyncio
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.collectors.alpha_pai_research import AlphaPaiResearchAdvisor
from data.collectors.brave_search import BraveSearchCollector
from intraday.record import RECORD_DIR, load_trades
from intraday.evolution import load_cognition_library

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            PROJECT_ROOT / "logs" / f"weekly_exploration_{datetime.now().strftime('%Y%m%d')}.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("weekly_exploration")


def get_week_date_range(date: datetime = None) -> tuple:
    """获取指定日期所在周的起止日期（周一到周日）"""
    if date is None:
        date = datetime.now()
    # 找到本周一
    monday = date - timedelta(days=date.weekday())
    # 本周日
    sunday = monday + timedelta(days=6)
    return monday.strftime("%Y-%m-%d"), sunday.strftime("%Y-%m-%d")


def load_week_trades(monday: str, sunday: str) -> List[Dict[str, Any]]:
    """加载本周所有交易日的交易记录"""
    trades = []
    current = datetime.strptime(monday, "%Y-%m-%d")
    end = datetime.strptime(sunday, "%Y-%m-%d")
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        day_trades = load_trades(date_str)
        for t in day_trades:
            trades.append({
                "date": date_str,
                "commodity": t.commodity,
                "direction": t.direction.value,
                "pnl": t.pnl,
                "status": t.status.value,
                "core_logic": t.core_logic,
            })
        current += timedelta(days=1)
    return trades


def build_trades_summary(trades: List[Dict[str, Any]]) -> str:
    """构建本周交易摘要"""
    if not trades:
        return "本周无交易。"
    
    win_count = sum(1 for t in trades if t["status"] == "WIN")
    loss_count = sum(1 for t in trades if t["status"] == "LOSS")
    total_pnl = sum(t["pnl"] for t in trades)
    
    lines = [
        f"本周交易 {len(trades)} 笔，胜率 {win_count}/{len(trades)}，总盈亏 {total_pnl:+.2f}",
        "",
        "明细:",
    ]
    for t in trades:
        emoji = "🟢" if t["status"] == "WIN" else "🔴" if t["status"] == "LOSS" else "⚪"
        lines.append(
            f"{emoji} {t['date']} {t['commodity']} {t['direction']} "
            f"盈亏:{t['pnl']:+.2f} | 逻辑:{t['core_logic'][:40]}"
        )
    return "\n".join(lines)


async def run_weekly_exploration(target_date: str = None):
    """执行每周策略探索"""
    if target_date:
        date = datetime.strptime(target_date, "%Y-%m-%d")
    else:
        date = datetime.now()
    
    monday, sunday = get_week_date_range(date)
    logger.info("=" * 60)
    logger.info(f"每周 Alpha派 策略探索: {monday} ~ {sunday}")
    logger.info("=" * 60)
    
    # 1. 加载本周交易记录
    trades = load_week_trades(monday, sunday)
    trades_summary = build_trades_summary(trades)
    logger.info(f"加载本周交易记录: {len(trades)} 笔")
    
    # 2. 搜索本周市场热点
    brave = BraveSearchCollector()
    hot_topics = ""
    try:
        logger.info("搜索本周市场热点...")
        results = await brave.search(
            query="本周期货市场 热点 政策 重大事件 品种涨跌",
            count=8,
            freshness="pw",
        )
        hot_topics = "\n".join([f"- {r.title}: {r.description}" for r in results[:5]])
        logger.info(f"搜索到 {len(results)} 条热点")
    except Exception as e:
        logger.error(f"市场热点搜索失败: {e}")
    
    # 3. 加载认知库摘要
    try:
        library = load_cognition_library()
        cog_summary = f"认知库共 {len(library.items)} 条，已验证高可靠度 {len(library.get_validated_items(require_generality=True))} 条"
    except Exception:
        cog_summary = "认知库加载失败"
    
    # 4. 调用 Alpha派 每周策略探索
    advisor = AlphaPaiResearchAdvisor()
    try:
        logger.info("调用 Alpha派 每周策略探索...")
        exploration = await advisor.weekly_exploration(
            week_trades=trades_summary,
            hot_topics=hot_topics,
            date=sunday,
        )
        logger.info("Alpha派 策略探索完成")
    except Exception as e:
        logger.error(f"Alpha派 策略探索失败: {e}")
        exploration = "策略探索生成失败。"
    
    # 5. 生成报告
    report_lines = [
        f"# 每周策略探索 ({monday} ~ {sunday})",
        "",
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 本周交易: {len(trades)} 笔 | {cog_summary}",
        "",
        "---",
        "",
        exploration,
        "",
        "---",
        "",
        "## 本周热点摘要",
        "",
        hot_topics or "无",
        "",
        "## 本周交易摘要",
        "",
        trades_summary,
        "",
    ]
    
    report_content = "\n".join(report_lines)
    
    # 6. 保存报告
    report_path = PROJECT_ROOT / "docs" / "commodity_chains" / "weekly_outlook.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    logger.info(f"报告已保存: {report_path}")
    
    # 打印到控制台
    print("\n" + "=" * 60)
    print(report_content)
    print("=" * 60)
    
    await brave.close()
    return report_content


async def main():
    parser = argparse.ArgumentParser(description="每周 Alpha派 策略探索")
    parser.add_argument(
        "--date",
        help="指定周日日期 YYYY-MM-DD，默认本周",
    )
    args = parser.parse_args()
    
    await run_weekly_exploration(target_date=args.date)


if __name__ == "__main__":
    asyncio.run(main())
