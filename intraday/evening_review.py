"""
19:00 复盘脚本
- 读取当日信号和交易记录
- 用 LLM 分析判断偏差
- 生成复盘报告
- 提取策略改进点
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.llm_integration import LLMClient
from intraday.record import load_signals, load_trades, save_review
from intraday.models import DailyReview, TradeStatus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("evening_review")


async def run_evening_review(date: str = None):
    """执行当日复盘"""
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info("=" * 60)
    logger.info(f"[{date}] 19:00 复盘开始")
    logger.info("=" * 60)
    
    signals = load_signals(date)
    trades = load_trades(date)
    
    review = DailyReview(date=date, signals=signals, trades=trades)
    review.compute_stats()
    
    # LLM 深度复盘
    if trades:
        review.review_summary = await _llm_review(trades, signals)
    else:
        review.review_summary = "今日无交易，无复盘内容。"
    
    # 保存复盘
    save_review(review)
    
    # 生成报告
    report_lines = [
        f"# 日内交易复盘 ({date})",
        "",
        "## 统计",
        f"- 扫描品种: {review.total_signals}",
        f"- 实际交易: {review.trade_count}",
        f"- 胜率: {review.accuracy}%",
        f"- 总盈亏: {review.total_pnl:+.2f}",
        f"- 平均盈亏: {review.avg_pnl:+.2f}",
        "",
        "## 交易明细 (时间窗口: 09:05 → 14:55, 数据源: AKShare 1分钟线)",
        "",
    ]
    
    for t in trades:
        emoji = "🟢" if t.status == TradeStatus.WIN else "🔴" if t.status == TradeStatus.LOSS else "⚪"
        report_lines.append(f"{emoji} **{t.commodity}** {t.direction.value}")
        report_lines.append(f"  信号: 入场{t.signal_entry} 止损{t.signal_stop} 目标{t.signal_target}")
        report_lines.append(f"  实际: 09:05 入场{t.actual_entry} → 14:55 平仓{t.actual_exit}")
        report_lines.append(f"  盈亏: {t.pnl:+} | 回撤: {t.max_drawdown}")
        report_lines.append(f"  逻辑: {t.core_logic}")
        if t.review_notes:
            report_lines.append(f"  复盘: {t.review_notes}")
        report_lines.append("")
    
    if review.review_summary:
        report_lines.append("## LLM 深度复盘")
        report_lines.append("")
        report_lines.append(review.review_summary)
        report_lines.append("")
    
    if review.lessons:
        report_lines.append("## 改进点")
        report_lines.append("")
        for lesson in review.lessons:
            report_lines.append(f"- {lesson}")
        report_lines.append("")
    
    report_lines.append("---")
    report_lines.append(f"*复盘时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    report_content = "\n".join(report_lines)
    
    report_dir = Path(__file__).parent.parent / "reports" / "intraday"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"review_{date}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    logger.info(f"复盘报告已保存: {report_path}")
    
    print("\n" + "=" * 60)
    print(report_content)
    print("=" * 60)
    
    return review


async def _llm_review(trades, signals) -> str:
    """用 LLM 生成深度复盘"""
    
    # 构建复盘数据
    trade_descriptions = []
    for t in trades:
        trade_descriptions.append(
            f"品种: {t.commodity}, 方向: {t.direction.value}, "
            f"信号入场: {t.signal_entry}, 实际平仓: {t.actual_exit}, "
            f"日内最高: {t.day_high}, 日内最低: {t.day_low}, "
            f"盈亏: {t.pnl}, 状态: {t.status.value}, "
            f"逻辑: {t.core_logic}"
        )
    
    prompt = f"""你是一位期货交易复盘专家。请对以下日内交易进行复盘分析。

【重要说明】
- 本策略只做日盘交易，时间窗口严格为 09:05 开仓 → 14:55 平仓，不过夜、不碰夜盘。
- 以下 "日内最高/最低" 数据来自 AKShare 1分钟线，已过滤为纯日盘(09:05-14:55)数据。

【当日交易记录】
{chr(10).join(trade_descriptions)}

请分析：
1. 本次判断的主要偏差在哪里？（09:05开盘判断 vs 09:05→14:55实际走势）
2. 止损和目标设置是否合理？
3. 有没有更好的入场或出场时机？（但必须在14:55前平仓）
4. 下次类似情况应如何改进？
5. 用简洁的语言总结今日得失。

输出要求：
- 客观、具体，不要泛泛而谈
- 指出具体的判断失误或成功之处
- 给出可操作的改进建议
- 请严格基于 09:05→14:55 的日盘时间窗口进行分析
"""
    
    client = LLMClient()
    try:
        resp = await client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )
        return resp.content
    except Exception as e:
        logger.error(f"LLM 复盘失败: {e}")
        return "复盘分析生成失败。"
    finally:
        await client.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="复盘日期 YYYY-MM-DD，默认今天")
    args = parser.parse_args()
    
    asyncio.run(run_evening_review(args.date))
