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
from data.collectors.alpha_pai import AlphaPaiCollector
from data.collectors.alpha_pai_research import AlphaPaiResearchAdvisor
from intraday.record import load_signals, load_trades, save_review
from intraday.models import DailyReview, TradeStatus
from intraday.evolution import run_evolution
from intraday.market_scan import run_market_scan, MarketScanResult

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
    
    # 【新增】市场行情扫描（全市场逆向学习）
    market_scan_result: MarketScanResult = None
    try:
        market_scan_result = await run_market_scan(date=date)
        logger.info("市场行情扫描完成")
    except Exception as e:
        logger.error(f"市场行情扫描失败: {e}")
    
    # Alpha派 Missed Opportunities 分析（无论有无交易都执行）
    missed_opportunities = ""
    if market_scan_result and market_scan_result.performances:
        try:
            advisor = AlphaPaiResearchAdvisor()
            scan_results = []
            for p in market_scan_result.performances:
                scan_results.append({
                    "code": p.commodity,
                    "name": p.name,
                    "open": p.open_price,
                    "close": p.close_price,
                    "change_pct": p.change_pct,
                    "amplitude_pct": p.amplitude_pct,
                    "direction_reason": "系统观望（无高置信度信号）",
                })
            traded_codes = {t.commodity for t in trades}
            missed_opportunities = await advisor.analyze_missed_opportunities(
                scan_results=scan_results,
                traded_codes=traded_codes,
            )
            if missed_opportunities:
                logger.info("Alpha派 Missed Opportunities 分析完成")
        except Exception as e:
            logger.error(f"Alpha派 Missed Opportunities 分析失败: {e}")
    
    # LLM 深度复盘
    if trades:
        review.review_summary = await _llm_review(trades, signals)
        # Alpha 派专家讨论（作为投研顾问，讨论优化点）
        alpha_pai_discussion = await _alpha_pai_expert_review(trades)
        if alpha_pai_discussion:
            review.review_summary += f"\n\n---\n\n【Alpha 派投研顾问讨论】\n{alpha_pai_discussion}"
    else:
        review.review_summary = "今日无交易，无复盘内容。"
    
    # 追加 Missed Opportunities 分析
    if missed_opportunities:
        review.review_summary += f"\n\n---\n\n【Alpha派 Missed Opportunities 分析】\n{missed_opportunities}"
    
    # 保存复盘
    save_review(review)
    
    # 策略自我进化（从自己交易 + 市场观察中提取）
    try:
        await run_evolution(review, market_scan_result)
    except Exception as e:
        logger.error(f"策略进化失败: {e}")
    
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
    
    if market_scan_result:
        report_lines.append("## 市场行情扫描")
        report_lines.append("")
        report_lines.append(f"[市场扫描报告](market_scan_{date}.md)")
        report_lines.append("")
        if market_scan_result.extracted_lessons:
            report_lines.append("**全市场提取的普适性规律：**")
            for lesson in market_scan_result.extracted_lessons:
                report_lines.append(f"- [{lesson.get('category', 'general')}] {lesson.get('lesson', '')} (普适性:{lesson.get('universality_score', 0)}/10)")
            report_lines.append("")
        if missed_opportunities:
            report_lines.append("## Alpha派 Missed Opportunities 分析")
            report_lines.append("")
            report_lines.append(missed_opportunities)
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


async def _alpha_pai_expert_review(trades) -> str:
    """Alpha 派投研顾问：作为专家讨论当日交易优化点"""
    
    # 构建讨论主题
    topics = []
    for t in trades:
        topics.append(
            f"品种{t.commodity}，方向{t.direction.value}，"
            f"开盘判断逻辑：{t.core_logic}，"
            f"实际走势：最高{t.day_high} 最低{t.day_low} 收盘{t.day_close}，"
            f"结果：{t.status.value}，盈亏{t.pnl}"
        )
    
    discussion_topic = "\n".join(topics)
    
    # 使用 Alpha 派 expert_discuss（智能问答模式）
    collector = AlphaPaiCollector()
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            None,
            collector.expert_discuss,
            f"作为期货投研专家，请对以下日内T+0交易进行复盘讨论，指出优化点和改进建议：\n{discussion_topic}",
            "",
        )
        return answer
    except Exception as e:
        logger.error(f"Alpha 派专家讨论失败: {e}")
        return ""
    finally:
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="复盘日期 YYYY-MM-DD，默认今天")
    args = parser.parse_args()
    
    asyncio.run(run_evening_review(args.date))
