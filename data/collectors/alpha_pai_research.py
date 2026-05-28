"""
Alpha派投研顾问 — 将 Alpha派 从"数据源"升级为"投研决策参与者"

核心能力：
1. 盘前顾问简报：基于隔夜信息给出当日市场观点
2. 信号验证：对高置信度交易信号进行专家级验证
3. Missed Opportunities：分析当日被系统错过的机会
4. 每周策略探索：跨品种深度扫描，发现新机会
5. 专家讨论：深度问答（复用现有 qa Think 模式）

调用规范：
- 所有方法均为异步（async），内部用 run_in_executor 包装同步的 Alpha派 CLI 调用
- 返回原始文本，不做二次加工（遵循 AGENTS.md）
- 调用失败时返回空字符串并记录日志，不影响主流程
"""

import asyncio
import logging
from typing import Optional
from dataclasses import dataclass
from datetime import datetime

from data.collectors.alpha_pai import AlphaPaiCollector

logger = logging.getLogger("alpha_pai_research")


@dataclass
class SignalValidationResult:
    """Alpha派信号验证结果"""

    verdict: str  # "赞同" / "质疑" / "补充"
    assessment: str  # 详细评估文本
    risk_reminders: str  # 关键风险提醒
    confidence_adjustment: int  # 建议的置信度调整（-2 ~ +1）


class AlphaPaiResearchAdvisor:
    """Alpha派投研顾问"""

    def __init__(self):
        self.collector = AlphaPaiCollector()

    # ------------------------------------------------------------------
    # 内部辅助：异步包装同步调用
    # ------------------------------------------------------------------

    async def _async_qa(self, question: str, mode: str = "Think") -> str:
        """异步执行 Alpha派 qa 调用"""
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                self.collector.qa,
                question,
                mode,
            )
            return result.answer
        except Exception as e:
            logger.error(f"Alpha派 qa 调用失败: {e}")
            return ""

    async def _async_expert_discuss(self, topic: str, context: str = "") -> str:
        """异步执行 Alpha派 expert_discuss 调用"""
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                self.collector.expert_discuss,
                topic,
                context,
            )
            return result
        except Exception as e:
            logger.error(f"Alpha派 expert_discuss 调用失败: {e}")
            return ""

    # ------------------------------------------------------------------
    # 1. 盘前顾问简报
    # ------------------------------------------------------------------

    async def get_premarket_briefing(
        self,
        overnight_news: str = "",
        market_moves: str = "",
        date: str = None,
    ) -> str:
        """
        盘前 Alpha派 顾问简报

        基于隔夜新闻和外盘变动，给出当日市场观点。
        返回原始 Markdown 文本，供 strategy.py 拼接到 LLM prompt。
        """
        date_str = date or datetime.now().strftime("%Y-%m-%d")

        prompt = f"""你是一位资深期货投研顾问。今天是 {date_str} 开盘前，请基于以下信息给出当日市场简报。

【隔夜新闻摘要】
{overnight_news or "暂无重要新闻"}

【外盘/关联市场变动】
{market_moves or "暂无重要变动"}

请输出结构化的盘前简报：
1. **今日最强驱动因素**：宏观/政策/外盘/资金？
2. **重点关注板块/品种**：哪些品种今日可能有高确定性机会？为什么？
3. **最大风险点**：今日需要警惕的反向风险是什么？
4. **操作建议**：对日内T+0策略的建议（顺势/观望/回避哪些品种）

要求：简洁、 actionable、只说高确定性判断，不泛泛而谈。"""

        return await self._async_qa(prompt, mode="Think")

    # ------------------------------------------------------------------
    # 2. 信号验证
    # ------------------------------------------------------------------

    async def validate_signal(
        self,
        commodity: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        target: float,
        core_logic: str,
        confidence: int,
        overnight_news: str = "",
    ) -> SignalValidationResult:
        """
        对单个交易信号进行 Alpha派 专家验证

        输入信号参数，返回验证结果（赞同/质疑/补充）。
        """
        prompt = f"""你是一位严格的期货交易风控专家。请对以下日内T+0交易信号进行独立验证。

【交易信号】
- 品种: {commodity}
- 方向: {direction}
- 建议入场价: {entry_price}
- 止损价: {stop_loss}
- 目标价: {target}
- 核心逻辑: {core_logic}
- 原系统置信度: {confidence}/10

【隔夜信息】
{overnight_news or "无"}

请独立判断：
1. 这个逻辑是否有漏洞？（论据是否充分、因果链是否通顺）
2. 止损设置是否合理？（是否覆盖了主要风险场景）
3. 目标价是否过于乐观/保守？
4. 有没有被原逻辑忽略的重要风险因素？

输出格式：
- ** verdict **: 赞同 / 质疑 / 补充
- **评估**: 50字以内的详细判断
- **风险提醒**: 最关键的一条风险提示
- **置信度调整建议**: -2 / -1 / 0 / +1（只给整数）"""

        answer = await self._async_qa(prompt, mode="Think")

        # 解析 answer 提取结构化信息（简单解析，容错处理）
        verdict = "补充"
        assessment = answer
        risk_reminders = ""
        confidence_adjustment = 0

        if "**verdict**" in answer.lower() or "** verdict **" in answer:
            for line in answer.split("\n"):
                line_lower = line.lower().strip()
                if "verdict" in line_lower:
                    if "赞同" in line:
                        verdict = "赞同"
                    elif "质疑" in line:
                        verdict = "质疑"
                    elif "补充" in line:
                        verdict = "补充"
                if "风险提醒" in line_lower or "risk" in line_lower:
                    risk_reminders = line.split(":", 1)[-1].strip() if ":" in line else line.strip()
                if "置信度调整" in line_lower or "confidence" in line_lower:
                    for num in [-2, -1, 0, 1]:
                        if str(num) in line:
                            confidence_adjustment = num
                            break

        return SignalValidationResult(
            verdict=verdict,
            assessment=assessment,
            risk_reminders=risk_reminders,
            confidence_adjustment=confidence_adjustment,
        )

    # ------------------------------------------------------------------
    # 3. Missed Opportunities 分析
    # ------------------------------------------------------------------

    async def analyze_missed_opportunities(
        self,
        scan_results: list,  # List[dict] with keys: code, name, direction_reason, open, close, change_pct, amplitude_pct
        traded_codes: set,  # 当日实际交易的品种代码集合
    ) -> str:
        """
        分析当日被系统错过的交易机会

        输入当日全部扫描品种的实际走势和观望原因，输出 Alpha派 对观望决策的评估。
        """
        # 构建被观望品种的表现摘要
        watched_performances = []
        for item in scan_results:
            code = item.get("code", "")
            if code in traded_codes:
                continue
            watched_performances.append(
                f"- {item.get('name', code)}({code}): "
                f"开盘{item.get('open', 'N/A')} → 收盘{item.get('close', 'N/A')}, "
                f"涨跌幅{item.get('change_pct', 'N/A')}%, 振幅{item.get('amplitude_pct', 'N/A')}%, "
                f"系统观望原因: {item.get('direction_reason', '无明确信号')}"
            )

        if not watched_performances:
            return ""

        prompt = f"""你是一位期货交易策略审计专家。请评估以下"观望决策"是否正确。

【当日被系统观望的品种实际表现】
{chr(10).join(watched_performances[:20])}  # 最多20个，避免prompt过长

请分析：
1. **正确观望的品种**：哪些品种虽然波动大，但实际没有高确定性机会？为什么？
2. **被错过的机会（Missed Opportunities）**：哪些品种出现了明显的、可捕捉的日内T+0机会，但系统因为什么原因错过了？
3. **系统缺陷诊断**：从以上案例中，策略的哪些参数或逻辑可能需要调整？（如置信度阈值过高、品种覆盖不足、对某类信号的识别盲区）

要求：具体、有案例支撑、不说套话。"""

        return await self._async_qa(prompt, mode="Think")

    # ------------------------------------------------------------------
    # 4. 每周策略探索
    # ------------------------------------------------------------------

    async def weekly_exploration(
        self,
        week_trades_summary: str = "",
        hot_topics: str = "",
        date: str = None,
    ) -> str:
        """
        每周 Alpha派 策略探索

        基于当周市场热点和交易记录，输出下周重点关注清单和策略建议。
        """
        date_str = date or datetime.now().strftime("%Y-%m-%d")

        prompt = f"""你是一位期货策略总监。本周结束（{date_str}），请进行跨品种的深度策略扫描。

【本周交易记录摘要】
{week_trades_summary or "本周无交易"}

【本周市场热点与政策变化】
{hot_topics or "暂无特别热点"}

请输出下周策略展望：
1. **下周重点关注品种**（3-5个，含理由）
2. **潜在跨品种套利/对冲机会**
3. **需要新增的监控品种或剔除的品种**
4. **策略逻辑风险预警**
5. **对现有日内T+0策略的优化建议**

要求：深度、系统、可执行。"""

        return await self._async_qa(prompt, mode="Think")

    # ------------------------------------------------------------------
    # 5. 专家讨论（复用现有）
    # ------------------------------------------------------------------

    async def expert_discuss(self, topic: str, context: str = "") -> str:
        """
        Alpha派 专家讨论（深度问答）

        复用现有的 expert_discuss 能力，提供异步包装。
        """
        return await self._async_expert_discuss(topic, context)
