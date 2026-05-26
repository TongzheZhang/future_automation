"""
日内T+0交易策略逻辑
- 隔夜新闻 + 开盘行情 → LLM 判断日内方向
- 置信度过滤，只交易高确定性机会
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from data.collectors.brave_search import BraveSearchCollector
from data.collectors.market_data import MarketDataCollector, MarketSnapshot
from research.llm_integration import LLMClient, extract_json_from_text
from intraday.models import IntradaySignal, Direction, MarketSnapshotData, CONTRACT_SIZE

logger = logging.getLogger(__name__)

# 品种最小变动价位对应的每手价值（简化）
TICK_VALUE = {
    "AG": 15,
    "AL": 25,
    "AO": 20,
    "AP": 10,
    "AU": 1000,
    "BC": 50,
    "BR": 25,
    "BU": 10,
    "C": 10,
    "CF": 25,
    "CJ": 5,
    "CS": 10,
    "CU": 50,
    "EB": 5,
    "EG": 10,
    "FG": 20,
    "HC": 10,
    "I": 50,
    "J": 50,
    "JD": 10,
    "JM": 30,
    "L": 5,
    "LC": 1,
    "LH": 16,
    "LU": 10,
    "M": 10,
    "MA": 10,
    "NI": 10,
    "NR": 50,
    "OI": 10,
    "P": 10,
    "PB": 25,
    "PF": 5,
    "PG": 20,
    "PK": 5,
    "PP": 5,
    "PX": 5,
    "RB": 10,
    "RM": 10,
    "RU": 50,
    "SA": 20,
    "SC": 100,
    "SF": 5,
    "SH": 30,
    "SI": 5,
    "SM": 5,
    "SN": 10,
    "SP": 10,
    "SR": 10,
    "SS": 25,
    "TA": 5,
    "UR": 20,
    "V": 5,
    "Y": 10,
    "ZN": 25,
}


class IntradayStrategy:
    """日内T+0策略"""
    
    def __init__(self):
        self.brave = BraveSearchCollector()
        self.market = MarketDataCollector()
        self.llm: Optional[LLMClient] = None
    
    async def _get_llm(self) -> LLMClient:
        if self.llm is None:
            self.llm = LLMClient()
        return self.llm
    
    async def scan_commodity(self, commodity: str, name: str) -> IntradaySignal:
        """
        扫描单个品种的日内交易机会
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # 1. 获取开盘行情
        snapshot = await self.market.async_get_snapshot(commodity)
        if not snapshot:
            logger.error(f"无法获取行情 {commodity}")
            return IntradaySignal(date=date_str, commodity=commodity, commodity_name=name)
        
        # 2. 搜索隔夜新闻
        overnight_news = await self._fetch_overnight_news(name, commodity)
        
        # 3. 搜索外盘/关联市场
        related_market = await self._fetch_related_market(name, commodity)
        
        # 4. LLM 判断
        signal = await self._llm_judge(
            commodity=commodity,
            name=name,
            snapshot=snapshot,
            overnight_news=overnight_news,
            related_market=related_market,
        )
        
        signal.market_snapshot = MarketSnapshotData(
            commodity=snapshot.commodity,
            name=snapshot.name,
            time=snapshot.time,
            open=snapshot.open,
            high=snapshot.high,
            low=snapshot.low,
            last=snapshot.last,
            prev_settle=snapshot.prev_settle,
            settle=snapshot.settle,
            bid=snapshot.bid,
            ask=snapshot.ask,
            open_interest=snapshot.open_interest,
            volume=snapshot.volume,
            date=snapshot.date,
            gap_pct=snapshot.gap_pct,
            change_pct=snapshot.change_pct,
            amplitude_pct=snapshot.amplitude_pct,
        )
        
        return signal
    
    async def _fetch_overnight_news(self, name: str, code: str) -> str:
        """获取隔夜新闻摘要"""
        try:
            results = await self.brave.search(
                query=f"{name} 期货 最新 消息 隔夜",
                count=5,
                freshness="pd",
            )
            summaries = []
            for r in results[:3]:
                summaries.append(f"{r.title}: {r.description[:80]}")
            return "\n".join(summaries) if summaries else "无重大隔夜新闻"
        except Exception as e:
            logger.error(f"搜索隔夜新闻失败 {code}: {e}")
            return "新闻获取失败"
    
    async def _fetch_related_market(self, name: str, code: str) -> str:
        """获取关联市场/外盘信息"""
        queries = {
            "RB": "新加坡铁矿石 隔夜 涨跌幅",
            "I": "新加坡铁矿石 隔夜 涨跌幅",
            "CU": "LME铜 隔夜 涨跌幅",
            "M": "CBOT大豆 美豆 隔夜 涨跌幅",
            "C": "CBOT玉米 隔夜 涨跌幅",
            "SC": "布伦特原油 WTI 隔夜 涨跌幅",
            "AL": "LME铝 隔夜 涨跌幅",
            "AU": "国际金价 黄金 隔夜 涨跌幅",
        }
        
        query = queries.get(code)
        if not query:
            return "无直接关联外盘"
        
        try:
            results = await self.brave.search(query, count=3, freshness="pd")
            summaries = []
            for r in results[:2]:
                summaries.append(f"{r.title}: {r.description[:80]}")
            return "\n".join(summaries) if summaries else "外盘信息获取失败"
        except Exception as e:
            logger.error(f"搜索外盘失败 {code}: {e}")
            return "外盘信息获取失败"
    
    async def _llm_judge(
        self,
        commodity: str,
        name: str,
        snapshot: MarketSnapshot,
        overnight_news: str,
        related_market: str,
    ) -> IntradaySignal:
        """LLM 综合判断日内方向"""
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        tick_value = TICK_VALUE.get(commodity, 10)
        
        system_prompt = """你是一个只做日内T+0的期货交易员，交易时间严格限制在日盘 09:00-15:00。
每天只做1-2笔高确定性交易，14:55前必须平仓，不持仓过夜，不交易夜盘。
你的风格是：客观、冷静、只看高确定性机会，没有把握就观望。
输出必须是 JSON 格式。"""
        
        # 量仓变化提示（只有当前快照，无历史对比，引导 LLM 做定性判断）
        oi_hint = "增仓" if snapshot.open_interest > 0 else "持仓数据缺失"
        vol_hint = "有成交" if snapshot.volume > 0 else "成交清淡"

        user_prompt = f"""品种: {name} ({commodity})
日期: {date_str}

【开盘行情】
- 昨结: {snapshot.prev_settle}
- 开盘价: {snapshot.open}（集合竞价产生，代表隔夜情绪）
- 09:05最新价: {snapshot.last}（开盘扫描时的实时成交价格，作为入场基准）
- 最高: {snapshot.high}
- 最低: {snapshot.low}
- 跳空幅度: {snapshot.gap_pct}%
- 涨跌幅(基于昨结): {snapshot.change_pct}%
- 持仓量: {snapshot.open_interest:,.0f}
- 成交量: {snapshot.volume:,.0f}

【量仓变化分析】
- 当前持仓状态: {oi_hint}
- 当前成交状态: {vol_hint}
- 请结合价格方向与量仓关系判断资金主动流入还是流出：
  * 价涨 + 持仓增 + 成交放 = 多头主动开仓，趋势较强
  * 价涨 + 持仓减 + 成交缩 = 空头回补，反弹持续性存疑
  * 价跌 + 持仓增 + 成交放 = 空头主动开仓，下行有动能
  * 价跌 + 持仓减 + 成交缩 = 多头离场，弱势整理

【隔夜新闻】
{overnight_news}

【关联市场/外盘】
{related_market}

【交易规则】
- 每手最小变动价值约 {tick_value} 元
- 只做1个方向，不双向交易
- 14:55前必须平仓
- 止损严格，单笔亏损控制在合理范围
- 建议入场价应参考日内实时价，而非直接以开盘价成交

请判断并输出 JSON：
{{
    "should_trade": true/false,
    "direction": "LONG/SHORT/NO_TRADE",
    "entry_price": 建议入场价（数字，参考日内实时价）,
    "stop_loss_price": 止损价（数字）,
    "target_price": 目标价（数字）,
    "confidence": 0-10 的整数,
    "core_logic": "核心逻辑（50字以内）",
    "risk_note": "最大风险提示"
}}

判断原则：
- 跳空>2%时，追高风险大，偏向观望或做回补
- 增仓上行=多头主动，减仓上行=空头回补；务必结合量仓变化判断
- 隔夜重大利空/利多+跳空确认 = 高确定性顺势交易
- 开盘价与日内实时价背离时，以实时价方向为准
- 没有明确催化剂时，建议观望
"""
        
        llm = await self._get_llm()
        try:
            response = await llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=1000,
            )
            result = extract_json_from_text(response.content)
        except Exception as e:
            logger.error(f"LLM 判断失败 {commodity}: {e}")
            return IntradaySignal(
                date=date_str,
                commodity=commodity,
                commodity_name=name,
                overnight_news=overnight_news,
            )
        
        direction_str = result.get("direction", "NO_TRADE").upper()
        if direction_str not in ["LONG", "SHORT", "NO_TRADE"]:
            direction_str = "NO_TRADE"
        
        signal = IntradaySignal(
            date=date_str,
            commodity=commodity,
            commodity_name=name,
            direction=Direction(direction_str),
            entry_price=float(result.get("entry_price", snapshot.open)),
            confidence=int(result.get("confidence", 0)),
            stop_loss_price=float(result.get("stop_loss_price", 0)),
            target_price=float(result.get("target_price", 0)),
            core_logic=result.get("core_logic", ""),
            overnight_news=overnight_news,
        )
        
        # 如果 LLM 建议不交易，但给了方向，也设为 NO_TRADE
        if not result.get("should_trade", False):
            signal.direction = Direction.NO_TRADE
        
        logger.info(
            f"信号生成 {commodity}: {signal.direction.value} | "
            f"置信度={signal.confidence} | 入场={signal.entry_price}"
        )
        return signal
    
    async def scan_all(
        self,
        commodities: List[Dict[str, str]],
        min_confidence: int = 7,
    ) -> List[IntradaySignal]:
        """
        扫描所有品种，返回高置信度信号
        commodities: [{"code": "RB", "name": "螺纹钢"}, ...]
        """
        signals = []
        for comm in commodities:
            try:
                signal = await self.scan_commodity(comm["code"], comm["name"])
                signals.append(signal)
            except Exception as e:
                logger.error(f"扫描失败 {comm['code']}: {e}")
        
        # 过滤 + 排序
        valid = [s for s in signals if s.should_trade()]
        valid.sort(key=lambda x: x.confidence, reverse=True)
        
        # 最多返回前3个
        return valid[:3]
    
    async def close(self):
        await self.brave.close()
        if self.llm:
            await self.llm.close()
