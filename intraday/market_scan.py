"""
市场行情扫描器
- 每日复盘时扫描全市场日盘表现（09:05-14:55）
- 找出涨幅/跌幅/振幅 Top 品种
- 逆向分析原因，提取普适性市场规律
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd

from data.collectors.market_data import MarketDataCollector
from data.collectors.minute_data import MinuteDataCollector
from data.collectors.brave_search import BraveSearchCollector
from data.collectors.alpha_pai import AlphaPaiCollector
from research.llm_integration import LLMClient, extract_json_from_text

logger = logging.getLogger("market_scan")


class CommodityDayPerformance:
    """单个品种的日盘表现"""
    
    def __init__(self, commodity: str, name: str):
        self.commodity = commodity
        self.name = name
        self.open_price: float = 0.0
        self.high_price: float = 0.0
        self.low_price: float = 0.0
        self.close_price: float = 0.0
        self.prev_settle: float = 0.0
        self.change_pct: float = 0.0
        self.amplitude_pct: float = 0.0
        self.volume: float = 0.0
        self.open_interest: float = 0.0
        self.overnight_news: str = ""
        self.related_market: str = ""
        self.alpha_pai_data: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "commodity": self.commodity,
            "name": self.name,
            "open": self.open_price,
            "high": self.high_price,
            "low": self.low_price,
            "close": self.close_price,
            "prev_settle": self.prev_settle,
            "change_pct": self.change_pct,
            "amplitude_pct": self.amplitude_pct,
            "volume": self.volume,
            "open_interest": self.open_interest,
        }


class MarketScanResult:
    """市场扫描结果"""
    
    def __init__(self, date: str):
        self.date = date
        self.performances: List[CommodityDayPerformance] = []
        self.top_gainers: List[CommodityDayPerformance] = []
        self.top_losers: List[CommodityDayPerformance] = []
        self.top_amplitude: List[CommodityDayPerformance] = []
        self.market_scan_summary: str = ""
        self.extracted_lessons: List[Dict[str, Any]] = []
        self.unique_samples: List[str] = []
        self.scan_report: str = ""
    
    def to_markdown(self) -> str:
        """生成市场扫描报告 Markdown"""
        lines = [
            f"# 市场行情扫描 ({self.date})",
            "",
            "## 当日市场概况",
            f"- 扫描品种数: {len(self.performances)}",
            f"- 上涨品种: {len([p for p in self.performances if p.change_pct > 0])}",
            f"- 下跌品种: {len([p for p in self.performances if p.change_pct < 0])}",
            f"- 平盘品种: {len([p for p in self.performances if p.change_pct == 0])}",
            "",
        ]
        
        if self.top_gainers:
            lines.append("## 涨幅 Top 5")
            lines.append("")
            lines.append("| 品种 | 名称 | 开盘价 | 收盘价 | 涨跌幅 | 振幅 |")
            lines.append("|------|------|--------|--------|--------|------|")
            for p in self.top_gainers[:5]:
                lines.append(f"| {p.commodity} | {p.name} | {p.open_price} | {p.close_price} | {p.change_pct:+.2f}% | {p.amplitude_pct:.2f}% |")
            lines.append("")
        
        if self.top_losers:
            lines.append("## 跌幅 Top 5")
            lines.append("")
            lines.append("| 品种 | 名称 | 开盘价 | 收盘价 | 涨跌幅 | 振幅 |")
            lines.append("|------|------|--------|--------|--------|------|")
            for p in self.top_losers[:5]:
                lines.append(f"| {p.commodity} | {p.name} | {p.open_price} | {p.close_price} | {p.change_pct:+.2f}% | {p.amplitude_pct:.2f}% |")
            lines.append("")
        
        if self.top_amplitude:
            lines.append("## 振幅 Top 5")
            lines.append("")
            lines.append("| 品种 | 名称 | 最高价 | 最低价 | 振幅 | 涨跌幅 |")
            lines.append("|------|------|--------|--------|------|--------|")
            for p in self.top_amplitude[:5]:
                lines.append(f"| {p.commodity} | {p.name} | {p.high_price} | {p.low_price} | {p.amplitude_pct:.2f}% | {p.change_pct:+.2f}% |")
            lines.append("")
        
        if self.market_scan_summary:
            lines.append("## LLM 逆向分析")
            lines.append("")
            lines.append(self.market_scan_summary)
            lines.append("")
        
        if self.extracted_lessons:
            lines.append("## 提取的普适性规律")
            lines.append("")
            for i, lesson in enumerate(self.extracted_lessons, 1):
                lines.append(f"{i}. **[{lesson.get('category', 'general')}]** {lesson.get('lesson', '')}")
                lines.append(f"   - 普适性评分: {lesson.get('universality_score', 'N/A')}/10")
                lines.append(f"   - 涉及品种: {', '.join(lesson.get('commodities', []))}")
                lines.append("")
        
        if self.unique_samples:
            lines.append("## 独特样本（仅供参考，不进入认知库）")
            lines.append("")
            for s in self.unique_samples:
                lines.append(f"- {s}")
            lines.append("")
        
        lines.append("---")
        lines.append(f"*扫描时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        return "\n".join(lines)


class MarketScanner:
    """市场行情扫描器"""
    
    def __init__(self):
        self.market = MarketDataCollector()
        self.minute = MinuteDataCollector()
        self.brave = BraveSearchCollector()
        self.alpha_pai = AlphaPaiCollector()
        self.llm: Optional[LLMClient] = None
    
    async def _get_llm(self) -> LLMClient:
        if self.llm is None:
            self.llm = LLMClient()
        return self.llm
    
    def load_commodities(self, config_path: str = "config/settings.yaml") -> List[Dict[str, str]]:
        """加载监控品种列表"""
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        commodities = []
        for group, items in cfg.get("monitored_commodities", {}).items():
            for item in items:
                commodities.append({
                    "code": item["code"],
                    "name": item["name"],
                    "group": group,
                })
        return commodities
    
    def _is_today(self, date_str: str) -> bool:
        return date_str == datetime.now().strftime("%Y-%m-%d")
    
    def _get_minute_performance(self, code: str, name: str, date: str) -> Optional[CommodityDayPerformance]:
        """使用分钟线数据获取历史日盘表现"""
        df = self.minute._get_data(code, date)
        
        if df is None or df.empty:
            return None
        
        # 过滤到策略窗口 09:05-14:55
        window_df = self.minute._filter_by_time(df, "09:05", "14:55")
        if window_df.empty:
            return None
        
        # 过滤掉 close 为 0、负数或 NaN 的异常 K 线
        window_df = window_df[window_df["close"].apply(lambda x: isinstance(x, (int, float)) and x > 0 and x == x)]
        if window_df.empty:
            logger.warning(f"{code} 分钟线数据全部异常（close≤0 或 NaN），跳过")
            return None
        
        perf = CommodityDayPerformance(code, name)
        perf.open_price = float(window_df.iloc[0]["open"])
        perf.high_price = float(window_df["high"].max())
        perf.low_price = float(window_df["low"].min())
        perf.close_price = float(window_df.iloc[-1]["close"])
        
        # 获取昨结（使用当日第一根K线的 open 作为参考，或从实时快照获取）
        # 历史数据中无法直接获取昨结，用开盘价近似
        perf.prev_settle = perf.open_price
        
        # 计算涨跌幅和振幅
        if perf.prev_settle != 0:
            perf.change_pct = round((perf.close_price - perf.prev_settle) / perf.prev_settle * 100, 2)
            perf.amplitude_pct = round((perf.high_price - perf.low_price) / perf.prev_settle * 100, 2)
        
        # 成交量和持仓量（分钟线可能不含有，设为0）
        if "volume" in window_df.columns:
            perf.volume = float(window_df["volume"].sum())
        
        # 分钟线不含持仓量，尝试从新浪快照补充（当天收盘后复盘时数据准确）
        try:
            snapshot = self.market.get_snapshot(code)
            if snapshot:
                if snapshot.open_interest > 0:
                    perf.open_interest = snapshot.open_interest
                if perf.volume == 0 and snapshot.volume > 0:
                    perf.volume = snapshot.volume
        except Exception:
            pass
        
        # 数据校验
        if not self._validate_performance(perf):
            logger.warning(f"{code} 数据校验失败，标记为异常")
            return None
        
        return perf
    
    def _validate_performance(self, perf: CommodityDayPerformance) -> bool:
        """数据校验：过滤异常值"""
        def _is_invalid(price: float) -> bool:
            """检查价格是否为 0、负数或 NaN"""
            return not isinstance(price, (int, float)) or price <= 0 or price != price

        # 各关键价格为 0、负数或 NaN
        if _is_invalid(perf.close_price) or _is_invalid(perf.open_price) or _is_invalid(perf.high_price) or _is_invalid(perf.low_price):
            return False
        # 价格逻辑校验：high >= low
        if perf.high_price < perf.low_price:
            return False
        # 涨跌幅绝对值超过20%（除特殊品种外）
        if abs(perf.change_pct) > 20 and perf.commodity not in ["T", "TF", "TS"]:
            return False
        return True
    
    async def scan_all_commodities(
        self,
        commodities: List[Dict[str, str]] = None,
        top_n: int = 5,
        date: str = None,
    ) -> MarketScanResult:
        """
        扫描所有品种的日盘表现
        
        Args:
            commodities: 指定品种列表
            top_n: Top N 排名
            date: 指定日期 YYYY-MM-DD，默认今天
        """
        date_str = date or datetime.now().strftime("%Y-%m-%d")
        result = MarketScanResult(date=date_str)
        
        if commodities is None:
            commodities = self.load_commodities()
        
        is_today = self._is_today(date_str)
        logger.info(f"开始扫描 {len(commodities)} 个品种的日盘表现 ({date_str})...")
        
        if is_today:
            # 今天：使用实时快照
            snapshots = await self.market.async_get_snapshots([c["code"] for c in commodities])
            for comm in commodities:
                code = comm["code"]
                snapshot = snapshots.get(code)
                if not snapshot or not isinstance(snapshot.last, (int, float)) or snapshot.last <= 0 or snapshot.last != snapshot.last:
                    logger.warning(f"无法获取有效行情 {code}，跳过")
                    continue
                
                perf = CommodityDayPerformance(code, comm["name"])
                perf.open_price = snapshot.open
                perf.high_price = snapshot.high
                perf.low_price = snapshot.low
                perf.close_price = snapshot.last
                perf.prev_settle = snapshot.prev_settle
                perf.change_pct = snapshot.change_pct
                perf.amplitude_pct = snapshot.amplitude_pct
                perf.volume = snapshot.volume
                perf.open_interest = snapshot.open_interest
                
                if self._validate_performance(perf):
                    result.performances.append(perf)
        else:
            # 历史日期：使用分钟线数据
            for comm in commodities:
                code = comm["code"]
                try:
                    perf = self._get_minute_performance(code, comm["name"], date_str)
                    if perf:
                        result.performances.append(perf)
                except Exception as e:
                    logger.warning(f"获取历史数据失败 {code}: {e}")
        
        logger.info(f"成功获取 {len(result.performances)} 个品种数据")
        
        # 计算排名
        if result.performances:
            result.top_gainers = sorted(
                [p for p in result.performances if p.change_pct is not None],
                key=lambda x: x.change_pct, reverse=True
            )[:top_n]
            result.top_losers = sorted(
                [p for p in result.performances if p.change_pct is not None],
                key=lambda x: x.change_pct
            )[:top_n]
            result.top_amplitude = sorted(
                [p for p in result.performances if p.amplitude_pct is not None],
                key=lambda x: x.amplitude_pct, reverse=True
            )[:top_n]
        
        # 对 Top 品种获取基本面数据
        top_commodities = list(set(
            [p.commodity for p in result.top_gainers + result.top_losers + result.top_amplitude]
        ))
        
        logger.info(f"对 Top {len(top_commodities)} 品种获取深度数据...")
        
        for code in top_commodities:
            perf = next((p for p in result.performances if p.commodity == code), None)
            if not perf:
                continue
            
            try:
                # 隔夜新闻
                news = await self.brave.search(
                    query=f"{perf.name} 期货 最新 消息 隔夜",
                    count=3,
                    freshness="pd",
                )
                perf.overnight_news = "\n".join([f"{r.title}: {r.description}" for r in news[:2]])
            except Exception as e:
                logger.error(f"搜索新闻失败 {code}: {e}")
            
            try:
                # 关联市场
                related = await self._fetch_related_market(perf.name, code)
                perf.related_market = related
            except Exception as e:
                logger.error(f"搜索外盘失败 {code}: {e}")
            
            try:
                # Alpha派基本面
                alpha_data = await self._fetch_alpha_pai(perf.name, code)
                perf.alpha_pai_data = alpha_data
            except Exception as e:
                logger.error(f"Alpha派数据失败 {code}: {e}")
        
        # LLM 逆向分析
        if result.performances:
            result.market_scan_summary = await self._llm_market_analysis(result)
            result.extracted_lessons, result.unique_samples = await self._extract_lessons(result)
        
        return result
    
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
        
        results = await self.brave.search(query, count=3, freshness="pd")
        return "\n".join([f"{r.title}: {r.description}" for r in results[:2]])
    
    async def _fetch_alpha_pai(self, name: str, code: str) -> str:
        """获取 Alpha派 基本面数据"""
        keywords_map = {
            "RB": ["螺纹钢", "钢铁", "库存", "供需"],
            "I": ["铁矿石", "进口矿", "矿山"],
            "CU": ["铜", "铜矿", "冶炼", "LME铜"],
            "M": ["豆粕", "大豆", "压榨", "库存"],
            "C": ["玉米", "临储", "饲用"],
            "SC": ["原油", "OPEC", "原油库存"],
            "AL": ["铝", "电解铝", "氧化铝"],
            "AU": ["黄金", "美联储", "实际利率"],
        }
        keywords = keywords_map.get(code, [name])
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.alpha_pai.get_fundamental_data,
            keywords,
            ["report", "roadShow", "comment"],
            None,
            14,
        )
    
    async def _llm_market_analysis(self, result: MarketScanResult) -> str:
        """LLM 逆向分析 Top 品种"""
        
        # 构建 Top 品种数据
        top_data = []
        for p in result.top_gainers[:3] + result.top_losers[:3] + result.top_amplitude[:2]:
            top_data.append(
                f"品种: {p.name}({p.commodity})\n"
                f"  开盘: {p.open_price} 收盘: {p.close_price} 最高: {p.high_price} 最低: {p.low_price}\n"
                f"  涨跌幅: {p.change_pct:+.2f}% 振幅: {p.amplitude_pct:.2f}%\n"
                f"  成交量: {p.volume:,.0f} 持仓量: {p.open_interest:,.0f}\n"
                f"  隔夜新闻: {p.overnight_news}\n"
                f"  关联市场: {p.related_market}\n"
                f"  基本面: {p.alpha_pai_data}"
            )
        
        # 全市场统计
        all_changes = [p.change_pct for p in result.performances]
        avg_change = sum(all_changes) / len(all_changes) if all_changes else 0
        up_count = len([c for c in all_changes if c > 0])
        down_count = len([c for c in all_changes if c < 0])
        
        prompt = f"""你是一位期货市场的逆向分析专家。请分析以下当日日盘（09:05-14:55）的市场表现，找出共性规律。

【全市场统计】
- 平均涨跌幅: {avg_change:+.2f}%
- 上涨品种: {up_count}个
- 下跌品种: {down_count}个
- 总扫描品种: {len(result.performances)}个

【Top 品种表现】
{chr(10).join(top_data)}

请分析：
1. 今日市场最强的驱动力是什么？（宏观、政策、外盘、资金？）
2. Top 涨跌品种的共性特征是什么？（板块联动、消息面、量仓配合？）
3. 哪些现象是"可跨品种复用的规律"？哪些只是"独特样本"？
4. 对日内T+0交易策略有什么启示？

输出要求：
- 客观、具体，不要泛泛而谈
- 明确指出哪些观察具有普适性，哪些可能只是当日特殊现象
- 给出可操作的策略改进建议
"""
        
        client = await self._get_llm()
        try:
            resp = await client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000,
            )
            return resp.content
        except Exception as e:
            logger.error(f"LLM 市场分析失败: {e}")
            return "市场分析生成失败。"
        finally:
            await client.close()
    
    async def _extract_lessons(self, result: MarketScanResult) -> Tuple[List[Dict], List[str]]:
        """从市场扫描中提取结构化的普适性 lessons"""
        
        top_data = []
        for p in result.top_gainers[:3] + result.top_losers[:3]:
            top_data.append(
                f"品种: {p.commodity} 涨跌幅: {p.change_pct:+.2f}% 振幅: {p.amplitude_pct:.2f}% "
                f"新闻: {p.overnight_news}"
            )
        
        prompt = f"""你是一位策略进化专家。请从以下市场观察中提取普适性规律（lessons）。

【重要原则】
- 只提取可跨品种验证的规律（如"跳空+利空确认=高确定性顺势"）
- 如果一个现象只在单个品种上出现，标记为"独特样本"
- 每条 lesson 必须给出跨品种适用性评分（1-10，10=非常普适）
- 适用性评分 < 6 的 lesson 不要输出

【当日市场观察】
{chr(10).join(top_data)}

【已有分析】
{result.market_scan_summary[:1000]}

请输出 JSON 数组格式：
[
    {{
        "lesson": "核心规律（简洁、可跨品种执行，50字以内）",
        "category": "signal_filter / entry_timing / exit_timing / risk_control / market_regime / general",
        "universality_score": 6-10 的整数（跨品种适用性）,
        "commodities": ["适用品种代码"],
        "is_unique_sample": false
    }}
]

如果只出现独特样本，输出空数组 []。
"""
        
        client = await self._get_llm()
        try:
            resp = await client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1500,
            )
            data = extract_json_from_text(resp.content)
            
            if not isinstance(data, list):
                return [], []
            
            lessons = []
            unique_samples = []
            
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                if item.get("is_unique_sample", False):
                    unique_samples.append(item.get("lesson", ""))
                    continue
                
                score = int(item.get("universality_score", 0))
                if score < 6:
                    unique_samples.append(item.get("lesson", ""))
                    continue
                
                lessons.append({
                    "lesson": item.get("lesson", ""),
                    "category": item.get("category", "general"),
                    "universality_score": score,
                    "commodities": item.get("commodities", []),
                })
            
            return lessons, unique_samples
        
        except Exception as e:
            logger.error(f"提取 lessons 失败: {e}")
            return [], []
        finally:
            await client.close()
    
    async def save_report(self, result: MarketScanResult):
        """保存市场扫描报告"""
        report_dir = Path(__file__).parent.parent / "reports" / "intraday"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = report_dir / f"market_scan_{result.date}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(result.to_markdown())
        
        logger.info(f"市场扫描报告已保存: {report_path}")
        return report_path
    
    async def close(self):
        await self.brave.close()
        if self.llm:
            await self.llm.close()


# 便捷函数
async def run_market_scan(
    commodities: List[Dict[str, str]] = None,
    top_n: int = 5,
    date: str = None,
) -> MarketScanResult:
    """执行市场行情扫描"""
    scanner = MarketScanner()
    try:
        result = await scanner.scan_all_commodities(commodities, top_n, date)
        await scanner.save_report(result)
        return result
    finally:
        await scanner.close()
