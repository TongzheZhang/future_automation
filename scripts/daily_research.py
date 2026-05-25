"""
每日自动投研脚本
- 执行完整的投研流程：政策监控 → 数据采集 → 分析 → 信号生成 → 报告输出
"""

import os
import sys
import json
import asyncio
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

import yaml

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.policy_analyzer import PolicyAnalyzer
from research.fundamental_analyzer import FundamentalAnalyzer
from research.chain_mapper import ChainMapper
from research.signal_generator import SignalGenerator
from signals.evaluator import SignalEvaluator
from signals.models import SignalBatch, TradingSignal
from data.collectors.alpha_pai import AlphaPaiCollector
from data.collectors.policy_news import PolicyNewsCollector
from data.collectors.fundamental import FundamentalCollector

# 配置日志
log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            log_dir / f"research_{datetime.now().strftime('%Y%m%d_%H%M')}.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("daily_research")


class DailyResearchPipeline:
    """每日投研流水线"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.policy_analyzer = PolicyAnalyzer(config_path)
        self.fundamental_analyzer = FundamentalAnalyzer(config_path)
        self.chain_mapper = ChainMapper()
        self.signal_generator = SignalGenerator()
        self.signal_evaluator = SignalEvaluator(config_path)
        self.alpha_pai = AlphaPaiCollector()
        self.policy_collector = PolicyNewsCollector(config_path)
        self.fundamental_collector = FundamentalCollector(config_path)
        
        self.results: Dict[str, Any] = {}
    
    async def run(self, focused_commodities: Optional[List[str]] = None):
        """执行完整的每日投研流程"""
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("开始每日投研流程")
        logger.info("=" * 60)
        
        # 确定监控品种
        commodities = self._get_commodities(focused_commodities)
        logger.info(f"监控品种数量: {len(commodities)}")
        
        # Step 1: 政策监控
        logger.info("\n[Step 1/5] 政策监控与分析...")
        try:
            policy_news = await self.policy_collector.fetch_all(days_back=1)
            logger.info(f"采集到 {len(policy_news)} 条政策新闻")
            
            # 分析每条政策
            policy_analyses = []
            for news in policy_news:
                try:
                    analysis = await self.policy_analyzer.analyze_policy(
                        policy_title=news.title,
                        policy_content=news.summary,
                        ministry=news.ministry,
                    )
                    if analysis:
                        policy_analyses.append(analysis)
                except Exception as e:
                    logger.error(f"政策分析失败 ({news.title[:50]}): {e}")
            
            high_impact = self.policy_analyzer.filter_high_impact(policy_analyses)
            try:
                policy_summary = await self.policy_analyzer.generate_policy_summary(high_impact)
            except Exception as e:
                logger.error(f"政策摘要生成失败: {e}")
                policy_summary = "政策摘要生成失败，请查看原始分析数据。"
            
            self.results["policy"] = {
                "news_count": len(policy_news),
                "analyses": policy_analyses,
                "high_impact": high_impact,
                "summary": policy_summary,
            }
            logger.info(f"政策分析完成: {len(policy_analyses)} 条分析, {len(high_impact)} 条高影响")
        except Exception as e:
            logger.error(f"政策监控阶段失败: {e}")
            self.results["policy"] = {"error": str(e), "summary": "政策监控失败"}
        
        # Step 2: 基本面数据采集
        logger.info("\n[Step 2/5] 基本面数据采集...")
        fundamental_data = {}
        
        for comm in commodities:
            try:
                keywords = comm.get("alpha_pai_keywords", [comm["name"]])
                data = await self.fundamental_collector.fetch_all_for_commodity(
                    commodity=comm["code"],
                    keywords=keywords,
                )
                fundamental_data[comm["code"]] = data
                logger.info(f"  基本面数据完成: {comm['code']} ({comm['name']})")
            except Exception as e:
                logger.error(f"  基本面数据失败 {comm['code']}: {e}")
        
        self.results["fundamental"] = fundamental_data
        
        # Step 3: 深度分析与产业链
        logger.info("\n[Step 3/5] 深度分析与产业链...")
        analyses = []
        
        for comm in commodities:
            comm_code = comm["code"]
            comm_name = comm["name"]
            
            # 查找相关的政策分析
            related_policy = None
            for pa in self.results["policy"].get("analyses", []):
                for impact in pa.get("direct_impacts", []):
                    if comm_name in impact.get("commodity", "") or any(kw in impact.get("commodity", "") for kw in comm.get("alpha_pai_keywords", [])):
                        related_policy = pa
                        break
                if related_policy:
                    break
            
            # 基本面分析
            fundamental = None
            comm_fund_data = fundamental_data.get(comm_code, {})
            alpha_pai_research = comm_fund_data.get("alpha_pai_research")
            
            if alpha_pai_research or comm_fund_data.get("indicators"):
                try:
                    fundamental_input = {
                        "commodity": comm_name,
                        "indicators": {},
                    }
                    for indicator_name, indicator_data in comm_fund_data.get("indicators", {}).items():
                        if hasattr(indicator_data, 'description'):
                            fundamental_input["indicators"][indicator_name] = indicator_data.description
                    
                    if alpha_pai_research:
                        fundamental_input["alpha_pai_summary"] = alpha_pai_research[:1000]
                    
                    fundamental = await self.fundamental_analyzer.deep_analyze_with_llm(
                        commodity=comm_name,
                        fundamental_data=fundamental_input,
                        alpha_pai_research=alpha_pai_research,
                    )
                except Exception as e:
                    logger.error(f"基本面分析失败 {comm_code}: {e}")
            
            # 产业链分析
            chain = None
            if related_policy or fundamental:
                try:
                    chain = await self.chain_mapper.analyze_transmission(
                        commodity=comm_code,
                        policy_impact=related_policy,
                    )
                except Exception as e:
                    logger.error(f"产业链分析失败 {comm_code}: {e}")
            
            analyses.append({
                "commodity_code": comm_code,
                "commodity_name": comm_name,
                "exchange": comm["exchange"],
                "policy_analysis": related_policy,
                "fundamental_analysis": fundamental,
                "chain_analysis": chain,
            })
        
        self.results["analyses"] = analyses
        logger.info(f"分析完成: {len(analyses)} 个品种")
        
        # Step 4: 信号生成
        logger.info("\n[Step 4/5] 交易信号生成...")
        try:
            raw_signals = await self.signal_generator.generate_batch(analyses)
            logger.info(f"原始信号数量: {len(raw_signals)}")
        except Exception as e:
            logger.error(f"信号生成失败: {e}")
            raw_signals = []
        
        # Step 5: 信号评估与过滤
        logger.info("\n[Step 5/5] 信号评估与过滤...")
        try:
            filtered_signals = self.signal_evaluator.filter_signals(raw_signals)
            logger.info(f"过滤后信号数量: {len(filtered_signals)}")
        except Exception as e:
            logger.error(f"信号过滤失败: {e}")
            filtered_signals = raw_signals
        
        self.results["signals"] = {
            "raw": raw_signals,
            "filtered": filtered_signals,
        }
        
        # 生成日报
        await self._generate_daily_report()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n每日投研完成，耗时 {elapsed:.1f} 秒")
        logger.info("=" * 60)
        
        return self.results
    
    def _get_commodities(self, focused: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """获取监控品种列表"""
        all_commodities = []
        for group, commodities in self.config.get("monitored_commodities", {}).items():
            all_commodities.extend(commodities)
        
        if focused:
            return [c for c in all_commodities if c["code"] in focused]
        return all_commodities
    
    async def _generate_daily_report(self):
        """生成日报"""
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        lines = [
            f"# 期货投研日报 ({report_time})",
            "",
            "## 一、政策摘要",
            "",
        ]
        
        policy_summary = self.results.get("policy", {}).get("summary", "暂无政策数据")
        lines.append(policy_summary)
        lines.append("")
        
        # 高影响政策详情
        high_impact = self.results.get("policy", {}).get("high_impact", [])
        if high_impact:
            lines.append("### 高影响政策详情")
            lines.append("")
            for i, pa in enumerate(high_impact[:5], 1):
                lines.append(f"**{i}. {pa.get('policy_title', '未命名')}** ({pa.get('ministry', '')})")
                for impact in pa.get("direct_impacts", [])[:3]:
                    lines.append(f"- {impact.get('commodity', '')}: {impact.get('direction', '')} — {impact.get('mechanism', '')}")
                lines.append("")
        
        lines.extend(["## 二、交易信号", ""])
        
        filtered_signals = self.results.get("signals", {}).get("filtered", [])
        if filtered_signals:
            lines.append(f"**今日共 {len(filtered_signals)} 个有效信号**")
            lines.append("")
            
            for i, sig in enumerate(filtered_signals, 1):
                lines.append(f"### {i}. {sig.commodity_name} ({sig.commodity_code}) — {sig.direction.value}")
                lines.append(f"- **置信度**: {sig.confidence:.2f} ({sig.conviction_level})")
                lines.append(f"- **核心逻辑**: {sig.core_logic}")
                lines.append(f"- **建议持仓**: {sig.holding_period_days} 天")
                lines.append(f"- **建议仓位**: {sig.position_sizing}")
                lines.append(f"- **风险等级**: {sig.risk_level.value}")
                lines.append(f"- **止损逻辑**: {sig.stop_loss_logic}")
                lines.append(f"- **目标逻辑**: {sig.target_logic}")
                if sig.risk_factors:
                    lines.append(f"- **风险因素**: {', '.join(sig.risk_factors)}")
                if sig.required_confirmations:
                    lines.append(f"- **待确认**: {', '.join(sig.required_confirmations)}")
                lines.append("")
        else:
            lines.append("**今日无有效交易信号。**")
            lines.append("")
        
        lines.extend(["## 三、品种跟踪", ""])
        
        for analysis in self.results.get("analyses", [])[:15]:
            comm = analysis["commodity_name"]
            code = analysis["commodity_code"]
            
            has_policy = "✅" if analysis.get("policy_analysis") else "❌"
            has_fundamental = "✅" if analysis.get("fundamental_analysis") else "❌"
            has_chain = "✅" if analysis.get("chain_analysis") else "❌"
            
            lines.append(f"- **{comm} ({code})**: 政策{has_policy} 基本面{has_fundamental} 产业链{has_chain}")
        
        lines.append("")
        lines.append("## 四、深度分析")
        lines.append("")
        
        for analysis in self.results.get("analyses", []):
            code = analysis["commodity_code"]
            name = analysis["commodity_name"]
            
            lines.append(f"### {name} ({code})")
            
            fundamental = analysis.get("fundamental_analysis")
            if fundamental:
                lines.append("**基本面评估**:")
                lines.append(f"- 供应: {fundamental.get('supply_assessment', 'N/A')}")
                lines.append(f"- 需求: {fundamental.get('demand_assessment', 'N/A')}")
                lines.append(f"- 库存: {fundamental.get('inventory_assessment', 'N/A')}")
                lines.append(f"- 利润: {fundamental.get('profit_assessment', 'N/A')}")
                lines.append(f"- 基差: {fundamental.get('basis_assessment', 'N/A')}")
                lines.append(f"- 季节性: {fundamental.get('seasonal_factor', 'N/A')}")
                lines.append(f"- 评分: {fundamental.get('overall_score', 'N/A')} ({fundamental.get('direction_bias', 'N/A')})")
                if fundamental.get('key_risks'):
                    lines.append(f"- 风险: {', '.join(fundamental['key_risks'])}")
                lines.append("")
            
            chain = analysis.get("chain_analysis")
            if chain and chain.get("transmission_path"):
                lines.append("**产业链传导**:")
                for path in chain["transmission_path"]:
                    lines.append(f"- {path}")
                if chain.get("bottleneck"):
                    lines.append(f"- 瓶颈: {chain['bottleneck']}")
                lines.append(f"- 产业链得分: {chain.get('overall_score', 'N/A')}")
                lines.append("")
            
            policy = analysis.get("policy_analysis")
            if policy and policy.get("direct_impacts"):
                lines.append("**政策影响**:")
                for impact in policy["direct_impacts"]:
                    lines.append(f"- {impact.get('commodity', 'N/A')}: {impact.get('direction', 'N/A')} ({impact.get('strength', 'N/A')}) — {impact.get('mechanism', 'N/A')}")
                lines.append("")
        
        lines.append("## 五、原始数据摘要")
        lines.append("")
        
        for comm_code, data in self.results.get("fundamental", {}).items():
            indicators = data.get("indicators", {})
            if indicators:
                lines.append(f"**{comm_code} 搜索摘要**:")
                for indicator_name, indicator_data in indicators.items():
                    if hasattr(indicator_data, 'description') and indicator_data.description:
                        lines.append(f"- {indicator_name}: {indicator_data.description[:80]}...")
                lines.append("")
        
        lines.append("---")
        lines.append(f"*报告生成时间: {report_time}*")
        lines.append("*本报告由 AI 自动生成，仅供参考，不构成投资建议。*")
        
        report_content = "\n".join(lines)
        
        # 保存报告
        report_dir = PROJECT_ROOT / "reports" / "daily"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"daily_report_{datetime.now().strftime('%Y%m%d')}.md"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(f"日报已保存: {report_path}")
        self.results["report_path"] = str(report_path)
        self.results["report_content"] = report_content
    
    async def close(self):
        await self.policy_analyzer.close()
        await self.fundamental_analyzer.close()
        await self.chain_mapper.close()
        await self.signal_generator.close()
        await self.policy_collector.close()
        await self.fundamental_collector.close()


async def main():
    parser = argparse.ArgumentParser(description="每日期货投研")
    parser.add_argument(
        "--focus",
        nargs="+",
        help="聚焦特定品种，如 --focus RB I CU",
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="配置文件路径",
    )
    args = parser.parse_args()
    
    pipeline = DailyResearchPipeline(config_path=args.config)
    
    try:
        results = await pipeline.run(focused_commodities=args.focus)
        
        # 打印信号摘要
        signals = results.get("signals", {}).get("filtered", [])
        if signals:
            print("\n" + "=" * 60)
            print("今日交易信号")
            print("=" * 60)
            for sig in signals:
                print(f"\n[{sig.direction.value}] {sig.commodity_name} ({sig.commodity_code})")
                print(f"  置信度: {sig.confidence:.2f} | 持仓: {sig.holding_period_days}天 | 仓位: {sig.position_sizing}")
                print(f"  逻辑: {sig.core_logic}")
        else:
            print("\n今日无有效交易信号。")
        
        print(f"\n日报路径: {results.get('report_path', 'N/A')}")
        
    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
