"""
09:05 开盘扫描脚本
- 获取开盘行情
- 搜索隔夜新闻
- LLM 判断日内方向
- 输出交易信号报告
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from intraday.strategy import IntradayStrategy
from intraday.record import save_signals
from intraday.models import IntradaySignal
from data.collectors.minute_data import MinuteDataCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            PROJECT_ROOT / "logs" / f"intraday_{datetime.now().strftime('%Y%m%d')}.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("morning_scan")


def load_commodities(config_path: str = "config/settings.yaml") -> List[Dict[str, str]]:
    """加载监控品种列表"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    commodities = []
    for group, items in cfg.get("monitored_commodities", {}).items():
        for item in items:
            commodities.append({"code": item["code"], "name": item["name"]})
    return commodities


async def run_morning_scan(
    focused: List[str] = None,
    config_path: str = "config/settings.yaml",
):
    """执行开盘扫描"""
    date_str = datetime.now().strftime("%Y-%m-%d")
    logger.info("=" * 60)
    logger.info(f"[{date_str}] 09:05 开盘扫描开始")
    logger.info(f"策略时间窗口: 09:05 开仓 → 14:55 平仓")
    logger.info("=" * 60)
    
    strategy = IntradayStrategy()
    
    try:
        # 加载品种
        all_commodities = load_commodities(config_path)
        if focused:
            commodities = [c for c in all_commodities if c["code"] in focused]
        else:
            # 默认只扫描流动性好的品种
            default_codes = {"RB", "M", "CU", "I", "SC", "AL", "AU"}
            commodities = [c for c in all_commodities if c["code"] in default_codes]
        
        logger.info(f"扫描品种: {[c['code'] for c in commodities]}")
        
        # 扫描
        signals = await strategy.scan_all(commodities)
        
        # 用 AKShare 分钟线校正精确入场价
        minute_collector = MinuteDataCollector()
        for sig in signals:
            if not sig.should_trade():
                continue
            precise_entry = minute_collector.get_entry_price(sig.commodity)
            open_price = minute_collector.get_open_price(sig.commodity)
            if precise_entry is not None:
                old_entry = sig.entry_price
                sig.entry_price = precise_entry
                if sig.market_snapshot:
                    sig.market_snapshot.last = precise_entry
                diff = round(precise_entry - old_entry, 2)
                open_diff = round(precise_entry - open_price, 2) if open_price is not None else None
                logger.info(
                    f"[{sig.commodity}] 09:05精确价校正: {old_entry} -> {precise_entry} "
                    f"(与信号差异:{diff:+}, 与09:00开盘差异:{open_diff:+})"
                )
            else:
                logger.warning(f"[{sig.commodity}] 无法获取09:05分钟线，保留信号原始入场价")
        
        # 保存信号
        all_signals = []
        # 先保存所有扫描结果（包括观望的）
        # 注意：strategy.scan_all 只返回 valid signals，这里重新扫描一遍保存全部
        # 简化处理：只保存 valid signals
        if signals:
            save_signals(date_str, signals)
        
        # 生成报告
        report_lines = [
            f"# 日内交易信号 ({date_str} 09:05)",
            "",
            f"**策略时间窗口**: 09:05 开仓 → 14:55 平仓",
            "",
            f"扫描品种数: {len(commodities)}",
            f"有效信号数: {len(signals)}",
            "",
        ]
        
        if signals:
            report_lines.append("## 推荐交易")
            report_lines.append("")
            for i, sig in enumerate(signals, 1):
                report_lines.append(f"### {i}. {sig.commodity_name} ({sig.commodity}) — {sig.direction.value}")
                report_lines.append(f"- **置信度**: {sig.confidence}/10")
                report_lines.append(f"- **建议入场**: {sig.entry_price}")
                report_lines.append(f"- **止损**: {sig.stop_loss_price}")
                report_lines.append(f"- **目标**: {sig.target_price}")
                report_lines.append(f"- **核心逻辑**: {sig.core_logic}")
                report_lines.append(f"- **跳空**: {sig.market_snapshot.gap_pct}%")
                report_lines.append(f"- **昨结**: {sig.market_snapshot.prev_settle} | **开盘**: {sig.market_snapshot.open} | **最新**: {sig.market_snapshot.last}")
                report_lines.append("")
        else:
            report_lines.append("**今日无高确定性日内交易机会，建议观望。**")
            report_lines.append("")
            # 输出观望品种的信息
            report_lines.append("## 品种扫描结果")
            report_lines.append("")
            for comm in commodities:
                report_lines.append(f"- **{comm['name']} ({comm['code']})**: 观望")
            report_lines.append("")
        
        report_lines.append("---")
        report_lines.append(f"*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        report_lines.append("*14:55 前必须平仓。本信号仅供参考，不构成投资建议。*")
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        report_dir = PROJECT_ROOT / "reports" / "intraday"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"signal_{date_str}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(f"报告已保存: {report_path}")
        
        # 打印到控制台
        print("\n" + "=" * 60)
        print(report_content)
        print("=" * 60)
        
        return signals
    
    finally:
        await strategy.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--focus", nargs="+", help="聚焦品种")
    args = parser.parse_args()
    
    asyncio.run(run_morning_scan(focused=args.focus))
