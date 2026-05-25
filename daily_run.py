"""
每日自动化运行脚本 (v3 生产信号入口)

由 OpenClaw Cron 调度执行
每天运行: 生成 v3 信号 → 发送飞书
"""
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from signals.feishu_sender import send_signal_report
from data.collectors.akshare_adapter import get_all_contracts_data
from main_v3 import V3_CONFIG, generate_signals, load_config
from signals.factors import create_v3_scorer


def daily_run():
    """每日主流程"""
    logger.info(f"\n{'='*50}")
    logger.info(f"  🚀 每日信号任务启动")
    logger.info(f"  🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*50}")
    
    # 1. 加载 v3 生产配置
    _, contracts = load_config()
    logger.info(
        f"  v3生产配置: conf≥{V3_CONFIG['min_confidence']:.0%}, "
        f"排除品种={sorted(V3_CONFIG['exclude_symbols'])}"
    )
    
    # 2. 加载数据并生成信号 (复用 main_v3 生产逻辑)
    logger.info("\n📡 生成交易信号...")
    data = get_all_contracts_data(contracts, use_cache=True)
    
    scorer = create_v3_scorer()
    gen, signals = generate_signals(data, contracts, scorer)
    
    if signals:
        logger.info(f"  发现 {len(signals)} 个信号:")
        for s in signals:
            logger.info(f"    {s.name}({s.symbol}): {s.direction}, 置信度 {s.confidence:.0%}")
    else:
        logger.info("  今日无符合条件信号")
    
    # 3. 格式化并发送飞书
    report = gen.format_report(signals)
    logger.info("\n📤 发送飞书...")
    success = send_signal_report(report)
    
    if success:
        logger.info("  ✅ 飞书推送成功")
    else:
        logger.error("  ❌ 飞书推送失败")
    
    logger.info(f"\n✅ 每日任务完成!")
    return signals


if __name__ == "__main__":
    daily_run()
