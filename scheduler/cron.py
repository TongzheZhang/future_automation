"""
定时任务调度入口
- 09:05: 开盘扫描
- 14:55: 平仓记录
- 19:00: 复盘

用法:
    python scheduler/cron.py
    # 后台持续运行
    nohup python scheduler/cron.py &
"""

import os
import sys
import time
import logging
import asyncio
from datetime import datetime
from pathlib import Path

import schedule

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from intraday.morning_scan import run_morning_scan
from intraday.close_position import run_close_position
from intraday.evening_review import run_evening_review

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            PROJECT_ROOT / "logs" / f"scheduler_{datetime.now().strftime('%Y%m%d')}.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("scheduler")


# 默认监控品种（可配置）
DEFAULT_FOCUS = ["RB", "M", "CU", "I"]


def job_morning_scan():
    """09:05 开盘扫描"""
    logger.info("定时任务触发: 开盘扫描")
    try:
        asyncio.run(run_morning_scan(focused=DEFAULT_FOCUS))
    except Exception as e:
        logger.error(f"开盘扫描失败: {e}")


def job_close_position():
    """14:55 平仓记录"""
    logger.info("定时任务触发: 平仓记录")
    try:
        asyncio.run(run_close_position())
    except Exception as e:
        logger.error(f"平仓记录失败: {e}")


def job_evening_review():
    """19:00 复盘"""
    logger.info("定时任务触发: 复盘")
    try:
        asyncio.run(run_evening_review())
    except Exception as e:
        logger.error(f"复盘失败: {e}")


def setup_schedule():
    """配置定时任务"""
    # 周一至周五运行（简化处理，不做节假日判断）
    schedule.every().monday.at("09:05").do(job_morning_scan)
    schedule.every().tuesday.at("09:05").do(job_morning_scan)
    schedule.every().wednesday.at("09:05").do(job_morning_scan)
    schedule.every().thursday.at("09:05").do(job_morning_scan)
    schedule.every().friday.at("09:05").do(job_morning_scan)
    
    schedule.every().monday.at("14:55").do(job_close_position)
    schedule.every().tuesday.at("14:55").do(job_close_position)
    schedule.every().wednesday.at("14:55").do(job_close_position)
    schedule.every().thursday.at("14:55").do(job_close_position)
    schedule.every().friday.at("14:55").do(job_close_position)
    
    schedule.every().monday.at("19:00").do(job_evening_review)
    schedule.every().tuesday.at("19:00").do(job_evening_review)
    schedule.every().wednesday.at("19:00").do(job_evening_review)
    schedule.every().thursday.at("19:00").do(job_evening_review)
    schedule.every().friday.at("19:00").do(job_evening_review)
    
    logger.info("定时任务已配置")
    logger.info("  09:05 开盘扫描 (周一至周五)")
    logger.info("  14:55 平仓记录 (周一至周五)")
    logger.info("  19:00 复盘 (周一至周五)")


def run_scheduler():
    """持续运行调度器"""
    setup_schedule()
    logger.info("调度器开始运行...")
    
    while True:
        schedule.run_pending()
        time.sleep(10)  # 每10秒检查一次


if __name__ == "__main__":
    run_scheduler()
