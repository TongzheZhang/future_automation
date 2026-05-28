"""
定时任务调度入口
- 09:05: 商品期货开盘扫描
- 09:35: 金融期货开盘扫描（中金所 09:30 开盘，等5分钟稳定）
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

from intraday.morning_scan import run_morning_scan, load_commodities
from intraday.close_position import run_close_position
from intraday.evening_review import run_evening_review
from scripts.generate_industry_research import run_generation
from scripts.weekly_alpha_pai_exploration import run_weekly_exploration
from research.industry_mapping import get_all_industries

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
DEFAULT_FOCUS = ["RB", "M", "CU", "I", "SC", "AL", "AU"]


def job_morning_scan():
    """09:05 商品期货开盘扫描"""
    logger.info("定时任务触发: 商品期货 09:05 开盘扫描")
    try:
        asyncio.run(run_morning_scan(focused=DEFAULT_FOCUS))
    except Exception as e:
        logger.error(f"商品期货扫描失败: {e}")


def job_cffex_scan():
    """09:35 金融期货开盘扫描（中金所 09:30 开盘，等5分钟稳定）"""
    logger.info("定时任务触发: 金融期货 09:35 开盘扫描")
    try:
        asyncio.run(run_morning_scan(
            exchange_filter="CFFEX",
            scan_label="09:35 金融期货扫描",
        ))
    except Exception as e:
        logger.error(f"金融期货扫描失败: {e}")


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


def job_industry_research_update():
    """每周一 08:00 更新行业一页纸"""
    logger.info("定时任务触发: 行业一页纸更新")
    try:
        industries = get_all_industries()
        run_generation(industries, dry_run=False, force=False)
    except Exception as e:
        logger.error(f"行业一页纸更新失败: {e}")


def job_weekly_exploration():
    """每周日 20:00 Alpha派 策略探索"""
    logger.info("定时任务触发: 每周 Alpha派 策略探索")
    try:
        asyncio.run(run_weekly_exploration())
    except Exception as e:
        logger.error(f"每周策略探索失败: {e}")


def setup_schedule():
    """配置定时任务"""
    # 09:05 商品期货扫描
    schedule.every().monday.at("09:05").do(job_morning_scan)
    schedule.every().tuesday.at("09:05").do(job_morning_scan)
    schedule.every().wednesday.at("09:05").do(job_morning_scan)
    schedule.every().thursday.at("09:05").do(job_morning_scan)
    schedule.every().friday.at("09:05").do(job_morning_scan)

    # 09:35 金融期货扫描（中金所 09:30 开盘，等5分钟稳定）
    schedule.every().monday.at("09:35").do(job_cffex_scan)
    schedule.every().tuesday.at("09:35").do(job_cffex_scan)
    schedule.every().wednesday.at("09:35").do(job_cffex_scan)
    schedule.every().thursday.at("09:35").do(job_cffex_scan)
    schedule.every().friday.at("09:35").do(job_cffex_scan)

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

    # 每周一 08:00 更新行业一页纸（只更新过期/缺失的）
    schedule.every().monday.at("08:00").do(job_industry_research_update)

    # 每周日 20:00 Alpha派 策略探索
    schedule.every().sunday.at("20:00").do(job_weekly_exploration)

    logger.info("定时任务已配置")
    logger.info("  08:00 行业一页纸更新 (每周一)")
    logger.info("  09:05 商品期货扫描 (周一至周五)")
    logger.info("  09:35 金融期货扫描 (周一至周五, 中金所09:30开盘)")
    logger.info("  14:55 平仓记录 (周一至周五)")
    logger.info("  19:00 复盘 (周一至周五)")
    logger.info("  20:00 每周 Alpha派 策略探索 (每周日)")


def run_scheduler():
    """持续运行调度器"""
    setup_schedule()
    logger.info("调度器开始运行...")

    while True:
        schedule.run_pending()
        time.sleep(10)  # 每10秒检查一次


if __name__ == "__main__":
    run_scheduler()