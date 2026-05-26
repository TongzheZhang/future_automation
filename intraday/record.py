"""
日内交易记录存储
- JSON 文件存储，便于查看和版本控制
- 支持按日期读写
"""

import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

from intraday.models import IntradaySignal, IntradayTrade, DailyReview

logger = logging.getLogger(__name__)

# 存储路径
PROJECT_ROOT = Path(__file__).parent.parent
RECORD_DIR = PROJECT_ROOT / "data" / "intraday_records"
RECORD_DIR.mkdir(parents=True, exist_ok=True)


def _signal_path(date: str) -> Path:
    return RECORD_DIR / f"signals_{date}.json"


def _trade_path(date: str) -> Path:
    return RECORD_DIR / f"trades_{date}.json"


def _review_path(date: str) -> Path:
    return RECORD_DIR / f"review_{date}.json"


def save_signals(date: str, signals: List[IntradaySignal]):
    """保存当日信号"""
    path = _signal_path(date)
    data = [s.model_dump(mode="json") for s in signals]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"信号已保存: {path}")


def load_signals(date: str) -> List[IntradaySignal]:
    """读取当日信号"""
    path = _signal_path(date)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [IntradaySignal(**item) for item in data]


def save_trades(date: str, trades: List[IntradayTrade]):
    """保存当日交易记录"""
    path = _trade_path(date)
    data = []
    for t in trades:
        d = t.model_dump(mode="json")
        d["entry_time"] = f"{t.date} 09:05:00"
        d["exit_time"] = t.closed_at.strftime("%Y-%m-%d %H:%M:%S") if t.closed_at else f"{t.date} 14:55:00"
        data.append(d)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"交易记录已保存: {path}")


def load_trades(date: str) -> List[IntradayTrade]:
    """读取当日交易记录"""
    path = _trade_path(date)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [IntradayTrade(**item) for item in data]


def save_review(review: DailyReview):
    """保存复盘"""
    path = _review_path(review.date)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(review.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
    logger.info(f"复盘已保存: {path}")


def load_review(date: str) -> Optional[DailyReview]:
    """读取复盘"""
    path = _review_path(date)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DailyReview(**data)


def get_all_trade_dates() -> List[str]:
    """获取所有有交易记录的日期"""
    dates = set()
    for path in RECORD_DIR.glob("trades_*.json"):
        date = path.stem.replace("trades_", "")
        dates.add(date)
    return sorted(dates)


def get_all_reviews() -> List[DailyReview]:
    """获取所有复盘"""
    reviews = []
    for path in sorted(RECORD_DIR.glob("review_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        reviews.append(DailyReview(**data))
    return reviews
