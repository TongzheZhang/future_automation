"""
期货分钟线数据采集器
- 基于 AKShare futures_zh_minute_sina 接口
- 提供纯日盘(09:05-14:55)的精确入场/出场价和高低点
- 解决新浪快照无法区分 09:00 开盘 vs 09:05 入场、以及 high/low 含夜盘的问题
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import akshare as ak
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# 默认 symbol 映射规则: 系统代码 -> AKShare symbol（直接加 "0"）
DEFAULT_AK_SUFFIX = "0"

# 策略时间窗口常量
ENTRY_TIME = "09:05:00"
EXIT_TIME = "14:55:00"
DAY_SESSION_START = "09:00:00"
DAY_SESSION_END = "15:00:00"


class MinuteDataCollector:
    """分钟线数据采集器"""

    def __init__(self, config_path: str = "config/settings.yaml"):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._symbol_map = self._load_symbol_map(config_path)

    def _load_symbol_map(self, config_path: str) -> Dict[str, str]:
        """从配置加载品种映射，无配置时默认加后缀 '0'"""
        path = Path(config_path)
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            return cfg.get("minute_data", {}).get("symbol_map", {})
        except Exception as e:
            logger.warning(f"加载 minute_data 配置失败: {e}")
            return {}

    def _to_ak_symbol(self, commodity: str) -> Optional[str]:
        """系统代码转 AKShare symbol"""
        ak_symbol = self._symbol_map.get(commodity)
        if ak_symbol:
            return ak_symbol
        return f"{commodity}{DEFAULT_AK_SUFFIX}"

    def _cache_key(self, ak_symbol: str) -> str:
        """生成缓存键（按自然日）"""
        return f"{ak_symbol}_{datetime.now().strftime('%Y-%m-%d')}"

    def _fetch_minute_data(
        self, ak_symbol: str, period: str = "1", max_retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """拉取分钟线数据，带重试机制"""
        for attempt in range(1, max_retries + 1):
            try:
                df = ak.futures_zh_minute_sina(symbol=ak_symbol, period=period)
                if df is None or df.empty:
                    logger.warning(f"AKShare 返回空数据 {ak_symbol} (attempt {attempt})")
                else:
                    return df
            except Exception as e:
                logger.warning(
                    f"拉取分钟线失败 {ak_symbol} (attempt {attempt}/{max_retries}): {e}"
                )
            if attempt < max_retries:
                import time

                time.sleep(0.5 * attempt)
        logger.error(f"拉取分钟线最终失败 {ak_symbol}")
        return None

    def _get_data(self, commodity: str) -> Optional[pd.DataFrame]:
        """获取品种分钟线（优先缓存，仅保留当天数据）"""
        ak_symbol = self._to_ak_symbol(commodity)
        if not ak_symbol:
            return None

        key = self._cache_key(ak_symbol)
        if key in self._cache:
            return self._cache[key]

        df = self._fetch_minute_data(ak_symbol)
        if df is not None and not df.empty:
            # AKShare 返回多交易日数据，仅保留当前自然日
            today = datetime.now().strftime("%Y-%m-%d")
            df = df[df["datetime"].astype(str).str.startswith(today)]
            if not df.empty:
                self._cache[key] = df
                return df
            logger.warning(f"{commodity} 分钟线无当日({today})数据")
            return None
        return None

    @staticmethod
    def _filter_by_time(
        df: pd.DataFrame, start_time: str, end_time: str
    ) -> pd.DataFrame:
        """
        按日盘时间过滤 DataFrame
        datetime 格式示例: '2026-05-25 09:05:00'
        start_time/end_time 格式: 'HH:MM:SS' 或 'HH:MM'
        """
        # 补齐为 HH:MM:SS 以保证字符串比较一致性
        if len(start_time) == 5:
            start_time += ":00"
        if len(end_time) == 5:
            end_time += ":00"
        time_series = df["datetime"].astype(str).str.slice(11, 19)
        return df[(time_series >= start_time) & (time_series <= end_time)]

    def get_entry_price(self, commodity: str) -> Optional[float]:
        """
        获取 09:05 分钟线的 open 作为精确入场价
        """
        df = self._get_data(commodity)
        if df is None or df.empty:
            return None

        filtered = self._filter_by_time(df, ENTRY_TIME[:5], ENTRY_TIME[:5])
        if filtered.empty:
            logger.warning(f"未找到 {commodity} 09:05 分钟线")
            return None

        # 取第一条匹配的 09:05 K线 open
        return float(filtered.iloc[0]["open"])

    def get_exit_price(self, commodity: str) -> Optional[float]:
        """
        获取 14:55 分钟线的 close 作为精确出场价
        """
        df = self._get_data(commodity)
        if df is None or df.empty:
            return None

        filtered = self._filter_by_time(df, EXIT_TIME[:5], EXIT_TIME[:5])
        if filtered.empty:
            logger.warning(f"未找到 {commodity} 14:55 分钟线")
            return None

        return float(filtered.iloc[0]["close"])

    def get_day_high_low(self, commodity: str) -> Tuple[Optional[float], Optional[float]]:
        """
        获取纯日盘(09:05-14:55)的 high/low
        返回 (high, low) 元组
        """
        df = self._get_data(commodity)
        if df is None or df.empty:
            return None, None

        # 先过滤到日盘范围 09:00-15:00，再精确到策略窗口 09:05-14:55
        day_df = self._filter_by_time(df, DAY_SESSION_START[:5], DAY_SESSION_END[:5])
        window_df = self._filter_by_time(day_df, ENTRY_TIME[:5], EXIT_TIME[:5])

        if window_df.empty:
            logger.warning(f"未找到 {commodity} 日盘策略窗口数据")
            return None, None

        return float(window_df["high"].max()), float(window_df["low"].min())

    def get_open_price(self, commodity: str) -> Optional[float]:
        """
        获取日盘第一根 K 线的 open 作为 09:00 开盘价参考
        """
        df = self._get_data(commodity)
        if df is None or df.empty:
            return None

        day_df = self._filter_by_time(df, DAY_SESSION_START[:5], DAY_SESSION_END[:5])
        if day_df.empty:
            logger.warning(f"未找到 {commodity} 日盘开盘数据")
            return None

        return float(day_df.iloc[0]["open"])

    # ------------------------------------------------------------------
    # 异步包装（AKShare 是同步 IO，用 asyncio.to_thread 避免阻塞事件循环）
    # ------------------------------------------------------------------

    async def async_get_entry_price(self, commodity: str) -> Optional[float]:
        return await asyncio.to_thread(self.get_entry_price, commodity)

    async def async_get_exit_price(self, commodity: str) -> Optional[float]:
        return await asyncio.to_thread(self.get_exit_price, commodity)

    async def async_get_day_high_low(
        self, commodity: str
    ) -> Tuple[Optional[float], Optional[float]]:
        return await asyncio.to_thread(self.get_day_high_low, commodity)

    async def async_get_open_price(self, commodity: str) -> Optional[float]:
        return await asyncio.to_thread(self.get_open_price, commodity)
