"""
期货行情数据采集器
- 新浪财经免费行情 API（无需认证）
- 支持国内期货主力连续合约
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import aiohttp
import requests

logger = logging.getLogger(__name__)

# 品种代码映射: 系统代码 -> 新浪 nf_ 代码
SINA_CODE_MAP = {
    "AG": "nf_AG0",
    "AL": "nf_AL0",
    "AO": "nf_AO0",
    "AP": "nf_AP0",
    "AU": "nf_AU0",
    "BC": "nf_BC0",
    "BR": "nf_BR0",
    "BU": "nf_BU0",
    "C": "nf_C0",
    "CF": "nf_CF0",
    "CJ": "nf_CJ0",
    "CS": "nf_CS0",
    "CU": "nf_CU0",
    "EB": "nf_EB0",
    "EG": "nf_EG0",
    "FG": "nf_FG0",
    "HC": "nf_HC0",
    "I": "nf_I0",
    "J": "nf_J0",
    "JD": "nf_JD0",
    "JM": "nf_JM0",
    "L": "nf_L0",
    "LC": "nf_LC0",
    "LH": "nf_LH0",
    "LU": "nf_LU0",
    "M": "nf_M0",
    "MA": "nf_MA0",
    "NI": "nf_NI0",
    "NR": "nf_NR0",
    "OI": "nf_OI0",
    "P": "nf_P0",
    "PB": "nf_PB0",
    "PF": "nf_PF0",
    "PG": "nf_PG0",
    "PK": "nf_PK0",
    "PP": "nf_PP0",
    "PX": "nf_PX0",
    "RB": "nf_RB0",
    "RM": "nf_RM0",
    "RU": "nf_RU0",
    "SA": "nf_SA0",
    "SC": "nf_SC0",
    "SF": "nf_SF0",
    "SH": "nf_SH0",
    "SI": "nf_SI0",
    "SM": "nf_SM0",
    "SN": "nf_SN0",
    "SP": "nf_SP0",
    "SR": "nf_SR0",
    "SS": "nf_SS0",
    "TA": "nf_TA0",
    "UR": "nf_UR0",
    "V": "nf_V0",
    "Y": "nf_Y0",
    "ZN": "nf_ZN0",
}


@dataclass
class MarketSnapshot:
    """行情快照"""
    commodity: str          # 系统代码如 RB
    name: str               # 品种名称
    time: str               # 时间 HHMMSS
    open: float
    high: float
    low: float
    last: float             # 最新价
    prev_settle: float      # 昨结
    settle: float = 0.0     # 当日结算价（盘中可能为0，盘后更新）
    bid: float = 0.0        # 买一价
    ask: float = 0.0        # 卖一价
    open_interest: float = 0.0   # 持仓量
    volume: float = 0.0     # 成交量
    date: str = ""          # 日期 YYYY-MM-DD
    
    @property
    def gap_pct(self) -> float:
        """跳空幅度"""
        if self.prev_settle == 0:
            return 0.0
        return round((self.open - self.prev_settle) / self.prev_settle * 100, 2)
    
    @property
    def change_pct(self) -> float:
        """涨跌幅（基于昨结）"""
        if self.prev_settle == 0:
            return 0.0
        return round((self.last - self.prev_settle) / self.prev_settle * 100, 2)
    
    @property
    def amplitude_pct(self) -> float:
        """振幅"""
        if self.prev_settle == 0:
            return 0.0
        return round((self.high - self.low) / self.prev_settle * 100, 2)


class MarketDataCollector:
    """行情数据采集器"""
    
    BASE_URL = "http://hq.sinajs.cn/list={code}"
    HEADERS = {"Referer": "https://finance.sina.com.cn"}
    
    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(self.HEADERS)
    
    def _fetch_raw(self, sina_code: str) -> Optional[str]:
        """获取原始行情字符串"""
        url = self.BASE_URL.format(code=sina_code)
        try:
            resp = self._session.get(url, timeout=10)
            resp.encoding = "gbk"
            if resp.status_code == 200:
                return resp.text
            else:
                logger.warning(f"行情请求失败 {sina_code}: {resp.status_code}")
                return None
        except Exception as e:
            logger.error(f"行情请求异常 {sina_code}: {e}")
            return None
    
    def _parse(self, commodity: str, raw: str) -> Optional[MarketSnapshot]:
        """解析新浪行情字符串
        
        新浪 nf_ 前缀期货行情字段布局 (实际测试验证):
            0: 品种名称
            1: 时间 (HHMMSS)
            2: 开盘价
            3: 最高价
            4: 最低价
            5: 最新价 (日内实时价)
            6: 买一价
            7: 卖一价
            8: 昨收 (上一交易日收盘价)
            9: 结算 (当日结算价, 盘后定价)
            10: 昨结 (上一交易日结算价, 日内关键参考)
            11: 买一量
            12: 卖一量
            13: 持仓量
            14: 成交量
            15: 交易所
            16: 品种名称(短)
            17: 日期 YYYY-MM-DD
            18+: 扩展字段
        
        关键：昨结在 parts[10]，不是 parts[9]（结算价）。
        日内交易应基于昨结计算涨跌幅和跳空，而非收盘价或当日结算价。
        """
        prefix = f'var hq_str_{SINA_CODE_MAP[commodity]}="'
        if prefix not in raw:
            logger.warning(f"无法解析行情 {commodity}: 格式不匹配")
            return None
        
        content = raw.split(prefix, 1)[1].split('"', 1)[0]
        if not content:
            logger.warning(f"行情数据为空 {commodity}")
            return None
        
        parts = content.split(",")
        if len(parts) < 19:
            logger.warning(f"行情数据字段不足 {commodity}: {len(parts)} fields")
            return None
        
        # 安全转换辅助函数，空值/非数字统一返回 0.0
        def _safe_float(val: str) -> float:
            try:
                return float(val) if val.strip() else 0.0
            except ValueError:
                return 0.0
        
        try:
            return MarketSnapshot(
                commodity=commodity,
                name=parts[0].strip(),
                time=parts[1].strip(),
                open=_safe_float(parts[2]),
                high=_safe_float(parts[3]),
                low=_safe_float(parts[4]),
                last=_safe_float(parts[5]),       # 日内最新价
                prev_settle=_safe_float(parts[10]),  # 昨结 (parts[10])
                settle=_safe_float(parts[9]),     # 当日结算价 (parts[9])
                bid=_safe_float(parts[6]),        # 买一价
                ask=_safe_float(parts[7]),        # 卖一价
                open_interest=_safe_float(parts[13]),
                volume=_safe_float(parts[14]),
                date=parts[17].strip(),
            )
        except (ValueError, IndexError) as e:
            logger.error(f"行情解析异常 {commodity}: {e}, raw={content[:100]}")
            return None
    
    def get_snapshot(self, commodity: str) -> Optional[MarketSnapshot]:
        """获取单个品种行情快照"""
        sina_code = SINA_CODE_MAP.get(commodity)
        if not sina_code:
            logger.warning(f"未找到品种映射 {commodity}")
            return None
        
        raw = self._fetch_raw(sina_code)
        if not raw:
            return None
        
        return self._parse(commodity, raw)
    
    async def _async_fetch_raw(self, sina_code: str, retries: int = 3) -> Optional[str]:
        """异步获取原始行情字符串，带指数退避重试"""
        url = self.BASE_URL.format(code=sina_code)
        for attempt in range(1, retries + 1):
            try:
                async with aiohttp.ClientSession(headers=self.HEADERS) as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            raw_bytes = await resp.read()
                            return raw_bytes.decode("gbk", errors="replace")
                        else:
                            logger.warning(f"行情请求失败 {sina_code}: {resp.status}")
            except Exception as e:
                logger.warning(f"行情请求异常 {sina_code} (attempt {attempt}/{retries}): {e}")
                if attempt < retries:
                    await asyncio.sleep(0.5 * attempt)
        return None

    async def async_get_snapshot(self, commodity: str) -> Optional[MarketSnapshot]:
        """异步获取单个品种行情快照"""
        sina_code = SINA_CODE_MAP.get(commodity)
        if not sina_code:
            logger.warning(f"未找到品种映射 {commodity}")
            return None
        raw = await self._async_fetch_raw(sina_code)
        if not raw:
            return None
        return self._parse(commodity, raw)

    async def async_get_snapshots(self, commodities: List[str]) -> Dict[str, MarketSnapshot]:
        """批量异步并发获取行情快照，带重试"""
        tasks = [self.async_get_snapshot(comm) for comm in commodities]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        snapshots: Dict[str, MarketSnapshot] = {}
        for comm, result in zip(commodities, results):
            if isinstance(result, Exception):
                logger.error(f"批量行情获取异常 {comm}: {result}")
            elif result is not None:
                snapshots[comm] = result
        return snapshots

    def get_snapshots(self, commodities: List[str]) -> Dict[str, MarketSnapshot]:
        """批量获取行情快照（同步兼容层）"""
        results = {}
        for comm in commodities:
            snap = self.get_snapshot(comm)
            if snap:
                results[comm] = snap
        return results
    
    def get_minute_snapshot(self, commodity: str) -> Optional[MarketSnapshot]:
        """
        通过 AKShare 分钟线构造行情快照（便捷兼容层）。
        优先返回 14:55 分钟线的 close 作为 last，纯日盘 high/low。
        若分钟线不可用，则回退到新浪快照。
        """
        from data.collectors.minute_data import MinuteDataCollector

        minute = MinuteDataCollector()
        exit_price = minute.get_exit_price(commodity)
        day_high, day_low = minute.get_day_high_low(commodity)
        open_price = minute.get_open_price(commodity)

        if exit_price is not None:
            return MarketSnapshot(
                commodity=commodity,
                name="",
                time="145500",
                open=open_price if open_price is not None else 0.0,
                high=day_high if day_high is not None else 0.0,
                low=day_low if day_low is not None else 0.0,
                last=exit_price,
                prev_settle=0.0,  # 分钟线不包含昨结，需调用方另行获取
                settle=0.0,
                bid=0.0,
                ask=0.0,
                open_interest=0.0,
                volume=0.0,
                date=datetime.now().strftime("%Y-%m-%d"),
            )

        # Fallback：新浪快照
        logger.info(f"分钟线不可用，回退到新浪快照 {commodity}")
        return self.get_snapshot(commodity)

    async def async_get_minute_snapshot(self, commodity: str) -> Optional[MarketSnapshot]:
        """异步版 get_minute_snapshot"""
        import asyncio
        return await asyncio.to_thread(self.get_minute_snapshot, commodity)

    def get_overnight_info(self, commodity: str) -> Dict[str, Any]:
        """
        获取品种的隔夜外盘/关联市场信息
        通过搜索获取，返回简化摘要
        """
        # 这里只是一个接口定义，实际外盘信息通过 Brave Search 在 strategy.py 中获取
        return {"commodity": commodity, "note": "通过搜索获取隔夜外盘信息"}
