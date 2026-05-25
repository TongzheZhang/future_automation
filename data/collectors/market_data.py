"""
期货行情数据采集器
- 新浪财经免费行情 API（无需认证）
- 支持国内期货主力连续合约
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

# 品种代码映射: 系统代码 -> 新浪 nf_ 代码
SINA_CODE_MAP = {
    "RB": "nf_RB0",
    "I": "nf_I0",
    "J": "nf_J0",
    "JM": "nf_JM0",
    "HC": "nf_HC0",
    "SC": "nf_SC0",
    "TA": "nf_TA0",
    "MA": "nf_MA0",
    "L": "nf_L0",
    "M": "nf_M0",
    "C": "nf_C0",
    "CF": "nf_CF0",
    "P": "nf_P0",
    "CU": "nf_CU0",
    "AL": "nf_AL0",
    "NI": "nf_NI0",
    "AU": "nf_AU0",
    "AG": "nf_AG0",
    "EG": "nf_EG0",
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
    bid: float              # 买一价
    ask: float              # 卖一价
    open_interest: float    # 持仓量
    volume: float           # 成交量
    date: str               # 日期 YYYY-MM-DD
    
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
        """解析新浪行情字符串"""
        # 格式: var hq_str_nf_RB0="名称,时间,开盘,最高,最低,最新,买价,卖价,结算,昨结,买一价,买一量,卖一量,持仓,成交量,...";
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
        
        try:
            return MarketSnapshot(
                commodity=commodity,
                name=parts[0].strip(),
                time=parts[1].strip(),
                open=float(parts[2]),
                high=float(parts[3]),
                low=float(parts[4]),
                last=float(parts[5]),
                prev_settle=float(parts[9]),
                bid=float(parts[6]),
                ask=float(parts[7]),
                open_interest=float(parts[13]),
                volume=float(parts[14]),
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
    
    def get_snapshots(self, commodities: List[str]) -> Dict[str, MarketSnapshot]:
        """批量获取行情快照"""
        results = {}
        for comm in commodities:
            snap = self.get_snapshot(comm)
            if snap:
                results[comm] = snap
        return results
    
    def get_overnight_info(self, commodity: str) -> Dict[str, Any]:
        """
        获取品种的隔夜外盘/关联市场信息
        通过搜索获取，返回简化摘要
        """
        # 这里只是一个接口定义，实际外盘信息通过 Brave Search 在 strategy.py 中获取
        return {"commodity": commodity, "note": "通过搜索获取隔夜外盘信息"}
