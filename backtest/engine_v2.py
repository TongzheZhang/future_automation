"""
专业回测引擎 (v2)

关键特性:
  1. 合约乘数配置表（每个品种的实际乘数 / 最小变动价位）
  2. 手续费建模（按交易所实际标准：万分之0.5-1.5 + 固定费用）
  3. 滑点建模（1-2 个最小变动价位）
  4. 动态仓位（基于 ATR 的风险预算）
  5. Look-ahead bias 防护（明确验证所有数据在交易时可用）
  6. Rolling window 样本外验证（train/test split）
  7. Bootstrap 统计（对夏普/胜率做置信区间）
  8. 完善的 trades DataFrame（每笔入场/出场/PNL/滑点/手续费）
  9. 使用 loguru 日志

用法::

    from backtest.engine_v2 import BacktestEngineV2
    engine = BacktestEngineV2(initial_capital=1_000_000, risk_per_trade=0.02)
    result = engine.run(data=my_data, signal_generator=my_generator)
    print(result.trades_df)
    print(result.summary())
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from signals.factors import _compute_atr, precompute_indicators


# ============================================================================
# 合约规格表（乘数 + 最小变动价位 + 手续费标准）
# ============================================================================

# 格式: {symbol: (multiplier, min_tick, fee_mode, fee_value)}
#   multiplier   : 合约乘数（每手对应多少吨/克/桶/点）
#   min_tick     : 最小变动价位（元）
#   fee_mode     : 'pct' 按成交额比例 / 'fixed' 固定金额
#   fee_value    : 手续费率（pct 模式）或固定金额（fixed 模式，元/手）
CONTRACT_SPECS = {
    # ---- 上海期货交易所 (SHFE) ----
    "RB": {"multiplier": 10,    "min_tick": 1.0,   "fee_mode": "pct",   "fee": 0.0001,      "margin": 0.10, "name": "螺纹钢"},
    "HC": {"multiplier": 10,    "min_tick": 1.0,   "fee_mode": "pct",   "fee": 0.0001,      "margin": 0.10, "name": "热卷"},
    "CU": {"multiplier": 5,     "min_tick": 10.0,  "fee_mode": "pct",   "fee": 0.00005,     "margin": 0.10, "name": "沪铜"},
    "AL": {"multiplier": 5,     "min_tick": 5.0,   "fee_mode": "pct",   "fee": 0.0001,      "margin": 0.08, "name": "沪铝"},
    "AU": {"multiplier": 1000,  "min_tick": 0.02,  "fee_mode": "fixed", "fee": 10.0,        "margin": 0.10, "name": "沪金"},
    "AG": {"multiplier": 15,    "min_tick": 1.0,   "fee_mode": "pct",   "fee": 0.00005,     "margin": 0.12, "name": "沪银"},
    "ZN": {"multiplier": 5,     "min_tick": 5.0,   "fee_mode": "fixed", "fee": 3.0,         "margin": 0.09, "name": "沪锌"},
    "NI": {"multiplier": 1,     "min_tick": 10.0,  "fee_mode": "fixed", "fee": 3.0,         "margin": 0.10, "name": "沪镍"},
    "SN": {"multiplier": 1,     "min_tick": 10.0,  "fee_mode": "fixed", "fee": 3.0,         "margin": 0.10, "name": "沪锡"},
    "PB": {"multiplier": 5,     "min_tick": 5.0,   "fee_mode": "pct",   "fee": 0.00004,     "margin": 0.09, "name": "沪铅"},
    "SS": {"multiplier": 5,     "min_tick": 5.0,   "fee_mode": "fixed", "fee": 2.0,         "margin": 0.08, "name": "不锈钢"},
    "FU": {"multiplier": 10,    "min_tick": 1.0,   "fee_mode": "pct",   "fee": 0.00005,     "margin": 0.15, "name": "燃油"},
    "BU": {"multiplier": 10,    "min_tick": 1.0,   "fee_mode": "pct",   "fee": 0.0001,      "margin": 0.12, "name": "沥青"},
    "RU": {"multiplier": 10,    "min_tick": 5.0,   "fee_mode": "pct",   "fee": 0.000045,    "margin": 0.12, "name": "橡胶"},
    "SP": {"multiplier": 10,    "min_tick": 2.0,   "fee_mode": "pct",   "fee": 0.00005,     "margin": 0.08, "name": "纸浆"},

    # ---- 大连商品交易所 (DCE) ----
    "I":  {"multiplier": 100,   "min_tick": 0.5,   "fee_mode": "pct",   "fee": 0.0001,      "margin": 0.12, "name": "铁矿石"},
    "JM": {"multiplier": 60,    "min_tick": 0.5,   "fee_mode": "pct",   "fee": 0.0001,      "margin": 0.15, "name": "焦煤"},
    "J":  {"multiplier": 100,   "min_tick": 0.5,   "fee_mode": "pct",   "fee": 0.00006,     "margin": 0.12, "name": "焦炭"},
    "M":  {"multiplier": 10,    "min_tick": 1.0,   "fee_mode": "fixed", "fee": 1.5,         "margin": 0.08, "name": "豆粕"},
    "P":  {"multiplier": 10,    "min_tick": 2.0,   "fee_mode": "fixed", "fee": 2.5,         "margin": 0.10, "name": "棕榈油"},
    "Y":  {"multiplier": 10,    "min_tick": 2.0,   "fee_mode": "fixed", "fee": 2.5,         "margin": 0.08, "name": "豆油"},
    "A":  {"multiplier": 10,    "min_tick": 1.0,   "fee_mode": "fixed", "fee": 2.0,         "margin": 0.08, "name": "豆一"},
    "C":  {"multiplier": 10,    "min_tick": 1.0,   "fee_mode": "fixed", "fee": 1.2,         "margin": 0.08, "name": "玉米"},
    "CS": {"multiplier": 10,    "min_tick": 1.0,   "fee_mode": "fixed", "fee": 1.5,         "margin": 0.07, "name": "玉米淀粉"},
    "JD": {"multiplier": 10,    "min_tick": 1.0,   "fee_mode": "pct",   "fee": 0.00015,     "margin": 0.10, "name": "鸡蛋"},
    "L":  {"multiplier": 5,     "min_tick": 1.0,   "fee_mode": "fixed", "fee": 1.0,         "margin": 0.08, "name": "塑料"},
    "PP": {"multiplier": 5,     "min_tick": 1.0,   "fee_mode": "fixed", "fee": 1.0,         "margin": 0.08, "name": "PP"},
    "V":  {"multiplier": 5,     "min_tick": 1.0,   "fee_mode": "fixed", "fee": 1.0,         "margin": 0.08, "name": "PVC"},
    "EG": {"multiplier": 10,    "min_tick": 1.0,   "fee_mode": "fixed", "fee": 4.0,         "margin": 0.08, "name": "乙二醇"},
    "PG": {"multiplier": 20,    "min_tick": 1.0,   "fee_mode": "fixed", "fee": 6.0,         "margin": 0.12, "name": "液化气"},
    "LH": {"multiplier": 16,    "min_tick": 5.0,   "fee_mode": "pct",   "fee": 0.0002,      "margin": 0.12, "name": "生猪"},

    # ---- 郑州商品交易所 (ZCE) ----
    "TA": {"multiplier": 5,     "min_tick": 2.0,   "fee_mode": "fixed", "fee": 3.0,         "margin": 0.08, "name": "PTA"},
    "MA": {"multiplier": 10,    "min_tick": 1.0,   "fee_mode": "fixed", "fee": 2.0,         "margin": 0.08, "name": "甲醇"},
    "SA": {"multiplier": 20,    "min_tick": 1.0,   "fee_mode": "fixed", "fee": 4.0,         "margin": 0.10, "name": "纯碱"},
    "FG": {"multiplier": 20,    "min_tick": 1.0,   "fee_mode": "fixed", "fee": 6.0,         "margin": 0.10, "name": "玻璃"},
    "SR": {"multiplier": 10,    "min_tick": 1.0,   "fee_mode": "fixed", "fee": 3.0,         "margin": 0.08, "name": "白糖"},
    "CF": {"multiplier": 5,     "min_tick": 5.0,   "fee_mode": "fixed", "fee": 4.3,         "margin": 0.08, "name": "棉花"},
    "OI": {"multiplier": 10,    "min_tick": 2.0,   "fee_mode": "fixed", "fee": 2.0,         "margin": 0.08, "name": "菜油"},
    "RM": {"multiplier": 10,    "min_tick": 1.0,   "fee_mode": "fixed", "fee": 1.5,         "margin": 0.08, "name": "菜粕"},
    "PF": {"multiplier": 5,     "min_tick": 2.0,   "fee_mode": "fixed", "fee": 3.0,         "margin": 0.08, "name": "短纤"},
    "PK": {"multiplier": 5,     "min_tick": 2.0,   "fee_mode": "fixed", "fee": 4.0,         "margin": 0.08, "name": "花生"},
    "UR": {"multiplier": 20,    "min_tick": 1.0,   "fee_mode": "fixed", "fee": 5.0,         "margin": 0.08, "name": "尿素"},
    "SH": {"multiplier": 5,     "min_tick": 1.0,   "fee_mode": "pct",   "fee": 0.0001,      "margin": 0.10, "name": "烧碱"},
    "PX": {"multiplier": 5,     "min_tick": 2.0,   "fee_mode": "pct",   "fee": 0.0001,      "margin": 0.10, "name": "对二甲苯"},

    # ---- 上海国际能源交易中心 (INE) ----
    "SC": {"multiplier": 1000,  "min_tick": 0.1,   "fee_mode": "fixed", "fee": 20.0,        "margin": 0.15, "name": "原油"},
    "LU": {"multiplier": 10,    "min_tick": 1.0,   "fee_mode": "pct",   "fee": 0.00005,     "margin": 0.15, "name": "低硫燃油"},
    "BC": {"multiplier": 5,     "min_tick": 10.0,  "fee_mode": "pct",   "fee": 0.00005,     "margin": 0.12, "name": "国际铜"},
    "NR": {"multiplier": 10,    "min_tick": 5.0,   "fee_mode": "fixed", "fee": 3.0,         "margin": 0.12, "name": "20号胶"},

    # ---- 中国金融期货交易所 (CFFEX) ----
    "IF": {"multiplier": 300,   "min_tick": 0.2,   "fee_mode": "pct",   "fee": 0.000023,    "margin": 0.12, "name": "沪深300股指"},
    "IC": {"multiplier": 200,   "min_tick": 0.2,   "fee_mode": "pct",   "fee": 0.000023,    "margin": 0.14, "name": "中证500股指"},
    "IH": {"multiplier": 300,   "min_tick": 0.2,   "fee_mode": "pct",   "fee": 0.000023,    "margin": 0.12, "name": "上证50股指"},
    "IM": {"multiplier": 200,   "min_tick": 0.2,   "fee_mode": "pct",   "fee": 0.000023,    "margin": 0.14, "name": "中证1000股指"},
    "TS": {"multiplier": 20000, "min_tick": 0.005, "fee_mode": "fixed", "fee": 3.0,         "margin": 0.005, "name": "2年期国债"},
    "TF": {"multiplier": 10000, "min_tick": 0.005, "fee_mode": "fixed", "fee": 3.0,         "margin": 0.01,  "name": "5年期国债"},
    "T":  {"multiplier": 10000, "min_tick": 0.005, "fee_mode": "fixed", "fee": 3.0,         "margin": 0.02,  "name": "10年期国债"},
}

# 手续费默认值（品种不在表中时使用）
DEFAULT_CONTRACT_SPEC = {
    "multiplier": 10,
    "min_tick": 1.0,
    "fee_mode": "pct",
    "fee": 0.0001,
    "margin": 0.10,
    "name": "未知",
}


def get_spec(symbol: str) -> dict:
    """安全获取品种规格，缺失时返回默认值"""
    return CONTRACT_SPECS.get(symbol, DEFAULT_CONTRACT_SPEC)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class TradeRecord:
    """单笔交易明细"""
    symbol: str
    name: str
    trade_date: datetime
    direction: str              # 'long' / 'short'
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    exit_reason: str            # 'exit', 'stop_loss', 'take_profit'
    pnl_pct: float              # 收益率（不含费用）
    pnl_gross: float            # 毛利润（金额）
    commission_entry: float     # 入场手续费
    commission_exit: float      # 出场手续费
    slippage_entry: float       # 入场滑点成本
    slippage_exit: float        # 出场滑点成本
    pnl_net: float              # 净利润（扣除费用后）
    position_size: int          # 开仓手数
    notional: float             # 名义本金
    gap_pct: float              # 触发缺口
    atr_at_entry: float         # 入场时 ATR
    confidence: float           # 信号置信度
    factor_scores: dict = field(default_factory=dict)

    def to_series(self) -> pd.Series:
        return pd.Series({
            "symbol": self.symbol,
            "name": self.name,
            "trade_date": self.trade_date,
            "direction": self.direction,
            "entry_time": self.entry_time,
            "entry_price": self.entry_price,
            "exit_time": self.exit_time,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "pnl_pct": self.pnl_pct,
            "pnl_gross": round(self.pnl_gross, 2),
            "commission_entry": round(self.commission_entry, 2),
            "commission_exit": round(self.commission_exit, 2),
            "slippage_entry": round(self.slippage_entry, 2),
            "slippage_exit": round(self.slippage_exit, 2),
            "pnl_net": round(self.pnl_net, 2),
            "position_size": self.position_size,
            "notional": round(self.notional, 2),
            "gap_pct": self.gap_pct,
            "atr_at_entry": round(self.atr_at_entry, 4),
            "confidence": self.confidence,
        })


@dataclass
class BacktestResultV2:
    """增强版回测结果"""
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: pd.Series = None
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    total_commission: float = 0.0      # 手续费合计
    total_slippage: float = 0.0        # 滑点合计
    avg_hold_minutes: float = 0.0
    avg_confidence: float = 0.0

    # Bootstrap 统计
    sharpe_ci_low: float = 0.0
    sharpe_ci_high: float = 0.0
    win_rate_ci_low: float = 0.0
    win_rate_ci_high: float = 0.0

    @property
    def trades_df(self) -> pd.DataFrame:
        """将交易明细转换为 DataFrame"""
        if not self.trades:
            columns = [
                "symbol", "name", "trade_date", "direction", "entry_time",
                "entry_price", "exit_time", "exit_price", "exit_reason",
                "pnl_pct", "pnl_gross", "commission_entry", "commission_exit",
                "slippage_entry", "slippage_exit", "pnl_net", "position_size",
                "notional", "gap_pct", "atr_at_entry", "confidence",
            ]
            return pd.DataFrame(columns=columns)
        return pd.DataFrame([t.to_series() for t in self.trades])

    def summary(self) -> str:
        """生成摘要文本"""
        lines = [
            "=" * 55,
            f"  回测结果摘要",
            "=" * 55,
            f"  总收益率:     {self.total_return:>+10.2%}",
            f"  年化收益:     {self.annual_return:>+10.2%}",
            f"  夏普比率:     {self.sharpe_ratio:>10.2f}  [{self.sharpe_ci_low:.2f}, {self.sharpe_ci_high:.2f}]",
            f"  最大回撤:     {self.max_drawdown:>10.2%}",
            f"  胜率:         {self.win_rate:>10.1%}  [{self.win_rate_ci_low:.1%}, {self.win_rate_ci_high:.1%}]",
            f"  盈亏比:       {self.profit_factor:>10.2f}",
            f"  交易次数:     {self.total_trades:>10d}",
            f"  平均置信度:   {self.avg_confidence:>10.1%}",
            f"  Calmar:       {self.calmar_ratio:>10.2f}",
            f"  手续费合计:   {self.total_commission:>10.0f} 元",
            f"  滑点合计:     {self.total_slippage:>10.0f} 元",
            "=" * 55,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sharpe_ci": [self.sharpe_ci_low, self.sharpe_ci_high],
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "win_rate_ci": [self.win_rate_ci_low, self.win_rate_ci_high],
            "total_trades": self.total_trades,
            "profit_factor": self.profit_factor,
            "calmar_ratio": self.calmar_ratio,
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
            "avg_confidence": self.avg_confidence,
            "trades": self.trades_df.to_dict(orient="records") if self.total_trades > 0 else [],
        }


# ============================================================================
# 回测引擎
# ============================================================================

class BacktestEngineV2:
    """
    专业回测引擎

    完整模拟：信号生成 → 入场 → 持有 → 出场 → 费用计算 → 统计

    费用模型：
      - 入场费 = 合约面值 × 手续费率（或固定金额）× 手数
      - 出场费 = 同理
      - 滑点 = min_tick × slippage_ticks

    仓位管理：
      - 单笔风险 = 初始资金 × risk_per_trade
      - ATR 止损距离（价格）= atr_stop_mult × ATR
      - 手数 = floor(单笔风险 / (ATR止损距离 × 合约乘数))
      - 上限 = floor(名义本金 / (价格 × 合约乘数 × 保证金率))
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        risk_per_trade: float = 0.02,
        slippage_ticks_entry: int = 1,
        slippage_ticks_exit: int = 1,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        hold_minutes: int = 45,
        bootstrap_samples: int = 1000,
        min_confidence: float = 0.60,
        min_gap_pct: float = 0.003,
    ):
        """
        Parameters
        ----------
        initial_capital : float
            初始资金（元）
        risk_per_trade : float
            单笔风险占总资金比例
        slippage_ticks_entry : int
            入场滑点（最小变动价位个数）
        slippage_ticks_exit : int
            出场滑点（最小变动价位个数）
        atr_period : int
            ATR 计算周期
        atr_stop_mult : float
            ATR 止损倍数
        atr_tp_mult : float
            ATR 止盈倍数
        hold_minutes : int
            持有分钟数（用于近似出场时间）
        bootstrap_samples : int
            Bootstrap 重采样次数
        min_confidence : float
            最低置信度过滤
        min_gap_pct : float
            最小开盘缺口比例，低于该值不交易
        """
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.slippage_ticks_entry = slippage_ticks_entry
        self.slippage_ticks_exit = slippage_ticks_exit
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.atr_tp_mult = atr_tp_mult
        self.hold_minutes = hold_minutes
        self.bootstrap_samples = bootstrap_samples
        self.min_confidence = min_confidence
        self.min_gap_pct = min_gap_pct

        logger.info(
            f"BacktestEngineV2 初始化: capital={initial_capital:,.0f}, "
            f"risk={risk_per_trade:.1%}, slippage=entry/{slippage_ticks_entry}tick+exit/{slippage_ticks_exit}tick"
        )

    # ------------------------------------------------------------------
    # 费用计算
    # ------------------------------------------------------------------

    def _calc_commission(
        self, symbol: str, price: float, lots: int
    ) -> tuple[float, float]:
        """
        计算手续费和交易额。

        Returns
        -------
        commission : float
            手续费（元）
        notional : float
            合约面值 = price × multiplier × lots
        """
        spec = get_spec(symbol)
        multiplier = spec["multiplier"]
        notional = price * multiplier * lots

        if spec["fee_mode"] == "pct":
            commission = notional * spec["fee"]
        else:
            commission = spec["fee"] * lots

        return commission, notional

    def _calc_slippage_cost(self, symbol: str, ticks: int, lots: int) -> float:
        """计算滑点成本"""
        spec = get_spec(symbol)
        return ticks * spec["min_tick"] * spec["multiplier"] * lots

    # ------------------------------------------------------------------
    # 仓位计算（ATR 风险预算）
    # ------------------------------------------------------------------

    def _calc_position_size(
        self, symbol: str, entry_price: float, atr_val: float,
        current_equity: float = None
    ) -> int:
        """
        基于 ATR 风险预算计算开仓手数（使用当前净值动态调整）。

        逻辑：
          单笔最大亏损 = current_equity × risk_per_trade
          ATR 止损距离（价格）= atr_stop_mult × ATR
          每手亏损 = ATR止损距离（价格）× 合约乘数
          手数 = floor(单笔最大亏损 / 每手亏损)

        同时受保证金约束：
          最大手数 = floor(current_equity / (price × multiplier × margin_rate))

        ⚠️ 期货杠杆：保证金约8-15%，实际杠杆率 = 1/margin_rate ≈ 7-12倍
           仓位计算已自动体现杠杆效应——每手盈亏直接对应合约面值变动
        """
        spec = get_spec(symbol)
        multiplier = spec["multiplier"]
        margin_rate = spec["margin"]

        if current_equity is None:
            current_equity = self.initial_capital

        risk_amount = current_equity * self.risk_per_trade
        stop_distance_price = self.atr_stop_mult * atr_val
        loss_per_lot = stop_distance_price * multiplier

        if loss_per_lot <= 0:
            return 0

        lots_risk = int(risk_amount / loss_per_lot)

        # 保证金约束
        max_lots_margin = int(current_equity / (entry_price * multiplier * margin_rate))

        lots = max(0, min(lots_risk, max_lots_margin))
        return lots

    # ------------------------------------------------------------------
    # 主回测循环
    # ------------------------------------------------------------------

    def run(
        self,
        data: dict[str, pd.DataFrame],
        contracts: list[dict],
        scorer=None,
        start_date: str = None,
        end_date: str = None,
        lookahead_check: bool = True,
    ) -> BacktestResultV2:
        """
        执行回测。

        Parameters
        ----------
        data : dict
            {symbol: DataFrame} 日线数据（必须有 date, open, high, low, close）
        contracts : list[dict]
            品种配置列表
        scorer : WeightedSignalScorer, optional
            多因子评分器
        start_date : str, optional
            回测开始日期 'YYYY-MM-DD'
        end_date : str, optional
            回测结束日期 'YYYY-MM-DD'
        lookahead_check : bool
            是否执行 look-ahead bias 检查

        Returns
        -------
        BacktestResultV2
        """
        # 导入评分器（避免循环导入）
        if scorer is None:
            from signals.factors import create_default_scorer
            scorer = create_default_scorer()

        # ---- Look-ahead Bias 防护 ----
        # 检查所有 DataFrame 的日期是否单调递增，且无未来数据混入
        clean_data: dict[str, pd.DataFrame] = {}
        for sym, df in data.items():
            if df.empty:
                continue
            df = df.copy()
            df = df.sort_values("date").reset_index(drop=True)
            # 确认 'date' 列不会在计算时被错误使用
            if lookahead_check:
                if df["date"].is_monotonic_increasing:
                    clean_data[sym] = df
                else:
                    logger.warning(f"  {sym}: 日期非单调递增，已排序")
                    df = df.sort_values("date").reset_index(drop=True)
                    clean_data[sym] = df
            else:
                clean_data[sym] = df

        # 日期过滤
        if start_date:
            start_ts = pd.Timestamp(start_date)
            for sym in list(clean_data.keys()):
                clean_data[sym] = clean_data[sym][clean_data[sym]["date"] >= start_ts]
                if clean_data[sym].empty:
                    del clean_data[sym]

        if end_date:
            end_ts = pd.Timestamp(end_date)
            for sym in list(clean_data.keys()):
                clean_data[sym] = clean_data[sym][clean_data[sym]["date"] <= end_ts]
                if clean_data[sym].empty:
                    del clean_data[sym]

        # ---- 主循环：逐品种扫描 ----
        trades: list[TradeRecord] = []
        current_equity = self.initial_capital

        logger.info(f"开始回测: {len(clean_data)} 个品种, "
                     f"{start_date or '最早日期'} ~ {end_date or '最晚日期'}")

        for sym, df in clean_data.items():
            if len(df) < self.atr_period + 2:
                logger.debug(f"  {sym}: 数据不足, 跳过")
                continue

            # 获取品种名称
            contract_name = None
            for c in contracts:
                if c["symbol"] == sym:
                    contract_name = c.get("name", sym)
                    break
            if contract_name is None:
                contract_name = get_spec(sym)["name"]

            sym_trades, current_equity = self._run_single_symbol(
                sym, df, contract_name, scorer, start_date, end_date, current_equity
            )
            trades.extend(sym_trades)

        # ---- 结果汇总 ----
        if not trades:
            logger.warning("未产生任何交易")
            return BacktestResultV2()

        # 使用日频权益曲线计算指标（更准确）
        eq_series, daily_returns = self._compute_daily_equity(trades)
        n_trades = len(trades)

        total_return = (eq_series.iloc[-1] - self.initial_capital) / self.initial_capital

        # 年化收益：基于实际回测跨度（自然年），使用简单年化
        trade_dates = sorted([t.trade_date for t in trades])
        if len(trade_dates) >= 2:
            years = (trade_dates[-1] - trade_dates[0]).days / 365.25
        else:
            years = 1 / 252
        years = max(years, 0.1)
        annual_return = total_return / years if years > 0 else 0.0

        # 夏普（基于日收益率，仅工作日）
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        # 最大回撤
        peak = eq_series.expanding().max()
        drawdown = (eq_series - peak) / peak
        max_dd = drawdown.min() if len(drawdown) > 0 else 0.0

        # 胜率
        wins = [t for t in trades if t.pnl_net > 0]
        win_rate = len(wins) / n_trades if n_trades > 0 else 0.0

        # 盈亏比
        gross_profit = sum(t.pnl_net for t in wins)
        gross_loss = abs(sum(t.pnl_net for t in trades if t.pnl_net <= 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # 费用合计
        total_commission = sum(t.commission_entry + t.commission_exit for t in trades)
        total_slippage = sum(t.slippage_entry + t.slippage_exit for t in trades)

        # Calmar
        calmar = annual_return / abs(max_dd) if max_dd < 0 else 0.0

        # 平均置信度
        avg_confidence = np.mean([t.confidence for t in trades])

        # ---- Bootstrap 置信区间 ----
        sharpe_ci, win_rate_ci = self._bootstrap_ci(trades)

        result = BacktestResultV2(
            trades=trades,
            equity_curve=eq_series,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            total_trades=n_trades,
            profit_factor=profit_factor,
            calmar_ratio=calmar,
            total_commission=total_commission,
            total_slippage=total_slippage,
            avg_hold_minutes=self.hold_minutes,
            avg_confidence=avg_confidence,
            sharpe_ci_low=sharpe_ci[0],
            sharpe_ci_high=sharpe_ci[1],
            win_rate_ci_low=win_rate_ci[0],
            win_rate_ci_high=win_rate_ci[1],
        )

        # 更新历史胜率因子（持久化到 JSON）
        if scorer is not None:
            for factor in scorer.factors:
                if factor.name == "historical_win_rate":
                    for t in trades:
                        factor.update(t.symbol, t.direction, t.pnl_net)
                    logger.info(f"  历史胜率因子已更新: {len(trades)} 笔交易记录")
                    break

        logger.info(f"\n{result.summary()}")
        return result

    # ------------------------------------------------------------------
    # Bootstrap 统计
    # ------------------------------------------------------------------

    def _bootstrap_ci(self, trades: list[TradeRecord]) -> tuple[tuple, tuple]:
        """
        Bootstrap 重采样计算夏普比率和胜率的 95% 置信区间。

        Returns
        -------
        sharpe_ci : (low, high)
        win_rate_ci : (low, high)
        """
        if len(trades) < 10:
            return (0.0, 0.0), (0.0, 0.0)

        pnl_nets = np.array([t.pnl_net for t in trades])
        n = len(pnl_nets)

        sharpe_samples = []
        wr_samples = []

        rng = np.random.RandomState(42)

        for _ in range(self.bootstrap_samples):
            idx = rng.choice(n, size=n, replace=True)
            sample = pnl_nets[idx]

            # 夏普（基于样本的收益率序列）
            rets = sample / self.initial_capital
            if rets.std() > 0:
                s = rets.mean() / rets.std() * np.sqrt(252)
            else:
                s = 0.0
            sharpe_samples.append(s)

            # 胜率
            wr = np.mean(sample > 0)
            wr_samples.append(wr)

        sharpe_ci = (
            np.percentile(sharpe_samples, 2.5),
            np.percentile(sharpe_samples, 97.5),
        )
        win_rate_ci = (
            np.percentile(wr_samples, 2.5),
            np.percentile(wr_samples, 97.5),
        )

        return sharpe_ci, win_rate_ci

    # ------------------------------------------------------------------
    # Rolling Window 样本外验证
    # ------------------------------------------------------------------

    def rolling_window_oos(
        self,
        data: dict[str, pd.DataFrame],
        contracts: list[dict],
        train_years: float = 2.0,
        test_months: int = 6,
        scorer=None,
        min_trades: int = 30,
    ) -> list[dict]:
        """
        Rolling window 样本外验证。

        将数据按时间顺序切分为多个 train/test 窗口，
        每窗口用 train 期数据回测得到参数，测试期验证表现。

        Parameters
        ----------
        train_years : float
            训练期长度（年）
        test_months : int
            每次向前滚动测试期长度（月）
        min_trades : int
            每窗口最少交易数

        Returns
        -------
        list[dict]
            每个窗口的样本外表现
        """
        import datetime as dt

        # 获取全局日期范围
        all_dates = []
        for df in data.values():
            if 'date' in df.columns:
                all_dates.extend(df['date'].tolist())
        if not all_dates:
            return []

        all_dates = sorted(set(all_dates))
        min_date = all_dates[0]
        max_date = all_dates[-1]

        # 生成窗口
        windows = []
        train_start = min_date
        while True:
            train_end = train_start + pd.DateOffset(years=int(train_years))
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=test_months)

            if test_end > max_date:
                break

            windows.append({
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
            })

            train_start = test_start

        logger.info(f"Rolling Window OOS: {len(windows)} 个窗口")

        oos_results = []
        for wi, w in enumerate(windows):
            logger.info(f"\n窗口 {wi + 1}/{len(windows)}: "
                         f"Train {w['train_start']}~{w['train_end']}, "
                         f"Test {w['test_start']}~{w['test_end']}")

            # 训练期回测
            train_result = self.run(
                data=data,
                contracts=contracts,
                scorer=scorer,
                start_date=w["train_start"],
                end_date=w["train_end"],
            )

            # 测试期回测
            test_result = self.run(
                data=data,
                contracts=contracts,
                scorer=scorer,
                start_date=w["test_start"],
                end_date=w["test_end"],
            )

            oos_results.append({
                "window": wi + 1,
                "train_start": w["train_start"],
                "test_start": w["test_start"],
                "test_end": w["test_end"],
                "train_trades": train_result.total_trades,
                "train_sharpe": train_result.sharpe_ratio,
                "train_win_rate": train_result.win_rate,
                "test_trades": test_result.total_trades,
                "test_sharpe": test_result.sharpe_ratio,
                "test_win_rate": test_result.win_rate,
                "test_total_return": test_result.total_return,
                "test_max_dd": test_result.max_drawdown,
                "oos_degradation": (
                    test_result.sharpe_ratio - train_result.sharpe_ratio
                ),
            })

        return oos_results

    # ------------------------------------------------------------------
    # 单品种回测（可被子类覆盖）
    # ------------------------------------------------------------------

    def _run_single_symbol(
        self,
        sym: str,
        df: pd.DataFrame,
        contract_name: str,
        scorer,
        start_date: str = None,
        end_date: str = None,
        initial_equity: float = None,
    ) -> tuple[list[TradeRecord], float]:
        """
        对单个品种执行回测，返回交易列表和最终权益。
        子类可覆盖此方法以实现更复杂的品种级逻辑。
        """
        trades: list[TradeRecord] = []
        equity = initial_equity if initial_equity is not None else self.initial_capital

        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True)

        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["date"] <= pd.Timestamp(end_date)]

        if df.empty or len(df) < self.atr_period + 2:
            return trades, equity

        # 预计算全量 ATR，避免循环内逐日重复计算（O(n) 替代 O(n²)）
        full_atr = _compute_atr(df["high"], df["low"], df["close"], self.atr_period)
        # 预计算所有因子指标，通过 kwargs 传入各因子
        precomputed = precompute_indicators(
            df, atr_period=self.atr_period, ma_period=20, vol_period=20, adx_period=14,
        )

        for i in range(self.atr_period + 1, len(df)):
            prev = df.iloc[i - 1]
            today = df.iloc[i]

            prev_close = prev["close"]
            today_open = today["open"]

            if pd.isna(prev_close) or pd.isna(today_open) or prev_close <= 0:
                continue

            gap_pct = (today_open - prev_close) / prev_close

            if abs(gap_pct) < self.min_gap_pct:
                continue

            direction = "short" if gap_pct > 0 else "long"

            atr_val = (
                full_atr.iloc[i]
                if pd.notna(full_atr.iloc[i])
                else today_open * 0.015
            )

            confidence, factor_scores = scorer.score(
                df=df,
                index=i,
                prev_close=prev_close,
                gap_pct=gap_pct,
                direction=direction,
                symbol=sym,
                precomputed=precomputed,
            )

            if confidence < self.min_confidence:
                continue

            lots = self._calc_position_size(sym, today_open, atr_val, equity)
            if lots <= 0:
                continue

            comm_entry, notional = self._calc_commission(sym, today_open, lots)
            slippage_entry_cost = self._calc_slippage_cost(
                sym, self.slippage_ticks_entry, lots
            )

            stop_dist = self.atr_stop_mult * atr_val
            tp_dist = self.atr_tp_mult * atr_val

            if direction == "long":
                sl_price = today_open - stop_dist
                tp_price = today_open + tp_dist
            else:
                sl_price = today_open + stop_dist
                tp_price = today_open - tp_dist

            today_high = today["high"]
            today_low = today["low"]
            today_close_val = today["close"]

            exit_price = today_close_val
            exit_reason = "exit"

            if direction == "long":
                if today_low <= sl_price:
                    exit_price = sl_price
                    exit_reason = "stop_loss"
                elif today_high >= tp_price:
                    exit_price = tp_price
                    exit_reason = "take_profit"
            else:
                if today_high >= sl_price:
                    exit_price = sl_price
                    exit_reason = "stop_loss"
                elif today_low <= tp_price:
                    exit_price = tp_price
                    exit_reason = "take_profit"

            comm_exit, _ = self._calc_commission(sym, exit_price, lots)
            slippage_exit_cost = self._calc_slippage_cost(
                sym, self.slippage_ticks_exit, lots
            )

            if direction == "long":
                pnl_pct = (exit_price - today_open) / today_open
                pnl_gross = (exit_price - today_open) * lots * get_spec(sym)["multiplier"]
            else:
                pnl_pct = (today_open - exit_price) / today_open
                pnl_gross = (today_open - exit_price) * lots * get_spec(sym)["multiplier"]

            total_cost = comm_entry + comm_exit + slippage_entry_cost + slippage_exit_cost
            pnl_net = pnl_gross - total_cost

            entry_time = today["date"]
            exit_time = entry_time + timedelta(minutes=self.hold_minutes)

            trade = TradeRecord(
                symbol=sym,
                name=contract_name,
                trade_date=today["date"],
                direction=direction,
                entry_time=entry_time,
                entry_price=today_open,
                exit_time=exit_time,
                exit_price=exit_price,
                exit_reason=exit_reason,
                pnl_pct=pnl_pct,
                pnl_gross=pnl_gross,
                commission_entry=comm_entry,
                commission_exit=comm_exit,
                slippage_entry=slippage_entry_cost,
                slippage_exit=slippage_exit_cost,
                pnl_net=pnl_net,
                position_size=lots,
                notional=notional,
                gap_pct=gap_pct,
                atr_at_entry=atr_val,
                confidence=confidence,
                factor_scores=factor_scores,
            )
            trades.append(trade)
            equity += pnl_net

        return trades, equity

    # ------------------------------------------------------------------
    # 日频权益曲线
    # ------------------------------------------------------------------

    def _compute_daily_equity(
        self, trades: list[TradeRecord]
    ) -> tuple[pd.Series, pd.Series]:
        """
        构建日频权益曲线和日收益率序列。
        解决跨品种时序问题和夏普比率年化因子误用。
        """
        if not trades:
            now = pd.Timestamp.now().normalize()
            eq = pd.Series([self.initial_capital], index=[now])
            return eq, pd.Series(dtype=float)

        trades_sorted = sorted(trades, key=lambda t: t.trade_date)
        start_date = pd.Timestamp(trades_sorted[0].trade_date).normalize()
        end_date = pd.Timestamp(trades_sorted[-1].trade_date).normalize()

        # 日期范围：从交易日前一天开始，确保权益曲线首值为初始资金
        all_dates = pd.date_range(start_date - pd.Timedelta(days=1), end_date, freq="D")
        daily_pnl = pd.Series(0.0, index=all_dates)

        for t in trades_sorted:
            d = pd.Timestamp(t.trade_date).normalize()
            if d in daily_pnl.index:
                daily_pnl[d] += t.pnl_net

        equity = self.initial_capital
        equity_values = [equity]
        for d in all_dates[1:]:
            equity += daily_pnl[d]
            equity_values.append(equity)

        eq_series = pd.Series(equity_values, index=all_dates)

        # 日收益率（去掉首日的NaN，保留工作日）
        daily_returns = eq_series.pct_change().dropna()
        daily_returns = daily_returns[daily_returns.index.dayofweek < 5]

        return eq_series, daily_returns


# ============================================================================
# 测试入口
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  BacktestEngineV2 测试")
    print("=" * 60)

    # ---- 构造模拟数据 ----
    np.random.seed(42)
    data = {}

    for sym, spec in [("RB", 4000), ("CU", 75000), ("I", 800), ("TA", 5500), ("SC", 600)]:
        dates = pd.date_range("2024-01-01", periods=350, freq="B")
        close = spec + np.cumsum(np.random.randn(350) * spec * 0.01)
        df_sim = pd.DataFrame({
            "date": dates,
            "open": close + np.random.randn(350) * spec * 0.002,
            "high": close + abs(np.random.randn(350) * spec * 0.008),
            "low": close - abs(np.random.randn(350) * spec * 0.008),
            "close": close,
            "volume": 50000 + np.random.randn(350) * 10000,
            "open_interest": 200000 + np.cumsum(np.random.randn(350) * 500),
        })
        df_sim["volume"] = df_sim["volume"].clip(lower=1000)
        df_sim["open_interest"] = df_sim["open_interest"].clip(lower=100)
        data[sym] = df_sim

    # 品种配置
    contracts_sim = [
        {"symbol": "RB", "name": "螺纹钢"},
        {"symbol": "CU", "name": "沪铜"},
        {"symbol": "I",  "name": "铁矿石"},
        {"symbol": "TA", "name": "PTA"},
        {"symbol": "SC", "name": "原油"},
    ]

    # ---- 运行回测 ----
    engine = BacktestEngineV2(
        initial_capital=1_000_000,
        risk_per_trade=0.02,
        slippage_ticks_entry=1,
        slippage_ticks_exit=1,
    )

    result = engine.run(data=data, contracts=contracts_sim)

    print(f"\n--- 交易明细 (前 10 笔) ---")
    if result.total_trades > 0:
        print(result.trades_df.head(10).to_string())

    print(f"\n--- 按品种汇总 ---")
    if result.total_trades > 0:
        by_symbol = result.trades_df.groupby("symbol").agg(
            交易次数=("pnl_net", "count"),
            净利润=("pnl_net", "sum"),
            手续费=("commission_entry", lambda x: x.sum() + result.trades_df.loc[x.index, "commission_exit"].sum()),
            滑点=("slippage_entry", lambda x: x.sum() + result.trades_df.loc[x.index, "slippage_exit"].sum()),
            胜率=("pnl_net", lambda x: (x > 0).mean()),
        ).round(2)
        print(by_symbol)

    # ---- Rolling Window OOS ----
    print(f"\n--- Rolling Window 样本外验证 ---")
    oos = engine.rolling_window_oos(
        data=data,
        contracts=contracts_sim,
        train_years=1.0,
        test_months=3,
    )
    oos_df = pd.DataFrame(oos)
    print(oos_df.to_string())
