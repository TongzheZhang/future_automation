"""
策略基类 - 所有策略的父类
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import pandas as pd


@dataclass
class Signal:
    """交易信号"""
    symbol: str            # 品种代码
    direction: str         # 'long' or 'short'
    entry_time: datetime   # 入场时间
    entry_price: float     # 入场价
    exit_time: datetime    # 目标出场时间
    stop_loss: float       # 止损价
    take_profit: float     # 止盈价
    confidence: float      # 置信度 0-1
    reason: str            # 信号理由
    strategy_name: str     # 策略名称
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'direction': '做多' if self.direction == 'long' else '做空',
            'entry_time': self.entry_time.strftime('%H:%M'),
            'entry_price': round(self.entry_price, 2),
            'exit_time': self.exit_time.strftime('%H:%M'),
            'stop_loss': round(self.stop_loss, 2),
            'take_profit': round(self.take_profit, 2),
            'confidence': f"{self.confidence:.0%}",
            'reason': self.reason,
            'strategy': self.strategy_name,
        }


@dataclass
class StrategyResult:
    """策略回测结果"""
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    avg_hold_minutes: float = 0.0
    equity_curve: pd.Series = None
    
    def summary(self) -> str:
        lines = [
            f"  总收益率:   {self.total_return:>+8.2%}",
            f"  年化收益:   {self.annual_return:>+8.2%}",
            f"  夏普比率:   {self.sharpe_ratio:>8.2f}",
            f"  最大回撤:   {self.max_drawdown:>8.2%}",
            f"  胜率:       {self.win_rate:>8.1%}",
            f"  交易次数:   {self.total_trades:>8d}",
            f"  盈亏比:     {self.profit_factor:>8.2f}",
            f"  Calmar:     {self.calmar_ratio:>8.2f}",
            f"  均持仓:     {self.avg_hold_minutes:>8.0f}分钟",
        ]
        return '\n'.join(lines)


class BaseStrategy(ABC):
    """策略基类"""
    
    name: str = "base"
    description: str = ""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.data: dict = {}           # {symbol: DataFrame}
        self.signals: list = []
        
    def load_data(self, data: dict):
        """加载数据"""
        self.data = data
    
    @abstractmethod
    def generate_signals(self) -> list[Signal]:
        """生成交易信号"""
        pass
    
    def backtest(self) -> StrategyResult:
        """回测策略"""
        return StrategyResult()
    
    def filter_by_date(self, start_date: str = None, end_date: str = None) -> dict:
        """按日期过滤数据"""
        filtered = {}
        for symbol, df in self.data.items():
            d = df.copy()
            if 'date' in d.columns:
                if start_date:
                    d = d[d['date'] >= pd.Timestamp(start_date)]
                if end_date:
                    d = d[d['date'] <= pd.Timestamp(end_date)]
            filtered[symbol] = d
        return filtered