"""
趋势突破策略 (Trend Breakout)

逻辑: 价格突破N日高点 → 趋势确认 →次日开盘做多 → 日内持有
     做空同理: 跌破N日低点 → 做空

适用: 趋势性强的品种(原油、铁矿石)
持仓: 日内
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger

from strategies.base import BaseStrategy, Signal, StrategyResult


class TrendBreakoutStrategy(BaseStrategy):
    """趋势突破策略"""
    
    name = "trend_breakout"
    description = "趋势突破：价格突破N日通道后顺势交易，日内操作"
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.lookback = config.get('lookback', 20)              # 通道周期
        self.hold_minutes = config.get('hold_minutes', 480)     # 全天持有
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)
        self.take_profit_pct = config.get('take_profit_pct', 0.025)
        self.atr_mult = config.get('atr_mult', 2.0)             # ATR倍数确认突破
        
    def generate_signals(self) -> list[Signal]:
        signals = []
        
        for symbol, df in self.data.items():
            if df.empty or len(df) < self.lookback + 2:
                continue
            
            df = df.copy()
            df['high_n'] = df['high'].rolling(self.lookback).max()
            df['low_n'] = df['low'].rolling(self.lookback).min()
            df['atr'] = df.apply(
                lambda r: max(r['high']-r['low'], abs(r['high']-df['close'].shift(1).loc[r.name]), abs(r['low']-df['close'].shift(1).loc[r.name]))
                if r.name > 0 else r['high']-r['low'], axis=1
            ).rolling(self.lookback).mean()
            
            # 最新信号
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            entry_price = last['open']
            entry_date = last['date']
            
            # 做多: 上破N日高点
            if prev['close'] > prev['high_n'] and last['open'] > prev['high_n']:
                atr = last['atr'] if not pd.isna(last['atr']) else last['close'] * 0.02
                signal = Signal(
                    symbol=symbol,
                    direction='long',
                    entry_time=entry_date,
                    entry_price=round(entry_price, 2),
                    exit_time=(entry_date + timedelta(minutes=self.hold_minutes)
                              if isinstance(entry_date, datetime) else datetime.now()),
                    stop_loss=round(entry_price - atr * self.atr_mult, 2),
                    take_profit=round(entry_price + atr * self.atr_mult * 1.5, 2),
                    confidence=0.65,
                    reason=f"突破{self.lookback}日高点 {prev['close']} > {prev['high_n']:.0f}",
                    strategy_name=self.name,
                )
                signals.append(signal)
            
            # 做空: 下破N日低点
            elif prev['close'] < prev['low_n'] and last['open'] < prev['low_n']:
                atr = last['atr'] if not pd.isna(last['atr']) else last['close'] * 0.02
                signal = Signal(
                    symbol=symbol,
                    direction='short',
                    entry_time=entry_date,
                    entry_price=round(entry_price, 2),
                    exit_time=(entry_date + timedelta(minutes=self.hold_minutes)
                              if isinstance(entry_date, datetime) else datetime.now()),
                    stop_loss=round(entry_price + atr * self.atr_mult, 2),
                    take_profit=round(entry_price - atr * self.atr_mult * 1.5, 2),
                    confidence=0.65,
                    reason=f"跌破{self.lookback}日低点 {prev['close']} < {prev['low_n']:.0f}",
                    strategy_name=self.name,
                )
                signals.append(signal)
        
        return signals
    
