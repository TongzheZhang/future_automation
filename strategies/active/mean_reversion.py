"""
均值回归策略 (Mean Reversion)

逻辑: 价格偏离均线太远 → 预期回归 → 反向交易
     用布林带和RSI双重确认

适用: 震荡市场、农产品
持仓: 日内
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger

from strategies.base import BaseStrategy, Signal, StrategyResult


class MeanReversionStrategy(BaseStrategy):
    """均值回归策略"""
    
    name = "mean_reversion"
    description = "均值回归：价格偏离均线太远时反向交易，布林带+RSI双确认"
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.ma_period = config.get('ma_period', 20)
        self.bb_std = config.get('bb_std', 2.0)            # 布林带标准差
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.hold_minutes = config.get('hold_minutes', 480)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)
        self.take_profit_pct = config.get('take_profit_pct', 0.015)
        
    def _calc_rsi(self, prices, period):
        """计算RSI"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        rsi = np.zeros(len(prices))
        rsi[:] = np.nan
        
        if len(prices) <= period:
            return rsi
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        if avg_loss == 0:
            rs = float('inf')
        else:
            rs = avg_gain / avg_loss
        rsi[period] = 100 - (100 / (1 + rs))
        
        for i in range(period + 1, len(prices)):
            avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
            if avg_loss == 0:
                rs = float('inf')
            else:
                rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self) -> list[Signal]:
        signals = []
        
        for symbol, df in self.data.items():
            if df.empty or len(df) < self.ma_period + 2:
                continue
            
            closes = df['close'].values
            df_ = df.copy()
            
            # 布林带
            df_['ma'] = df_['close'].rolling(self.ma_period).mean()
            df_['std'] = df_['close'].rolling(self.ma_period).std()
            df_['bb_upper'] = df_['ma'] + self.bb_std * df_['std']
            df_['bb_lower'] = df_['ma'] - self.bb_std * df_['std']
            
            # RSI
            rsi = self._calc_rsi(closes, self.rsi_period)
            df_['rsi'] = rsi
            
            prev = df_.iloc[-2]
            today = df_.iloc[-1]
            
            # 超卖反弹(做多): 价格碰下轨 + RSI<30
            if (not pd.isna(prev['bb_lower']) and 
                prev['low'] <= prev['bb_lower'] and
                not pd.isna(prev['rsi']) and prev['rsi'] < self.rsi_oversold):
                signal = Signal(
                    symbol=symbol,
                    direction='long',
                    entry_time=today['date'],
                    entry_price=round(today['open'], 2),
                    exit_time=(today['date'] + timedelta(minutes=self.hold_minutes)
                              if isinstance(today['date'], datetime) else datetime.now()),
                    stop_loss=round(today['open'] * (1 - self.stop_loss_pct), 2),
                    take_profit=round(today['open'] * (1 + self.take_profit_pct), 2),
                    confidence=0.7,
                    reason=f"超卖反弹 RSI={prev['rsi']:.0f} BB下轨={prev['bb_lower']:.0f}",
                    strategy_name=self.name,
                )
                signals.append(signal)
            
            # 超买回落(做空): 价格碰上轨 + RSI>70
            elif (not pd.isna(prev['bb_upper']) and
                  prev['high'] >= prev['bb_upper'] and
                  not pd.isna(prev['rsi']) and prev['rsi'] > self.rsi_overbought):
                signal = Signal(
                    symbol=symbol,
                    direction='short',
                    entry_time=today['date'],
                    entry_price=round(today['open'], 2),
                    exit_time=(today['date'] + timedelta(minutes=self.hold_minutes)
                              if isinstance(today['date'], datetime) else datetime.now()),
                    stop_loss=round(today['open'] * (1 + self.stop_loss_pct), 2),
                    take_profit=round(today['open'] * (1 - self.take_profit_pct), 2),
                    confidence=0.7,
                    reason=f"超买回落 RSI={prev['rsi']:.0f} BB上轨={prev['bb_upper']:.0f}",
                    strategy_name=self.name,
                )
                signals.append(signal)
        
        return signals
    
