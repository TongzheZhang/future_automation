"""
开场缺口回补策略 (Opening Gap Fill)

逻辑: 每日开盘价相对前一日收盘有较大跳空时,
     大概率在短期内回补缺口。
     
     9:00开盘 → 检测缺口 → 如果>阈值 → 反向交易 → 30-60分钟出场

适用: 黑色系、有色金属等波动大的品种
持仓: 30-60分钟
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger

from strategies.base import BaseStrategy, Signal, StrategyResult


class OpeningGapFillStrategy(BaseStrategy):
    """开盘缺口回补策略"""
    
    name = "opening_gap_fill"
    description = "开盘跳空回补：检测开盘价与前日收盘价的缺口，做反向交易等回补"
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        # 可配置参数
        self.gap_threshold = config.get('gap_threshold', 0.005)    # 缺口阈值 0.5%
        self.hold_minutes = config.get('hold_minutes', 45)          # 持仓45分钟
        self.stop_loss_pct = config.get('stop_loss_pct', 0.015)     # 止损1.5%
        self.take_profit_pct = config.get('take_profit_pct', 0.01)  # 止盈1%
        self.max_gap = config.get('max_gap', 0.03)                  # 最大缺口3%(太大不做)
        
    def generate_signals(self) -> list[Signal]:
        """基于最新数据生成当日信号"""
        signals = []
        
        for symbol, df in self.data.items():
            if df.empty or len(df) < 2:
                continue
            
            # 最新两个交易日
            prev = df.iloc[-2]  # 昨日
            today = df.iloc[-1]  # 今日
            
            prev_close = prev['close']
            today_open = today['open']
            
            if pd.isna(prev_close) or pd.isna(today_open) or prev_close == 0:
                continue
            
            gap_pct = (today_open - prev_close) / prev_close
            
            # 缺口不够大，跳过
            if abs(gap_pct) < self.gap_threshold:
                continue
            
            # 缺口太大，风险高，跳过
            if abs(gap_pct) > self.max_gap:
                continue
            
            # 方向：缺口向上→做空(等回补); 缺口向下→做多(等回补)
            direction = 'short' if gap_pct > 0 else 'long'
            
            entry_price = today_open
            entry_time = today['date'] if 'date' in today else datetime.now()
            
            if direction == 'short':
                stop_loss = entry_price * (1 + self.stop_loss_pct)
                take_profit = entry_price * (1 - self.take_profit_pct)
            else:
                stop_loss = entry_price * (1 - self.stop_loss_pct)
                take_profit = entry_price * (1 + self.take_profit_pct)
            
            exit_time = (entry_time + timedelta(minutes=self.hold_minutes)
                        if isinstance(entry_time, datetime)
                        else datetime.now() + timedelta(minutes=self.hold_minutes))
            
            confidence = min(abs(gap_pct) / self.max_gap, 1.0) * 0.8
            
            signal = Signal(
                symbol=symbol,
                direction=direction,
                entry_time=entry_time,
                entry_price=round(entry_price, 2),
                exit_time=exit_time,
                stop_loss=round(stop_loss, 2),
                take_profit=round(take_profit, 2),
                confidence=round(confidence, 2),
                reason=f"开盘缺口{gap_pct:+.2%}: 开盘{entry_price} vs 昨收{prev_close}",
                strategy_name=self.name,
            )
            signals.append(signal)
        
        return signals
    
