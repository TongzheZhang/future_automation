"""
14:55 平仓记录脚本
- 获取各品种14:55左右的行情
- 对比开盘信号，计算盈亏
- 保存交易记录
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.collectors.market_data import MarketDataCollector
from data.collectors.minute_data import MinuteDataCollector
from intraday.record import load_signals, save_trades
from intraday.models import IntradayTrade, Direction, TradeStatus, CONTRACT_SIZE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("close_position")


def calculate_pnl(direction: Direction, entry: float, exit: float, commodity: str) -> float:
    """计算盈亏（元），按合约乘数*价格差"""
    contract_size = CONTRACT_SIZE.get(commodity, 1)
    pnl = 0.0
    if direction == Direction.LONG:
        pnl = (exit - entry) * contract_size
    elif direction == Direction.SHORT:
        pnl = (entry - exit) * contract_size
    return round(pnl, 2)


async def run_close_position():
    """执行平仓记录"""
    date_str = datetime.now().strftime("%Y-%m-%d")
    logger.info("=" * 60)
    logger.info(f"[{date_str}] 14:55 平仓记录")
    logger.info(f"平仓时间窗口: 09:05 入场 → 14:55 出场")
    logger.info("=" * 60)
    
    # 读取当日信号
    signals = load_signals(date_str)
    if not signals:
        logger.info("今日无信号，无需平仓")
        return []
    
    # 只处理实际交易的信号
    trade_signals = [s for s in signals if s.should_trade()]
    if not trade_signals:
        logger.info("今日无交易信号，无需平仓")
        return []
    
    market = MarketDataCollector()
    minute_collector = MinuteDataCollector()
    trades = []
    
    for sig in trade_signals:
        # 优先使用 AKShare 分钟线获取精确出场价和纯日盘高低点
        actual_exit = await minute_collector.async_get_exit_price(sig.commodity)
        day_high, day_low = await minute_collector.async_get_day_high_low(sig.commodity)
        
        # Fallback：分钟线失败时回退到新浪快照
        if actual_exit is None or day_high is None or day_low is None:
            exit_snapshot = await market.async_get_snapshot(sig.commodity)
            if not exit_snapshot:
                logger.error(f"无法获取14:55收盘行情 {sig.commodity}")
                continue
            if actual_exit is None:
                actual_exit = exit_snapshot.last
            if day_high is None:
                day_high = exit_snapshot.high
            if day_low is None:
                day_low = exit_snapshot.low
        
        # 读取09:05保存的入场基准
        entry_snapshot = sig.market_snapshot
        actual_entry = entry_snapshot.last if entry_snapshot and entry_snapshot.last > 0 else sig.entry_price
        
        # 构建交易记录（数据源：AKShare 1分钟线）
        trade = IntradayTrade(
            date=date_str,
            commodity=sig.commodity,
            direction=sig.direction,
            signal_entry=sig.entry_price,
            signal_stop=sig.stop_loss_price,
            signal_target=sig.target_price,
            confidence=sig.confidence,
            core_logic=sig.core_logic,
            actual_entry=actual_entry,   # 09:05 精确入场价
            actual_exit=actual_exit,      # 14:55 精确出场价
            day_high=day_high,
            day_low=day_low,
            day_close=actual_exit,
        )
        
        # 计算盈亏
        pnl = calculate_pnl(sig.direction, trade.actual_entry, trade.actual_exit, sig.commodity)
        trade.pnl = pnl
        
        # 计算最大回撤（金额）：基于纯日盘(09:05-14:55)高低点
        contract_size = CONTRACT_SIZE.get(sig.commodity, 1)
        if sig.direction == Direction.LONG:
            trade.max_drawdown = round((trade.actual_entry - day_low) * contract_size, 2)
        else:
            trade.max_drawdown = round((day_high - trade.actual_entry) * contract_size, 2)
        
        # 状态判断
        if pnl > 0:
            trade.status = TradeStatus.WIN
        elif pnl < 0:
            trade.status = TradeStatus.LOSS
        else:
            trade.status = TradeStatus.BREAKEVEN
        
        trade.closed_at = datetime.now()
        trades.append(trade)
        
        logger.info(
            f"{sig.commodity} {sig.direction.value} | 09:05入场:{trade.actual_entry} "
            f"14:55平仓:{trade.actual_exit} | 盈亏:{pnl} | {trade.status.value}"
        )
    
    # 保存交易记录
    if trades:
        save_trades(date_str, trades)
    
    # 生成平仓报告
    report_lines = [
        f"# 平仓记录 ({date_str} 14:55)",
        "",
        f"**平仓时间窗口**: 09:05 入场 → 14:55 出场",
        "",
    ]
    
    total_pnl = sum(t.pnl for t in trades)
    win_count = sum(1 for t in trades if t.status == TradeStatus.WIN)
    
    for t in trades:
        emoji = "🟢" if t.status == TradeStatus.WIN else "🔴" if t.status == TradeStatus.LOSS else "⚪"
        report_lines.append(f"{emoji} **{t.commodity}** {t.direction.value}")
        report_lines.append(f"  09:05入场: {t.actual_entry} → 14:55平仓: {t.actual_exit}")
        report_lines.append(f"  盈亏: {t.pnl:+} | 最高: {t.day_high} | 最低: {t.day_low}")
        report_lines.append(f"  最大回撤: {t.max_drawdown} (基于纯日盘 09:05-14:55)")
        report_lines.append("")
    
    report_lines.append(f"**当日总盈亏: {total_pnl:+.2f} | 胜场: {win_count}/{len(trades)}**")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append(f"*平仓时间: {datetime.now().strftime('%H:%M:%S')}*")
    
    report_content = "\n".join(report_lines)
    
    report_dir = Path(__file__).parent.parent / "reports" / "intraday"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"close_{date_str}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print("\n" + "=" * 60)
    print(report_content)
    print("=" * 60)
    
    return trades


if __name__ == "__main__":
    asyncio.run(run_close_position())
