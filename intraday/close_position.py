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
from intraday.record import load_signals, save_trades
from intraday.models import IntradayTrade, Direction, TradeStatus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("close_position")


def calculate_pnl(direction: Direction, entry: float, exit: float, commodity: str) -> float:
    """计算盈亏（简化，按1手，元）"""
    # 简化计算，实际应考虑合约大小
    # 这里用价格差 * 一个简化系数
    pnl = 0.0
    if direction == Direction.LONG:
        pnl = exit - entry
    elif direction == Direction.SHORT:
        pnl = entry - exit
    
    # 简化：不乘以合约大小，只返回价格差
    # 实际交易时应根据品种乘以合约吨数
    return round(pnl, 2)


async def run_close_position():
    """执行平仓记录"""
    date_str = datetime.now().strftime("%Y-%m-%d")
    logger.info("=" * 60)
    logger.info(f"[{date_str}] 14:55 平仓记录")
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
    trades = []
    
    for sig in trade_signals:
        # 获取收盘行情
        snapshot = market.get_snapshot(sig.commodity)
        if not snapshot:
            logger.error(f"无法获取收盘行情 {sig.commodity}")
            continue
        
        # 构建交易记录
        trade = IntradayTrade(
            date=date_str,
            commodity=sig.commodity,
            direction=sig.direction,
            signal_entry=sig.entry_price,
            signal_stop=sig.stop_loss_price,
            signal_target=sig.target_price,
            confidence=sig.confidence,
            core_logic=sig.core_logic,
            actual_entry=sig.entry_price,  # 简化：假设按建议价成交
            actual_exit=snapshot.last,     # 按最新价平仓
            day_high=snapshot.high,
            day_low=snapshot.low,
            day_close=snapshot.last,
        )
        
        # 计算盈亏
        pnl = calculate_pnl(sig.direction, trade.actual_entry, trade.actual_exit, sig.commodity)
        trade.pnl = pnl
        
        # 计算最大回撤
        if sig.direction == Direction.LONG:
            trade.max_drawdown = round(sig.entry_price - snapshot.low, 2)
        else:
            trade.max_drawdown = round(snapshot.high - sig.entry_price, 2)
        
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
            f"{sig.commodity} {sig.direction.value} | 入场:{trade.actual_entry} "
            f"平仓:{trade.actual_exit} | 盈亏:{pnl} | {trade.status.value}"
        )
    
    # 保存交易记录
    if trades:
        save_trades(date_str, trades)
    
    # 生成平仓报告
    report_lines = [
        f"# 平仓记录 ({date_str} 14:55)",
        "",
    ]
    
    total_pnl = sum(t.pnl for t in trades)
    win_count = sum(1 for t in trades if t.status == TradeStatus.WIN)
    
    for t in trades:
        emoji = "🟢" if t.status == TradeStatus.WIN else "🔴" if t.status == TradeStatus.LOSS else "⚪"
        report_lines.append(f"{emoji} **{t.commodity}** {t.direction.value}")
        report_lines.append(f"  入场: {t.actual_entry} → 平仓: {t.actual_exit}")
        report_lines.append(f"  盈亏: {t.pnl:+} | 最高: {t.day_high} | 最低: {t.day_low}")
        report_lines.append(f"  最大回撤: {t.max_drawdown}")
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
