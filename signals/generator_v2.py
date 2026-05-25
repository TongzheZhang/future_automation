"""
改进版信号生成器 (v2)

相比 v1 的改进：
  1. 使用多因子评分系统替代单一缺口阈值
  2. 只输出置信度 >= 0.6 的信号
  3. 动态止损（基于 ATR 而非固定百分比）
  4. 更详细的多因子推理说明
  5. 按置信度降序排列信号

用法::

    from signals.generator_v2 import SignalGeneratorV2
    gen = SignalGeneratorV2(config)            # 自定义 or 默认
    signals = gen.generate(data, contracts)    # 生成信号列表
    report  = gen.format_report(signals)       # 格式化为 Markdown 报告
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from typing import Optional
import math

import numpy as np
import pandas as pd
import yaml
from loguru import logger

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from signals.factors import (
    BaseFactor,
    GapMagnitudeFactor,
    VolumeConfirmationFactor,
    TrendAlignmentFactor,
    OpenInterestFactor,
    VolatilityRegimeFactor,
    TimeDecayFactor,
    HistoricalWinRateFactor,
    ADXTrendStrengthFactor,
    WeightedSignalScorer,
    create_default_scorer,
    create_v3_scorer,
    _compute_atr,
    precompute_indicators,
)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class SignalV2:
    """增强版交易信号"""

    symbol: str                         # 品种代码
    name: str                           # 品种中文名
    direction: str                      # 'long' 或 'short'
    entry_price: float                  # 建议入场价
    stop_loss: float                    # 动态止损价 (ATR-based)
    take_profit: float                  # 止盈价
    confidence: float                   # 综合置信度 (0-1)
    hold_minutes: int                   # 建议持仓分钟数
    gap_pct: float                      # 缺口百分比
    atr: float                          # 当前 ATR 值
    atr_pct: float                      # ATR 相对价格百分比
    factor_scores: dict = field(default_factory=dict)  # 各因子得分
    factor_details: list[str] = field(default_factory=list)  # 因子评语
    recommendation: str = ""            # 综合建议

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "direction": "做多" if self.direction == "long" else "做空",
            "direction_code": self.direction,
            "entry_price": round(self.entry_price, 2),
            "stop_loss": round(self.stop_loss, 2),
            "take_profit": round(self.take_profit, 2),
            "confidence": self.confidence,
            "hold_minutes": self.hold_minutes,
            "gap_pct": self.gap_pct,
            "atr": round(self.atr, 2),
            "atr_pct": self.atr_pct,
            "factor_scores": self.factor_scores,
            "reason": " | ".join(self.factor_details),
            "recommendation": self.recommendation,
        }


# ============================================================================
# 信号生成器
# ============================================================================

class SignalGeneratorV2:
    """
    改进版信号生成器

    工作流程：
      1. 遍历所有品种的日线数据
      2. 检测最新开盘缺口
      3. 调用多因子评分器计算置信度
      4. 置信度 >= min_confidence → 生成信号
      5. 计算动态止损/止盈（ATR-based）
      6. 按置信度排序输出
    """

    def __init__(
        self,
        scorer: WeightedSignalScorer = None,
        min_confidence: float = 0.60,
        min_gap_pct: float = 0.003,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        hold_minutes: int = 45,
    ):
        """
        Parameters
        ----------
        scorer : WeightedSignalScorer
            多因子评分器；None 则使用默认配置
        min_confidence : float
            最低置信度阈值
        min_gap_pct : float
            最小缺口百分比（绝对28值），低于此直接跳过
        atr_period : int
            ATR 计算周期
        atr_stop_mult : float
            ATR 止损倍数（止损 = entry ± atr_stop_mult × ATR）
        atr_tp_mult : float
            ATR 止盈倍数（止盈 = entry ± atr_tp_mult × ATR）
        hold_minutes : int
            默认持仓分钟数
        """
        self.scorer = scorer or create_default_scorer()
        self.min_confidence = min_confidence
        self.min_gap_pct = min_gap_pct
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.atr_tp_mult = atr_tp_mult
        self.hold_minutes = hold_minutes

        logger.info(
            f"SignalGeneratorV2 初始化: min_conf={min_confidence}, "
            f"min_gap={min_gap_pct:.1%}, ATR(stop={atr_stop_mult}x, tp={atr_tp_mult}x)"
        )

    def generate(
        self,
        data: dict[str, pd.DataFrame],
        contracts: list[dict],
    ) -> list[SignalV2]:
        """
        扫描所有品种，生成交易信号。

        Parameters
        ----------
        data : dict
            {symbol: DataFrame} 日线数据
        contracts : list[dict]
            品种配置列表

        Returns
        -------
        list[SignalV2]
            按置信度降序排列的信号列表
        """
        signals: list[SignalV2] = []

        for c in contracts:
            symbol = c["symbol"]
            if symbol not in data:
                logger.debug(f"  {symbol}: 无数据, 跳过")
                continue

            df = data[symbol].copy()
            if df.empty or len(df) < self.atr_period + 2:
                logger.debug(f"  {symbol}: 数据不足 {len(df)} 条")
                continue

            # 确保排28序
            df = df.sort_values("date").reset_index(drop=True)

            # 最新两个交易日
            idx = len(df) - 1
            prev = df.iloc[idx - 1]
            latest = df.iloc[idx]

            prev_close = prev["close"]
            today_open = latest["open"]

            if pd.isna(prev_close) or pd.isna(today_open) or prev_close <= 0:
                continue

            gap_pct = (today_open - prev_close) / prev_close

            # 缺口太小，跳过
            if abs(gap_pct) < self.min_gap_pct:
                continue

            # 确定方向（缺口向上→做空等回补；缺口向下→做多）
            direction = "short" if gap_pct > 0 else "long"

            # ---- 预计算指标 + ATR (用于动态止损和因子评分) ----
            precomputed = precompute_indicators(
                df, atr_period=self.atr_period, ma_period=20, vol_period=20, adx_period=14,
            )
            atr_val = precomputed["atr"].iloc[idx] if pd.notna(precomputed["atr"].iloc[idx]) else today_open * 0.015
            atr_pct = atr_val / today_open

            # ---- 多因子评分 ----
            confidence, factor_scores = self.scorer.score(
                df=df,
                index=idx,
                prev_close=prev_close,
                gap_pct=gap_pct,
                direction=direction,
                symbol=symbol,
                precomputed=precomputed,
            )

            if confidence < self.min_confidence:
                logger.debug(
                    f"  {symbol}: 置信度={confidence:.1%} < {self.min_confidence:.1%}, 过滤"
                )
                continue

            # ---- 动态止损/止盈 (ATR-based) ----
            entry_price = today_open
            stop_dist = self.atr_stop_mult * atr_val
            tp_dist = self.atr_tp_mult * atr_val

            if direction == "long":
                stop_loss = entry_price - stop_dist
                take_profit = entry_price + tp_dist
            else:
                stop_loss = entry_price + stop_dist
                take_profit = entry_price - tp_dist

            # ---- 生成因子评语 ----
            factor_details = self._build_factor_details(
                gap_pct=gap_pct,
                direction=direction,
                atr_val=atr_val,
                atr_pct=atr_pct,
                factor_scores=factor_scores,
                df=df,
                idx=idx,
            )

            # ---- 综合建议 ----
            if confidence >= 0.80:
                recommendation = "🔥 强烈推荐"
            elif confidence >= 0.70:
                recommendation = "⭐ 推荐"
            elif confidence >= 0.60:
                recommendation = "💡 可考虑"
            else:
                recommendation = "⚠️ 观望"

            signal = SignalV2(
                symbol=symbol,
                name=c.get("name", symbol),
                direction=direction,
                entry_price=round(entry_price, 2),
                stop_loss=round(stop_loss, 2),
                take_profit=round(take_profit, 2),
                confidence=confidence,
                hold_minutes=self.hold_minutes,
                gap_pct=round(gap_pct, 6),
                atr=round(atr_val, 4),
                atr_pct=round(atr_pct, 6),
                factor_scores=factor_scores,
                factor_details=factor_details,
                recommendation=recommendation,
            )
            signals.append(signal)

        # 按置信度降序排列
        signals.sort(key=lambda s: s.confidence, reverse=True)

        logger.info(f"共生成 {len(signals)} 个信号 (阈值≥{self.min_confidence:.0%})")
        for s in signals:
            logger.info(
                f"  {s.symbol} {s.direction:5s} conf={s.confidence:.1%} "
                f"gap={s.gap_pct:+.2%} sl={s.stop_loss} tp={s.take_profit}"
            )

        return signals

    def _build_factor_details(
        self,
        gap_pct: float,
        direction: str,
        atr_val: float,
        atr_pct: float,
        factor_scores: dict,
        df: pd.DataFrame,
        idx: int,
    ) -> list[str]:
        """构建人类可读的因子评语"""
        details = []

        # 基本描述
        details.append(
            f"开盘跳空{gap_pct:+.2%}"
        )

        # 缺口幅度
        gs = factor_scores.get("gap_magnitude", 0)
        details.append(
            f"缺口幅度评分={gs:.2f} (ATR={atr_val:.1f}, {atr_pct:.2%})"
        )

        # 量能
        vs = factor_scores.get("volume_confirmation", 0)
        vol_ratio = df["volume"].iloc[idx] / df["volume"].iloc[max(0, idx - 20):idx].mean()
        details.append(f"量能确认评分={vs:.2f} (量比={vol_ratio:.1f}x)")

        # 趋势
        ts = factor_scores.get("trend_alignment", 0)
        direction_cn = "做空" if direction == "short" else "做多"
        ts_label = "逆势(高分)" if ts > 0.7 else "顺势" if ts < 0.5 else "震荡"
        details.append(f"趋势对齐评分={ts:.2f} ({ts_label})")

        # 波动率
        vs_regime = factor_scores.get("volatility_regime", 0)
        vol_label = "低" if vs_regime > 0.7 else "中" if vs_regime > 0.4 else "高"
        details.append(f"波动率区间评分={vs_regime:.2f} ({vol_label}波动)")

        # 止损信息
        details.append(f"ATR动态止损={self.atr_stop_mult}×ATR, 止盈={self.atr_tp_mult}×ATR")

        return details

    def format_report(self, signals: list[SignalV2]) -> str:
        """格式化为 Markdown 信号日报"""
        now = datetime.now()

        if not signals:
            return (
                f"# 🔔 期货日内信号日报 (v2)\n"
                f"**{now.strftime('%Y-%m-%d %H:%M')}**\n\n"
                f"暂无符合条件的交易信号 (置信度≥{self.min_confidence:.0%})。\n\n"
                f"> 系统持续监控中，下一个交易日盘前再次扫描。\n"
            )

        lines = [
            f"# 🔔 期货日内信号日报 (v2)",
            f"**{now.strftime('%Y-%m-%d %H:%M')}**  |  "
            f"{len(signals)} 个信号 | "
            f"最低置信度≥{self.min_confidence:.0%}",
            "",
            "---",
            "",
        ]

        for i, s in enumerate(signals):
            conf_emoji = (
                "🔥" if s.confidence >= 0.80
                else "⭐" if s.confidence >= 0.70
                else "💡"
            )

            lines.extend([
                f"### {conf_emoji} 信号 {i + 1}: {s.name}({s.symbol}) — {s.recommendation}",
                "",
                f"| 项目 | 内容 |",
                f"|------|------|",
                f"| **操作** | **{'做多 📈' if s.direction == 'long' else '做空 📉'}** |",
                f"| **综合置信度** | **{s.confidence:.1%}** |",
                f"| **参考入场价** | {s.entry_price} |",
                f"| **止损价** | {s.stop_loss} (ATR动态) |",
                f"| **止盈价** | {s.take_profit} (ATR动态) |",
                f"| **持有时间** | ~{s.hold_minutes} 分钟 |",
                f"| **当前缺口** | {s.gap_pct:+.2%} |",
                f"| **当前 ATR** | {s.atr} ({s.atr_pct:.2%}) |",
                "",
                "**因子评分详情:**",
                "",
            ])

            for fname, fscore in s.factor_scores.items():
                bar = "█" * int(fscore * 10) + "░" * max(0, 10 - int(fscore * 10))
                lines.append(f"- {fname:25s} {bar} {fscore:.2f}")

            lines.extend([
                "",
                f"**推理:** {' | '.join(s.factor_details)}",
                "",
                "---",
                "",
            ])

        lines.extend([
            "## ⚠️ 风险提示",
            "",
            f"- 信号由多因子模型 (7因子+ATR动态止损) 自动生成",
            f"- 止损基于当前 ATR={self.atr_stop_mult}×ATR 动态计算",
            "- 请严格止损，单笔风险控制在总资金 1-2% 以内",
            "- 策略本质为均值回归，极端趋势行情可能连续亏损",
            "",
            "*Spark 自进化投研系统 v2 · 自动生成*",
        ])

        return "\n".join(lines)


# ============================================================================
# 测试入口
# ============================================================================

if __name__ == "__main__":
    # 加载真实配置和数据
    config_dir = Path(__file__).parent.parent / "config"

    with open(config_dir / "settings.yaml") as f:
        settings = yaml.safe_load(f)
    with open(config_dir / "contracts.yaml") as f:
        contracts_data = yaml.safe_load(f)

    contracts = contracts_data["contracts"]

    # 尝试加载真实数据
    try:
        from data.collectors.akshare_adapter import get_all_contracts_data
        data = get_all_contracts_data(contracts)
    except Exception as e:
        logger.warning(f"无法加载真实数据: {e}，使用模拟数据")
        # 兜底：模拟数据
        np.random.seed(42)
        data = {}
        for c in contracts:
            dates = pd.date_range("2025-01-01", periods=200, freq="B")
            close = 4000 + np.cumsum(np.random.randn(200) * 30)
            data[c["symbol"]] = pd.DataFrame({
                "date": dates,
                "open": close + np.random.randn(200) * 5,
                "high": close + abs(np.random.randn(200) * 20),
                "low": close - abs(np.random.randn(200) * 20),
                "close": close,
                "volume": 50000 + np.random.randn(200) * 10000,
                "open_interest": 200000 + np.cumsum(np.random.randn(200) * 500),
            })

    print("=" * 60)
    print("  SignalGeneratorV2 测试")
    print("=" * 60)

    gen = SignalGeneratorV2(min_confidence=0.60)
    signals = gen.generate(data, contracts)
    report = gen.format_report(signals)
    print(report)

    # 输出 JSON 风格的信号概要
    print("\n--- JSON 概要 ---")
    import json as _json
    for s in signals:
        print(_json.dumps(s.to_dict(), ensure_ascii=False, indent=2))