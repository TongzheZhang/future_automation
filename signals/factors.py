"""
多因子信号评分系统

提供 7 个独立因子和一个加权汇总评分器，替代原有单一缺口阈值判断。
每个因子返回 0-1 分值，综合置信度 = Σ(score_i × weight_i) / Σweight_i

因子清单:
  1. GapMagnitudeFactor     - 缺口幅度（相对 ATR 归一化）
  2. VolumeConfirmationFactor - 量能确认（当日量 vs 20 日均量）
  3. TrendAlignmentFactor   - 趋势对齐（缺口方向 vs MA 趋势）
  4. OpenInterestFactor     - 持仓量变化
  5. VolatilityRegimeFactor - 波动率区间
  6. TimeDecayFactor        - 时间衰减
  7. HistoricalWinRateFactor- 历史胜率
  8. ADXTrendStrengthFactor - ADX趋势强度（高ADX=不逆势）【v3新增】
"""
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from loguru import logger


# ============================================================================
# 基类
# ============================================================================

class BaseFactor(ABC):
    """因子基类——所有因子必须实现 compute 方法"""

    name: str = "base"
    weight: float = 1.0  # 默认权重，可在 WeightedSignalScorer 中被覆盖

    def __init__(self, weight: float = None):
        if weight is not None:
            self.weight = weight

    @abstractmethod
    def compute(
        self,
        df: pd.DataFrame,
        index: int,
        prev_close: float,
        gap_pct: float,
        direction: str,
        **kwargs,
    ) -> float:
        """
        计算因子的 0-1 分制评分。

        Parameters
        ----------
        df : pd.DataFrame
            该品种完整的日线数据（含 date, open, high, low, close, volume, open_interest）
        index : int
            当前交易日在 df 中的位置
        prev_close : float
            前一交易日收盘价
        gap_pct : float
            开盘缺口百分比（可正可负）
        direction : str
            'long' 或 'short'（已按缺口方向判定）
        **kwargs
            附加上下文信息

        Returns
        -------
        float
            0-1 分值
        """
        ...

    def __repr__(self):
        return f"{self.name}(w={self.weight:.2f})"


# ============================================================================
# 工具函数
# ============================================================================

def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """计算 Average True Range"""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _compute_ma(series: pd.Series, period: int) -> pd.Series:
    """简单移动平均"""
    return series.rolling(period, min_periods=period).mean()


def _compute_returns_std(close: pd.Series, period: int = 20) -> pd.Series:
    """计算收益率滚动波动率"""
    return close.pct_change().rolling(period, min_periods=period).std()


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """计算 Average Directional Index (ADX)"""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1)
    adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    return adx


def precompute_indicators(
    df: pd.DataFrame,
    atr_period: int = 14,
    ma_period: int = 20,
    vol_period: int = 20,
    adx_period: int = 14,
) -> dict:
    """
    预计算所有因子可能用到的技术指标序列。
    在回测/信号生成主循环外调用一次，避免循环内 O(n²) 重复计算。

    Returns:
        dict: {
            'atr': pd.Series,
            'ma': pd.Series,
            'volatility': pd.Series,
            'adx': pd.Series,
        }
    """
    result = {}
    if {"high", "low", "close"}.issubset(df.columns):
        result["atr"] = _compute_atr(df["high"], df["low"], df["close"], atr_period)
    if "close" in df.columns:
        result["ma"] = _compute_ma(df["close"], ma_period)
        result["volatility"] = _compute_returns_std(df["close"], vol_period)
    if {"high", "low", "close"}.issubset(df.columns):
        result["adx"] = _compute_adx(df["high"], df["low"], df["close"], adx_period)
    return result


# ============================================================================
# 因子 1：缺口幅度因子
# ============================================================================

class GapMagnitudeFactor(BaseFactor):
    """
    缺口幅度因子

    使用 ATR 对缺口大小进行归一化：
      score = min(|gap_pct| / (atr_pct × k), 1.0)

    缺口太小 → 分数低（噪声）
    缺口太大 → 分数也降（极端行情不宜逆势）
    """

    name = "gap_magnitude"

    def __init__(self, atr_period: int = 14, k: float = 1.5, weight: float = 1.0):
        """
        Parameters
        ----------
        atr_period : int
            ATR 计算周期
        k : float
            归一化乘数；缺口/ATR 比达到 k 倍时得分=1
        """
        super().__init__(weight=weight)
        self.atr_period = atr_period
        self.k = k

    def compute(
        self,
        df: pd.DataFrame,
        index: int,
        prev_close: float,
        gap_pct: float,
        direction: str,
        **kwargs,
    ) -> float:
        if index < self.atr_period + 1:
            return 0.5  # 数据不足，给中性分数

        # 优先使用预计算指标
        precomputed = kwargs.get('precomputed', {})
        atr_series = precomputed.get('atr')
        if atr_series is not None and index < len(atr_series):
            atr_val = atr_series.iloc[index]
        else:
            sub = df.iloc[:index + 1]
            atr = _compute_atr(sub['high'], sub['low'], sub['close'], self.atr_period)
            atr_val = atr.iloc[-1]
        atr_pct = atr_val / prev_close if prev_close > 0 else 0.02

        if atr_pct <= 0:
            return 0.5

        # 缺口相对 ATR 的倍数
        ratio = abs(gap_pct) / (atr_pct * self.k)

        # 缺口过小 → 低分；适中 → 高分；过大 → 回落
        score = min(ratio, 2.0 - ratio, 1.0)
        score = max(score, 0.1)
        return float(score)


# ============================================================================
# 因子 2：量能确认因子
# ============================================================================

class VolumeConfirmationFactor(BaseFactor):
    """
    量能确认因子

    逻辑：高量能突破更可靠，缩量跳空多为假信号。
      score = min(vol / avg_vol, 2.0) / 2.0   (归一化到 0-1)
    """

    name = "volume_confirmation"

    def __init__(self, vol_ma_period: int = 20, weight: float = 1.0):
        super().__init__(weight=weight)
        self.vol_ma_period = vol_ma_period

    def compute(
        self,
        df: pd.DataFrame,
        index: int,
        prev_close: float,
        gap_pct: float,
        direction: str,
        **kwargs,
    ) -> float:
        if index < self.vol_ma_period:
            return 0.5

        vol = df['volume'].iloc[index]
        avg_vol = df['volume'].iloc[index - self.vol_ma_period:index].mean()

        if avg_vol <= 0:
            return 0.5

        ratio = vol / avg_vol
        score = min(ratio, 2.0) / 2.0
        return float(score)


# ============================================================================
# 因子 3：趋势对齐因子
# ============================================================================

class TrendAlignmentFactor(BaseFactor):
    """
    趋势对齐因子

    逻辑：
    - 缺口方向与 MA 趋势同向 → 趋势延续信号 → 追势风险较低，但回补概率也低 → 中等分数
    - 缺口方向与 MA 趋势逆向 → 逆势交易，回补概率更高，但风险也更大 → 给较高分数
      因为我们的策略本质是均值回归（期待回补）
    """

    name = "trend_alignment"

    def __init__(self, ma_period: int = 20, weight: float = 1.0):
        super().__init__(weight=weight)
        self.ma_period = ma_period

    def compute(
        self,
        df: pd.DataFrame,
        index: int,
        prev_close: float,
        gap_pct: float,
        direction: str,
        **kwargs,
    ) -> float:
        if index < self.ma_period + 2:
            return 0.5

        # 优先使用预计算指标
        precomputed = kwargs.get('precomputed', {})
        ma_series = precomputed.get('ma')
        if ma_series is not None and index < len(ma_series):
            ma_val = ma_series.iloc[index]
            ma_prev = ma_series.iloc[index - self.ma_period] if index >= self.ma_period else ma_val
            ma_slope = (ma_val - ma_prev) / ma_prev if ma_prev != 0 else 0
        else:
            sub = df.iloc[:index + 1]
            ma = _compute_ma(sub['close'], self.ma_period)
            ma_slope = (ma.iloc[-1] - ma.iloc[-self.ma_period]) / ma.iloc[-self.ma_period] if ma.iloc[-1] != 0 else 0

        # gap > 0 表示向上跳空 → direction='short' → 做空
        # gap < 0 表示向下跳空 → direction='long' → 做多
        # 均线上升(ma_slope>0)  → 趋势向上
        # 做空 + 趋势向上 → 逆势 → 高分
        # 做空 + 趋势向下 → 顺势 → 低分

        trend_up = ma_slope > 0.01  # 上升趋势
        trend_down = ma_slope < -0.01  # 下降趋势

        if direction == 'short' and trend_up:
            score = 0.85  # 做空 + 上升趋势: 逆势, 回补概率高
        elif direction == 'long' and trend_down:
            score = 0.85  # 做多 + 下降趋势: 逆势, 回补概率高
        elif direction == 'short' and trend_down:
            score = 0.40  # 做空 + 下降趋势: 顺势, 回补概率低
        elif direction == 'long' and trend_up:
            score = 0.40  # 做多 + 上升趋势: 顺势, 回补概率低
        else:
            score = 0.60  # 震荡市

        return score


# ============================================================================
# 因子 4：持仓量变化因子
# ============================================================================

class OpenInterestFactor(BaseFactor):
    """
    持仓量变化因子

    逻辑：
    - 增仓跳空：多空博弈加剧，趋势延续性强 → 回补信号减弱 → 低分
    - 减仓跳空：获利平仓/止损出局，趋势接近尾声 → 回补概率高 → 高分
    - 持仓不变：中性
    """

    name = "open_interest"

    def __init__(self, oi_ma_period: int = 5, weight: float = 1.0):
        super().__init__(weight=weight)
        self.oi_ma_period = oi_ma_period

    def compute(
        self,
        df: pd.DataFrame,
        index: int,
        prev_close: float,
        gap_pct: float,
        direction: str,
        **kwargs,
    ) -> float:
        if index < self.oi_ma_period or 'open_interest' not in df.columns:
            return 0.5

        oi_current = df['open_interest'].iloc[index]
        oi_prev = df['open_interest'].iloc[index - 1]

        if pd.isna(oi_current) or pd.isna(oi_prev) or oi_prev == 0:
            return 0.5

        oi_change_pct = (oi_current - oi_prev) / oi_prev

        if oi_change_pct > 0.05:
            score = 0.30  # 大幅增仓, 趋势延续
        elif oi_change_pct > 0.02:
            score = 0.45  # 小幅增仓
        elif oi_change_pct < -0.05:
            score = 0.85  # 大幅减仓, 趋势尾声
        elif oi_change_pct < -0.02:
            score = 0.70  # 小幅减仓
        else:
            score = 0.60

        return score


# ============================================================================
# 因子 5：波动率区间因子
# ============================================================================

class VolatilityRegimeFactor(BaseFactor):
    """
    波动率区间因子

    逻辑：
    - 低波动环境：跳空信号更可靠，回补概率高 → 高分
    - 高波动环境：跳空信号噪声大，止损易触发 → 低分
    """

    name = "volatility_regime"

    def __init__(self, vol_period: int = 20, weight: float = 1.0):
        super().__init__(weight=weight)
        self.vol_period = vol_period

    def compute(
        self,
        df: pd.DataFrame,
        index: int,
        prev_close: float,
        gap_pct: float,
        direction: str,
        **kwargs,
    ) -> float:
        if index < self.vol_period + 1:
            return 0.5

        # 优先使用预计算指标
        precomputed = kwargs.get('precomputed', {})
        vol_series = precomputed.get('volatility')
        if vol_series is not None and index < len(vol_series):
            current_vol = vol_series.iloc[index]
            vol_window = vol_series.iloc[max(0, index - self.vol_period * 2 + 1):index + 1].dropna()
        else:
            sub = df.iloc[:index + 1]
            vol = _compute_returns_std(sub['close'], self.vol_period)
            current_vol = vol.iloc[-1]
            vol_window = vol.iloc[-(self.vol_period * 2):].dropna()
        if len(vol_window) < self.vol_period:
            return 0.5

        percentile = (vol_window < current_vol).mean()

        if percentile > 0.8:
            score = 0.25  # 极高波动
        elif percentile > 0.6:
            score = 0.40  # 高波动
        elif percentile > 0.4:
            score = 0.65  # 中等
        elif percentile > 0.2:
            score = 0.80  # 低波动
        else:
            score = 0.90  # 极低波动

        return score


# ============================================================================
# 因子 6：时间衰减因子
# ============================================================================

class TimeDecayFactor(BaseFactor):
    """
    时间衰减因子

    逻辑：缺口产生后越早入场，回补概率越高（适用于日内场景）。
    用当日缺口检测点的"时效性"度量——这里用最近几天的数据新鲜度做代理。

    对于日线回测场景，考虑：
    - 最近一次大缺口距今越近 → 市场记忆越新鲜 → 分数越高
    - 缺口发生在多日之前 → 衰减

    实际使用中，此因子依赖于入场时机，当与分钟级数据结合时更有意义。
    """

    name = "time_decay"

    def __init__(self, decay_days: int = 5, weight: float = 0.5):
        """
        Parameters
        ----------
        decay_days : int
            衰减周期（天）
        """
        super().__init__(weight=weight)
        self.decay_days = decay_days

    def compute(
        self,
        df: pd.DataFrame,
        index: int,
        prev_close: float,
        gap_pct: float,
        direction: str,
        **kwargs,
    ) -> float:
        # 当最近 N 天内有多个缺口时，市场不确定性强
        if index < self.decay_days:
            return 0.7

        close_series = df['close'].iloc[index - self.decay_days:index]
        prev_close_series = df['close'].iloc[index - self.decay_days - 1:index - 1]

        recent_gaps = [
            abs((close_series.iloc[i] - prev_close_series.iloc[i]) / prev_close_series.iloc[i])
            for i in range(len(close_series))
            if prev_close_series.iloc[i] != 0
        ]

        n_big_gaps = sum(1 for g in recent_gaps if g > 0.01)

        if n_big_gaps >= 3:
            score = 0.35  # 连续缺口太多，市场混乱
        elif n_big_gaps >= 2:
            score = 0.50
        elif n_big_gaps >= 1:
            score = 0.65
        else:
            score = 0.85  # 近几日无缺口，当前缺口更显著

        return score


# ============================================================================
# 因子 7：历史胜率因子
# ============================================================================

class HistoricalWinRateFactor(BaseFactor):
    """
    历史胜率因子

    逻辑：跟踪该品种该方向的历史交易表现。
    支持 JSON 持久化，进程重启后自动加载历史记录。
    """

    name = "historical_win_rate"

    def __init__(self, min_samples: int = 10, weight: float = 0.7,
                 persist: bool = True):
        """
        Parameters
        ----------
        min_samples : int
            最少样本数，低于此数返回中性分
        persist : bool
            是否启用 JSON 持久化
        """
        super().__init__(weight=weight)
        self.min_samples = min_samples
        self.persist = persist
        self._history_file = Path(__file__).parent.parent / "data" / "historical_win_rate.json"
        self._history: dict = defaultdict(list)  # {(symbol, direction): [1/0 win/loss]}
        if self.persist:
            self._load()

    def _load(self):
        """从 JSON 加载历史记录"""
        if self._history_file.exists():
            try:
                import json
                raw = json.loads(self._history_file.read_text())
                for k, v in raw.items():
                    symbol, direction = k.split("::")
                    self._history[(symbol, direction)] = v
            except Exception as e:
                logger.warning(f"加载历史胜率记录失败: {e}")

    def _save(self):
        """保存到 JSON"""
        if not self.persist:
            return
        try:
            import json
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            raw = {f"{k[0]}::{k[1]}": v for k, v in self._history.items()}
            self._history_file.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"保存历史胜率记录失败: {e}")

    def update(self, symbol: str, direction: str, pnl: float):
        """更新历史记录（回测后调用）"""
        key = (symbol, direction)
        self._history[key].append(1 if pnl > 0 else 0)
        # 保留最近 500 条，避免文件过大
        if len(self._history[key]) > 500:
            self._history[key] = self._history[key][-500:]
        self._save()

    def compute(
        self,
        df: pd.DataFrame,
        index: int,
        prev_close: float,
        gap_pct: float,
        direction: str,
        **kwargs,
    ) -> float:
        symbol = kwargs.get('symbol', 'UNKNOWN')
        key = (symbol, direction)
        history = self._history.get(key, [])

        if len(history) < self.min_samples:
            return 0.5  # 样本不足，中性

        win_rate = sum(history) / len(history)

        # 将胜率映射到 0-1
        if win_rate > 0.60:
            score = 0.85
        elif win_rate > 0.55:
            score = 0.70
        elif win_rate > 0.50:
            score = 0.55
        elif win_rate > 0.40:
            score = 0.40
        else:
            score = 0.20

        return score


# ============================================================================
# 因子 8：ADX 趋势强度因子【v3新增】
# ============================================================================

class ADXTrendStrengthFactor(BaseFactor):
    """
    ADX 趋势强度因子

    逻辑：
    - ADX > 25: 强趋势 → 绝不逆势做均值回归 → 得分极低（0.15）
    - ADX 20-25: 中等趋势 → 谨慎 → 得分偏低（0.35）
    - ADX < 20: 弱趋势/震荡 → 均值回归有效 → 得分高（0.85）

    这是 v3 最关键的改进：过滤掉强趋势行情，只保留适合均值回归的震荡市。
    """

    name = "adx_trend_strength"

    def __init__(self, adx_period: int = 14, strong_threshold: float = 25.0,
                 weak_threshold: float = 20.0, weight: float = 1.5):
        """
        Parameters
        ----------
        adx_period : int
            ADX 计算周期
        strong_threshold : float
            ADX 超过此值视为强趋势
        weak_threshold : float
            ADX 低于此值视为弱趋势/震荡
        """
        super().__init__(weight=weight)
        self.adx_period = adx_period
        self.strong_threshold = strong_threshold
        self.weak_threshold = weak_threshold

    def compute(
        self,
        df: pd.DataFrame,
        index: int,
        prev_close: float,
        gap_pct: float,
        direction: str,
        **kwargs,
    ) -> float:
        if index < self.adx_period * 2:
            return 0.5

        # 优先使用预计算指标
        precomputed = kwargs.get('precomputed', {})
        adx_series = precomputed.get('adx')
        if adx_series is not None and index < len(adx_series):
            adx_val = adx_series.iloc[index]
        else:
            sub = df.iloc[:index + 1]
            high = sub['high']
            low = sub['low']
            close = sub['close']

            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ], axis=1).max(axis=1)

            up_move = high.diff()
            down_move = -low.diff()
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

            atr = tr.ewm(alpha=1.0 / self.adx_period, adjust=False).mean()
            plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1.0 / self.adx_period, adjust=False).mean() / atr.replace(0, np.nan)
            minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1.0 / self.adx_period, adjust=False).mean() / atr.replace(0, np.nan)

            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1)
            adx = dx.ewm(alpha=1.0 / self.adx_period, adjust=False).mean()
            adx_val = adx.iloc[-1]

        if pd.isna(adx_val):
            return 0.5

        # 阶梯评分
        if adx_val > self.strong_threshold:
            score = 0.15  # 强趋势，不逆势
        elif adx_val > self.weak_threshold + 2:
            score = 0.35  # 中等趋势
        elif adx_val > self.weak_threshold:
            score = 0.55  # 弱趋势
        else:
            score = 0.85  # 震荡市，均值回归有效

        return score


# ============================================================================
# 加权信号评分器
# ============================================================================

class WeightedSignalScorer:
    """
    加权信号评分器

    汇总多个因子的评分，输出综合置信度（0-1）。

    用法::

        scorer = WeightedSignalScorer(factors=[...])
        confidence = scorer.score(df=df, index=10, prev_close=4200.0,
                                  gap_pct=0.012, direction='short')

    """

    def __init__(self, factors: list[BaseFactor]):
        """
        Parameters
        ----------
        factors : list[BaseFactor]
            因子列表（包含权重配置）
        """
        self.factors = factors
        total_w = sum(f.weight for f in factors) or 1.0
        self._weights = [f.weight / total_w for f in factors]
        logger.info(
            f"初始化评分器: {len(factors)} 个因子, "
            f"权重 {[f'{f.name}={w:.2f}' for f, w in zip(factors, self._weights)]}"
        )

    @property
    def factor_names(self) -> list[str]:
        return [f.name for f in self.factors]

    def score(
        self,
        df: pd.DataFrame,
        index: int,
        prev_close: float,
        gap_pct: float,
        direction: str,
        precomputed: dict = None,
        **kwargs,
    ) -> tuple[float, dict[str, float]]:
        """
        计算综合置信度和各因子子得分。

        Parameters
        ----------
        precomputed : dict, optional
            由 precompute_indicators() 预计算的技术指标序列，
            可显著减少回测主循环内的重复计算。

        Returns
        -------
        confidence : float
            0-1 综合置信度
        details : dict
            {factor_name: score}
        """
        scores = []
        details = {}

        # 将预计算指标通过 kwargs 传入各因子
        if precomputed is not None:
            kwargs.setdefault("precomputed", precomputed)

        for factor in self.factors:
            s = factor.compute(
                df=df, index=index, prev_close=prev_close,
                gap_pct=gap_pct, direction=direction, **kwargs,
            )
            s = max(0.0, min(1.0, s))
            scores.append(s)
            details[factor.name] = round(s, 4)

        confidence = sum(s * w for s, w in zip(scores, self._weights))
        confidence = max(0.0, min(1.0, confidence))

        return round(confidence, 4), details

    def explain(
        self,
        df: pd.DataFrame,
        index: int,
        prev_close: float,
        gap_pct: float,
        direction: str,
        **kwargs,
    ) -> str:
        """生成人类可读的评分说28明"""
        confidence, details = self.score(
            df=df, index=index, prev_close=prev_close,
            gap_pct=gap_pct, direction=direction, **kwargs,
        )
        lines = [f"综合置信度: {confidence:.1%}"]
        for name, score in details.items():
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            lines.append(f"  {name:25s} {bar} {score:.4f}")
        return "\n".join(lines)


# ============================================================================
# 默认评分器工厂
# ============================================================================

def create_default_scorer() -> WeightedSignalScorer:
    """
    创建默认评分器（使用推荐权重配置）。

    Returns
    -------
    WeightedSignalScorer
    """
    factors = [
        GapMagnitudeFactor(atr_period=14, k=1.5, weight=1.2),
        VolumeConfirmationFactor(vol_ma_period=20, weight=1.0),
        TrendAlignmentFactor(ma_period=20, weight=1.1),
        OpenInterestFactor(oi_ma_period=5, weight=0.8),
        VolatilityRegimeFactor(vol_period=20, weight=0.9),
        TimeDecayFactor(decay_days=5, weight=0.5),
        HistoricalWinRateFactor(min_samples=10, weight=0.7),
    ]
    return WeightedSignalScorer(factors=factors)


def create_v3_scorer() -> WeightedSignalScorer:
    """
    创建 v3 评分器（包含 ADX 趋势强度过滤 + 更高权重向核心因子倾斜）。

    v3 关键改进：
    - ADX 因子权重最高(1.5)，强趋势行情直接过滤
    - 提高波动率区间因子权重(1.2)，低波环境更可靠
    - 趋势对齐因子权重提升(1.3)，逆势信号更受重视
    - 缺口幅度因子提升(1.3)，适中缺口最重要

    Returns
    -------
    WeightedSignalScorer
    """
    factors = [
        GapMagnitudeFactor(atr_period=14, k=1.5, weight=1.3),
        VolumeConfirmationFactor(vol_ma_period=20, weight=1.0),
        TrendAlignmentFactor(ma_period=20, weight=1.3),
        OpenInterestFactor(oi_ma_period=5, weight=0.8),
        VolatilityRegimeFactor(vol_period=20, weight=1.2),
        TimeDecayFactor(decay_days=5, weight=0.5),
        HistoricalWinRateFactor(min_samples=10, weight=0.5),
        ADXTrendStrengthFactor(adx_period=14, weight=1.5),  # 【v3核心】
    ]
    return WeightedSignalScorer(factors=factors)


# ============================================================================
# 测试入口
# ============================================================================

if __name__ == "__main__":
    # 构造模拟数据
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=200, freq="B")
    close = 4000 + np.cumsum(np.random.randn(200) * 20)
    df_test = pd.DataFrame({
        "date": dates,
        "open": close + np.random.randn(200) * 5,
        "high": close + abs(np.random.randn(200) * 15),
        "low": close - abs(np.random.randn(200) * 15),
        "close": close,
        "volume": 50000 + np.random.randn(200) * 10000,
        "open_interest": 200000 + np.cumsum(np.random.randn(200) * 500),
    })
    df_test["volume"] = df_test["volume"].clip(lower=1000)
    df_test["open_interest"] = df_test["open_interest"].clip(lower=100)

    print("=" * 60)
    print("  多因子评分系28统测试")
    print("=" * 60)

    # 测试单因子
    index = 150
    prev_close = df_test["close"].iloc[index - 1]
    gap_pct_vals = [0.005, -0.012, 0.0005]
    gap_pct = gap_pct_vals[1]  # -1.2%
    direction = "long"  # 负缺口做多

    for factor_cls in [
        GapMagnitudeFactor, VolumeConfirmationFactor, TrendAlignmentFactor,
        OpenInterestFactor, VolatilityRegimeFactor, TimeDecayFactor,
        HistoricalWinRateFactor,
    ]:
        factor = factor_cls()
        score = factor.compute(df_test, index, prev_close, gap_pct, direction, symbol="TEST")
        print(f"  {factor.name:25s} → {score:.4f}")

    # 测试评分器
    print("\n--- 加权评分器 ---")
    scorer = create_default_scorer()
    for gap in gap_pct_vals:
        dir_ = 'short' if gap > 0 else 'long'
        conf, details = scorer.score(df_test, index, prev_close, gap, dir_, symbol="TEST")
        print(f"\n  缺口={gap:+.1%} 方向={dir_:5s} 置信度={conf:.1%}")
        print(scorer.explain(df_test, index, prev_close, gap, dir_, symbol="TEST"))