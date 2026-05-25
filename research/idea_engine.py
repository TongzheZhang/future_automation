"""Convert research articles and LLM analysis into structured strategy ideas."""
from __future__ import annotations

import hashlib
import re
from typing import Any

from research.models import ResearchArticle, SourceRef, StrategyIdea, utc_now_iso


FAMILY_KEYWORDS = [
    ("trend_breakout", ("趋势", "突破", "走强", "走弱", "单边", "延续")),
    ("mean_reversion", ("回归", "修复", "超跌", "超涨", "回补")),
    ("gap_filter", ("缺口", "跳空", "隔夜", "外盘", "开盘")),
    ("regime_filter", ("宏观", "政策", "波动", "风险偏好", "情绪")),
]


def extract_symbols(text: str, contracts: list[dict]) -> list[str]:
    found: list[str] = []
    upper_text = text.upper()
    for contract in contracts:
        symbol = contract.get("symbol", "")
        name = contract.get("name", "")
        ak_symbol = contract.get("akshare_symbol", "")
        if symbol and re.search(rf"\b{re.escape(symbol.upper())}\b", upper_text):
            found.append(symbol)
            continue
        if ak_symbol and ak_symbol.upper() in upper_text:
            found.append(symbol)
            continue
        if name and name in text:
            found.append(symbol)
    return sorted(set(found))


def generate_strategy_ideas(
    articles: list[ResearchArticle],
    analysis: dict[str, Any],
    contracts: list[dict],
    config: dict | None = None,
) -> list[StrategyIdea]:
    """Generate validated ideas from LLM output first, then fallback article rules."""
    config = config or {}
    max_ideas = int(config.get("max_ideas_per_run", 6))
    ideas: list[StrategyIdea] = []

    for raw in analysis.get("ideas", []) or []:
        idea = _idea_from_llm(raw, articles, contracts)
        if idea:
            ideas.append(idea)
        if len(ideas) >= max_ideas:
            break

    if not ideas:
        for article in articles:
            idea = _idea_from_article(article, analysis, contracts)
            if idea:
                ideas.append(idea)
            if len(ideas) >= max_ideas:
                break

    unique: list[StrategyIdea] = []
    seen = set()
    for idea in ideas:
        key = (tuple(idea.symbols), idea.strategy_family, idea.title)
        if key in seen:
            continue
        idea.validate()
        seen.add(key)
        unique.append(idea)
    return unique[:max_ideas]


def _idea_from_llm(
    raw: dict[str, Any],
    articles: list[ResearchArticle],
    contracts: list[dict],
) -> StrategyIdea | None:
    title = str(raw.get("title", "")).strip()
    thesis = str(raw.get("thesis", "")).strip()
    if not title or not thesis:
        return None
    text = " ".join([title, thesis, str(raw.get("expected_regime", ""))])
    symbols = _safe_symbols(raw.get("symbols"), contracts) or extract_symbols(text, contracts)
    if not symbols:
        symbols = _symbols_from_articles(articles, contracts)[:2]
    if not symbols:
        return None
    family = _safe_family(str(raw.get("strategy_family", "")) or infer_strategy_family(text))
    direction_bias = _safe_direction(str(raw.get("direction_bias", "")) or infer_direction_bias(text))
    refs = _source_refs_for_symbols(articles, symbols) or [a.source_ref() for a in articles[:3]]
    params = build_candidate_params(symbols, family, direction_bias, text)
    return StrategyIdea(
        id=idea_id(title, symbols, refs),
        created_at=utc_now_iso(),
        title=title[:120],
        symbols=symbols,
        strategy_family=family,
        thesis=thesis,
        source_refs=refs,
        entry_logic=entry_logic_for_family(family, direction_bias),
        filters=filters_for_family(family, text),
        exit_logic="沿用现有回测框架：固定持仓窗口并使用 ATR 止损/止盈，先做离线参数对比。",
        risk_controls=["仅进入研究候选池", "必须通过回测和样本外验证", "不自动进入生产信号"],
        candidate_params=params,
        expected_regime=str(raw.get("expected_regime", "相关基本面线索持续发酵的行情")).strip(),
        invalidation=str(raw.get("invalidation", "来源线索反转，或回测显示胜率/夏普无改善")).strip(),
        novelty_score=score_novelty(refs, text),
        testability_score=score_testability(symbols, params),
        confidence=score_confidence(refs, symbols, analysis_mode="ollama"),
    )


def _idea_from_article(
    article: ResearchArticle,
    analysis: dict[str, Any],
    contracts: list[dict],
) -> StrategyIdea | None:
    text = article.text_blob(2500)
    symbols = article.symbols or extract_symbols(text, contracts)
    if not symbols:
        hints = analysis.get("symbol_hints", [])
        symbols = _safe_symbols(hints, contracts)
    if not symbols:
        return None
    family = infer_strategy_family(text)
    direction_bias = infer_direction_bias(text)
    refs = [article.source_ref()]
    title = f"{'/'.join(symbols)} {family_label(family)}研究候选"
    thesis = build_thesis(article, family, direction_bias)
    params = build_candidate_params(symbols, family, direction_bias, text)
    return StrategyIdea(
        id=idea_id(title, symbols, refs),
        created_at=utc_now_iso(),
        title=title,
        symbols=symbols,
        strategy_family=family,
        thesis=thesis,
        source_refs=refs,
        entry_logic=entry_logic_for_family(family, direction_bias),
        filters=filters_for_family(family, text),
        exit_logic="先用现有 v3 回测参数做基准，再比较候选参数对 Sharpe、胜率、回撤的影响。",
        risk_controls=["研究候选不自动交易", "限制到已有安全参数", "至少需要跨年份或 walk-forward 验证"],
        candidate_params=params,
        expected_regime=expected_regime_for_text(text),
        invalidation="若来源线索消失、方向偏置失效，或候选参数相对 v3 基准没有提升，则归档。",
        novelty_score=score_novelty(refs, text),
        testability_score=score_testability(symbols, params),
        confidence=score_confidence(refs, symbols, analysis.get("mode", "fallback")),
    )


def infer_strategy_family(text: str) -> str:
    for family, words in FAMILY_KEYWORDS:
        if any(word in text for word in words):
            return family
    return "regime_filter"


def infer_direction_bias(text: str) -> str:
    long_words = ("下降", "减少", "去库", "检修", "供应收缩", "走强", "反弹", "需求改善")
    short_words = ("累库", "增加", "过剩", "走弱", "下滑", "需求疲弱", "供应恢复")
    long_score = sum(word in text for word in long_words)
    short_score = sum(word in text for word in short_words)
    if long_score > short_score:
        return "long"
    if short_score > long_score:
        return "short"
    return "neutral"


def build_candidate_params(
    symbols: list[str],
    family: str,
    direction_bias: str,
    text: str,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "include_symbols": symbols,
        "exclude_symbols": [],
        "min_gap_pct": 0.005,
        "min_confidence": 0.64,
        "hold_minutes": 45,
        "atr_stop_mult": 2.0,
        "atr_tp_mult": 3.0,
        "factor_weight_overrides": {},
        "direction_bias": direction_bias,
    }
    if family == "gap_filter":
        params["factor_weight_overrides"] = {"adx_trend_strength": 1.7, "volatility_regime": 1.3}
        if "外盘" in text or "隔夜" in text:
            params["min_confidence"] = 0.68
            params["hold_minutes"] = 30
    elif family == "trend_breakout":
        params["min_confidence"] = 0.70
        params["hold_minutes"] = 60
        params["factor_weight_overrides"] = {"trend_alignment": 1.5, "volume_confirmation": 1.2}
    elif family == "mean_reversion":
        params["min_confidence"] = 0.66
        params["atr_tp_mult"] = 2.5
        params["factor_weight_overrides"] = {"gap_magnitude": 1.4, "adx_trend_strength": 1.8}
    else:
        params["min_confidence"] = 0.67
        params["factor_weight_overrides"] = {"volatility_regime": 1.5, "time_decay": 0.8}
    return params


def entry_logic_for_family(family: str, direction_bias: str) -> str:
    bias_text = {
        "long": "优先观察做多候选，做空信号提高过滤门槛。",
        "short": "优先观察做空候选，做多信号保持基准门槛。",
        "neutral": "不预设方向，使用现有缺口方向和多因子评分。",
    }[direction_bias]
    if family == "trend_breakout":
        return f"当外部线索支持趋势延续时，提高趋势和量能因子权重；{bias_text}"
    if family == "gap_filter":
        return f"当开盘缺口来自外部信息传导或隔夜波动时，提高置信度门槛；{bias_text}"
    if family == "mean_reversion":
        return f"仅在趋势强度不高且缺口幅度适中时测试均值回归；{bias_text}"
    return f"把宏观/政策/波动状态作为交易过滤器，而不是直接生成信号；{bias_text}"


def filters_for_family(family: str, text: str) -> list[str]:
    filters = ["保留现有 v3 品种白名单逻辑", "置信度必须达到候选参数门槛"]
    if "外盘" in text or "隔夜" in text:
        filters.append("外盘/隔夜冲击相关信号需要单独分组评估")
    if "库存" in text or "基差" in text:
        filters.append("库存或基差线索未延续时不启用该候选")
    if family in {"trend_breakout", "gap_filter"}:
        filters.append("ADX 强趋势环境下避免机械逆势")
    return filters


def build_thesis(article: ResearchArticle, family: str, direction_bias: str) -> str:
    return (
        f"来源《{article.title}》提示 {family_label(family)} 相关线索。"
        f"候选方向偏置为 {direction_bias}，适合先映射到现有多因子参数做离线验证。"
    )


def expected_regime_for_text(text: str) -> str:
    if "库存" in text or "基差" in text:
        return "基本面线索明确、库存或基差持续变化的阶段"
    if "外盘" in text or "隔夜" in text:
        return "外盘波动对国内开盘形成传导的阶段"
    if "宏观" in text or "政策" in text:
        return "宏观预期或政策扰动主导风险偏好的阶段"
    return "公开信息持续发酵且价格波动放大的阶段"


def score_novelty(refs: list[SourceRef], text: str) -> float:
    base = 0.35 + min(len({r.source for r in refs}) * 0.12, 0.3)
    keyword_bonus = sum(word in text for word in ("库存", "外盘", "基差", "政策", "开工", "检修")) * 0.04
    return round(min(base + keyword_bonus, 0.9), 2)


def score_testability(symbols: list[str], params: dict[str, Any]) -> float:
    score = 0.45
    if symbols:
        score += 0.25
    if set(params).issubset(StrategyIdea.SAFE_PARAM_KEYS):
        score += 0.2
    if len(symbols) <= 3:
        score += 0.05
    return round(min(score, 0.95), 2)


def score_confidence(refs: list[SourceRef], symbols: list[str], analysis_mode: str) -> float:
    score = 0.35 + min(len(refs), 3) * 0.07 + min(len(symbols), 2) * 0.04
    if analysis_mode == "ollama":
        score += 0.08
    return round(min(score, 0.75), 2)


def idea_id(title: str, symbols: list[str], refs: list[SourceRef]) -> str:
    raw = "|".join([title, ",".join(symbols), ",".join(ref.id for ref in refs)])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def family_label(family: str) -> str:
    return {
        "gap_filter": "缺口过滤",
        "trend_breakout": "趋势延续",
        "mean_reversion": "均值回归",
        "regime_filter": "状态过滤",
    }.get(family, family)


def _safe_family(family: str) -> str:
    allowed = {"gap_filter", "trend_breakout", "mean_reversion", "regime_filter"}
    return family if family in allowed else "regime_filter"


def _safe_direction(direction: str) -> str:
    return direction if direction in {"long", "short", "neutral"} else "neutral"


def _safe_symbols(raw_symbols: Any, contracts: list[dict]) -> list[str]:
    allowed = {c.get("symbol") for c in contracts}
    if not isinstance(raw_symbols, list):
        return []
    return sorted({str(symbol).upper() for symbol in raw_symbols if str(symbol).upper() in allowed})


def _symbols_from_articles(articles: list[ResearchArticle], contracts: list[dict]) -> list[str]:
    symbols = sorted({s for article in articles for s in (article.symbols or extract_symbols(article.text_blob(), contracts))})
    return symbols


def _source_refs_for_symbols(articles: list[ResearchArticle], symbols: list[str]) -> list[SourceRef]:
    refs: list[SourceRef] = []
    for article in articles:
        article_symbols = article.symbols or []
        if any(symbol in article_symbols for symbol in symbols):
            refs.append(article.source_ref())
    return refs[:5]

