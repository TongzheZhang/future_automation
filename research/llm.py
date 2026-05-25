"""Local Ollama integration with deterministic fallback analysis."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

try:
    from loguru import logger
except ImportError:  # pragma: no cover - exercised only when optional deps are absent.
    import logging

    logger = logging.getLogger(__name__)

from research.models import ResearchArticle


class OllamaError(RuntimeError):
    """Raised when Ollama cannot return usable JSON."""


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "qwen2.5:7b",
        timeout: float = 45.0,
        temperature: float = 0.2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

    def is_available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3) as response:
                return response.status == 200
        except Exception:
            return False

    def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc

        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Ollama returned invalid envelope: {exc}") from exc

        content = envelope.get("message", {}).get("content", "")
        parsed = parse_json_content(content)
        if parsed is None:
            raise OllamaError("Ollama response did not contain valid JSON")
        return parsed


def parse_json_content(content: str) -> dict[str, Any] | None:
    """Parse strict JSON, or extract the first object from chatty content."""
    if not content:
        return None
    try:
        value = json.loads(content)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", content, flags=re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def summarize_articles(
    articles: list[ResearchArticle],
    config: dict,
    client: OllamaClient | None = None,
) -> dict[str, Any]:
    """Summarize articles and extract strategy hypotheses with fallback."""
    llm_cfg = config.get("llm", {})
    if not articles:
        return fallback_analysis([], reason="no_articles")
    if not llm_cfg.get("enabled", True):
        return fallback_analysis(articles, reason="llm_disabled")

    client = client or OllamaClient(
        base_url=llm_cfg.get("base_url", "http://127.0.0.1:11434"),
        model=llm_cfg.get("model", "qwen2.5:7b"),
        timeout=float(llm_cfg.get("timeout_seconds", 45)),
        temperature=float(llm_cfg.get("temperature", 0.2)),
    )

    if not client.is_available():
        return fallback_analysis(articles, reason="ollama_unavailable")

    messages = [
        {
            "role": "system",
            "content": (
                "你是期货量化研究助理。只返回 JSON，不要输出交易建议。"
                "目标是把公开信息转成可回测的策略假设。"
            ),
        },
        {
            "role": "user",
            "content": _build_prompt(articles),
        },
    ]
    try:
        parsed = client.chat_json(messages)
    except Exception as exc:  # noqa: BLE001 - any LLM failure should degrade gracefully.
        logger.warning(f"LLM analysis fallback: {exc}")
        return fallback_analysis(articles, reason="ollama_invalid_json")
    if not isinstance(parsed, dict):
        return fallback_analysis(articles, reason="ollama_invalid_json")

    parsed.setdefault("mode", "ollama")
    parsed.setdefault("ideas", [])
    parsed.setdefault("key_drivers", [])
    parsed.setdefault("risks", [])
    parsed.setdefault("market_context", "")
    return parsed


def _build_prompt(articles: list[ResearchArticle]) -> str:
    article_blocks = []
    for idx, article in enumerate(articles[:8], start=1):
        article_blocks.append(
            f"[{idx}] {article.title}\n"
            f"source={article.source} url={article.url}\n"
            f"text={article.text_blob(1200)}"
        )
    return (
        "请基于以下公开信息，输出 JSON：\n"
        "{\n"
        '  "market_context": "一句话概括",\n'
        '  "key_drivers": ["驱动1", "驱动2"],\n'
        '  "risks": ["风险1"],\n'
        '  "ideas": [\n'
        "    {\n"
        '      "title": "策略假设标题",\n'
        '      "symbols": ["SA"],\n'
        '      "strategy_family": "gap_filter|trend_breakout|mean_reversion|regime_filter",\n'
        '      "thesis": "为什么可能有效",\n'
        '      "direction_bias": "long|short|neutral",\n'
        '      "expected_regime": "适用行情",\n'
        '      "invalidation": "失效条件"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        + "\n\n".join(article_blocks)
    )


def fallback_analysis(articles: list[ResearchArticle], reason: str) -> dict[str, Any]:
    drivers = _keyword_drivers(articles)
    symbols = sorted({symbol for article in articles for symbol in article.symbols})
    context = "公开信息不足，使用关键词规则生成研究候选。"
    if drivers:
        context = "，".join(drivers[:3]) + " 相关线索值得跟踪。"
    return {
        "mode": "fallback",
        "fallback_reason": reason,
        "market_context": context,
        "key_drivers": drivers,
        "risks": ["LLM 不可用或信息源不足，候选策略需要回测验证"],
        "symbol_hints": symbols,
        "ideas": [],
    }


def _keyword_drivers(articles: list[ResearchArticle]) -> list[str]:
    text = " ".join(article.text_blob(1000) for article in articles)
    mapping = [
        ("库存", "库存变化"),
        ("开工", "开工率变化"),
        ("检修", "装置检修"),
        ("基差", "基差变化"),
        ("外盘", "外盘传导"),
        ("政策", "政策扰动"),
        ("需求", "需求预期"),
        ("宏观", "宏观情绪"),
        ("波动", "波动率抬升"),
    ]
    return [label for key, label in mapping if key in text]
