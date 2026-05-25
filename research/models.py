"""Shared data models for the research pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class SourceRef:
    """Compact reference to a source article."""

    id: str
    title: str
    url: str
    source: str
    published_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchArticle:
    """Normalized article/search result used by the research engine."""

    id: str
    title: str
    url: str
    source: str
    fetched_at: str
    published_at: str | None = None
    summary: str = ""
    content: str = ""
    symbols: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def source_ref(self) -> SourceRef:
        return SourceRef(
            id=self.id,
            title=self.title,
            url=self.url,
            source=self.source,
            published_at=self.published_at,
        )

    def text_blob(self, max_chars: int = 1800) -> str:
        text = "\n".join(part for part in [self.title, self.summary, self.content] if part)
        return text[:max_chars]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyIdea:
    """Structured strategy hypothesis, not a production trading signal."""

    id: str
    created_at: str
    title: str
    symbols: list[str]
    strategy_family: str
    thesis: str
    source_refs: list[SourceRef]
    entry_logic: str
    filters: list[str]
    exit_logic: str
    risk_controls: list[str]
    candidate_params: dict[str, Any]
    expected_regime: str
    invalidation: str
    novelty_score: float
    testability_score: float
    confidence: float
    status: str = "candidate"

    REQUIRED_FIELDS = (
        "id",
        "created_at",
        "title",
        "symbols",
        "strategy_family",
        "thesis",
        "source_refs",
        "entry_logic",
        "filters",
        "exit_logic",
        "risk_controls",
        "candidate_params",
        "expected_regime",
        "invalidation",
        "novelty_score",
        "testability_score",
        "confidence",
        "status",
    )

    SAFE_PARAM_KEYS = {
        "include_symbols",
        "exclude_symbols",
        "min_gap_pct",
        "min_confidence",
        "hold_minutes",
        "atr_stop_mult",
        "atr_tp_mult",
        "factor_weight_overrides",
        "direction_bias",
    }

    def validate(self) -> None:
        missing = [name for name in self.REQUIRED_FIELDS if getattr(self, name, None) in (None, "")]
        if missing:
            raise ValueError(f"StrategyIdea missing required fields: {', '.join(missing)}")
        if not self.symbols:
            raise ValueError("StrategyIdea.symbols must contain at least one symbol")
        unsafe = set(self.candidate_params) - self.SAFE_PARAM_KEYS
        if unsafe:
            raise ValueError(f"Unsafe candidate_params keys: {', '.join(sorted(unsafe))}")
        for name in ("novelty_score", "testability_score", "confidence"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")
        if self.status not in {"candidate", "testing", "archived", "rejected"}:
            raise ValueError(f"Unsupported status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_refs"] = [ref.to_dict() for ref in self.source_refs]
        return data
