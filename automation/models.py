"""Data models for automation orchestration and candidate experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from research.models import StrategyIdea, utc_now_iso


SAFE_CANDIDATE_PARAM_KEYS = StrategyIdea.SAFE_PARAM_KEYS


@dataclass
class CandidatePlugin:
    """Declarative candidate strategy plugin generated from a research idea."""

    id: str
    idea_id: str
    created_at: str
    title: str
    symbols: list[str]
    strategy_family: str
    source_refs: list[dict[str, Any]]
    params: dict[str, Any]
    status: str = "new"
    test_window: dict[str, str | None] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)

    VALID_STATUSES = {
        "new",
        "testing",
        "ready_for_review",
        "rejected",
        "needs_more_data",
        "archived",
    }

    def validate(self) -> None:
        if not self.id:
            raise ValueError("candidate id is required")
        if not self.idea_id:
            raise ValueError("idea_id is required")
        if not self.symbols:
            raise ValueError("candidate symbols must not be empty")
        unsafe = set(self.params) - SAFE_CANDIDATE_PARAM_KEYS
        if unsafe:
            raise ValueError(f"unsafe candidate params: {', '.join(sorted(unsafe))}")
        if self.status not in self.VALID_STATUSES:
            raise ValueError(f"unsupported candidate status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidatePlugin":
        plugin = cls(
            id=str(data.get("id", "")),
            idea_id=str(data.get("idea_id", "")),
            created_at=str(data.get("created_at", utc_now_iso())),
            title=str(data.get("title", "")),
            symbols=list(data.get("symbols", []) or []),
            strategy_family=str(data.get("strategy_family", "regime_filter")),
            source_refs=list(data.get("source_refs", []) or []),
            params=dict(data.get("params", {}) or {}),
            status=str(data.get("status", "new")),
            test_window=dict(data.get("test_window", {}) or {}),
            validation=dict(data.get("validation", {}) or {}),
        )
        plugin.validate()
        return plugin


@dataclass
class ExperimentResult:
    candidate_id: str
    idea_id: str
    status: str
    reason: str
    metrics: dict[str, Any]
    baseline_metrics: dict[str, Any]
    passed: bool
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AutomationRunResult:
    task: str
    status: str
    started_at: str
    finished_at: str
    artifacts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    sent: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
