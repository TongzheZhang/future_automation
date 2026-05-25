"""Candidate plugin persistence and conversion from research ideas."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import yaml

from automation.models import CandidatePlugin
from research.models import utc_now_iso


def candidate_id_for_idea(idea: dict) -> str:
    raw = "|".join(
        [
            str(idea.get("id", "")),
            str(idea.get("title", "")),
            ",".join(idea.get("symbols", []) or []),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def load_ideas(path: Path) -> list[dict]:
    if not path.exists():
        return []
    ideas: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ideas.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return ideas


def load_candidate(path: Path) -> CandidatePlugin:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return CandidatePlugin.from_dict(data)


def load_candidates(directory: Path) -> list[CandidatePlugin]:
    if not directory.exists():
        return []
    candidates: list[CandidatePlugin] = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            candidates.append(load_candidate(path))
        except ValueError:
            continue
    return candidates


def save_candidate(candidate: CandidatePlugin, directory: Path) -> Path:
    candidate.validate()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{candidate.id}.yaml"
    path.write_text(
        yaml.safe_dump(candidate.to_dict(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def ideas_to_candidates(
    ideas: Iterable[dict],
    existing: Iterable[CandidatePlugin],
    test_window: dict | None = None,
) -> list[CandidatePlugin]:
    existing_idea_ids = {candidate.idea_id for candidate in existing}
    candidates: list[CandidatePlugin] = []
    for idea in ideas:
        idea_id = str(idea.get("id", ""))
        if not idea_id or idea_id in existing_idea_ids:
            continue
        params = dict(idea.get("candidate_params", {}) or {})
        candidate = CandidatePlugin(
            id=candidate_id_for_idea(idea),
            idea_id=idea_id,
            created_at=utc_now_iso(),
            title=str(idea.get("title", "")),
            symbols=list(idea.get("symbols", []) or []),
            strategy_family=str(idea.get("strategy_family", "regime_filter")),
            source_refs=list(idea.get("source_refs", []) or []),
            params=params,
            status="new",
            test_window=dict(test_window or {}),
        )
        candidate.validate()
        candidates.append(candidate)
        existing_idea_ids.add(idea_id)
    return candidates


def update_candidate_status(
    candidate: CandidatePlugin,
    directory: Path,
    status: str,
    validation: dict,
) -> Path:
    candidate.status = status
    candidate.validation = validation
    return save_candidate(candidate, directory)
