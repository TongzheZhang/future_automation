import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from automation.candidates import ideas_to_candidates, load_candidate, save_candidate
from automation.models import CandidatePlugin


def test_candidate_rejects_unsafe_params():
    candidate = CandidatePlugin(
        id="c1",
        idea_id="i1",
        created_at="2026-05-14T00:00:00Z",
        title="bad",
        symbols=["SA"],
        strategy_family="gap_filter",
        source_refs=[],
        params={"python_code": "print(1)"},
    )
    with pytest.raises(ValueError, match="unsafe"):
        candidate.validate()


def test_ideas_to_candidates_skips_existing_and_persists(tmp_path):
    idea = {
        "id": "idea-1",
        "title": "SA candidate",
        "symbols": ["SA"],
        "strategy_family": "gap_filter",
        "source_refs": [{"title": "src", "url": "https://example.com"}],
        "candidate_params": {"include_symbols": ["SA"], "min_confidence": 0.7},
    }
    candidates = ideas_to_candidates([idea], [], test_window={"start_date": "2024-01-01"})
    assert len(candidates) == 1
    path = save_candidate(candidates[0], tmp_path)
    loaded = load_candidate(path)
    assert loaded.idea_id == "idea-1"
    assert loaded.params["include_symbols"] == ["SA"]
    assert ideas_to_candidates([idea], [loaded]) == []
