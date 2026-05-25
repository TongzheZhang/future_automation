import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.runner import run_research_cycle


def test_dry_run_generates_report_and_idea(tmp_path, monkeypatch):
    import research.runner as runner

    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    contracts = [{"symbol": "SA", "name": "纯碱"}, {"symbol": "SC", "name": "原油"}]
    config = {
        "output_dir": "research",
        "llm": {"enabled": False},
        "max_ideas_per_run": 3,
    }
    result = run_research_cycle(contracts, config, dry_run=True, limit=2)
    assert result.articles_fetched == 2
    assert result.ideas_generated >= 1
    assert Path(result.report_path).exists()
    assert Path(result.ideas_path).exists()
