import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_daily_run_reuses_main_v3_config():
    import daily_run
    import main_v3

    assert daily_run.V3_CONFIG is main_v3.V3_CONFIG
    assert daily_run.generate_signals is main_v3.generate_signals


def test_weekly_dry_run_composes_research_and_experiment(monkeypatch, tmp_path):
    import automation.orchestrator as orchestrator
    from automation.models import AutomationRunResult

    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", tmp_path)

    def fake_research(config, dry_run=False, send=True):
        return AutomationRunResult(
            task="research",
            status="ok",
            started_at="s",
            finished_at="f",
            artifacts=["research.md"],
            errors=[],
        )

    def fake_experiment(config, dry_run=False, send=True):
        return AutomationRunResult(
            task="experiment",
            status="ok",
            started_at="s",
            finished_at="f",
            artifacts=["experiment.md"],
            errors=[],
        )

    monkeypatch.setattr(orchestrator, "run_research", fake_research)
    monkeypatch.setattr(orchestrator, "run_experiment", fake_experiment)
    result = orchestrator.run_weekly(
        {"output_dir": "automation", "research_dir": "research", "legacy_evolution": {"enabled": True}},
        dry_run=True,
        send=False,
    )
    assert result.status == "ok"
    assert any("weekly_" in artifact for artifact in result.artifacts)
    assert (tmp_path / "automation" / "state.json").exists()
