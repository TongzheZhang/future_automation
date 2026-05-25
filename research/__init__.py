"""Research pipeline for strategy idea discovery."""

from research.models import StrategyIdea

__all__ = ["ResearchRunResult", "StrategyIdea", "run_research_cycle"]


def __getattr__(name):
    if name in {"ResearchRunResult", "run_research_cycle"}:
        from research.runner import ResearchRunResult, run_research_cycle

        return {"ResearchRunResult": ResearchRunResult, "run_research_cycle": run_research_cycle}[name]
    raise AttributeError(name)
