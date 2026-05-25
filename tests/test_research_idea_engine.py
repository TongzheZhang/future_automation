import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.idea_engine import generate_strategy_ideas
from research.models import ResearchArticle, utc_now_iso


def test_strategy_idea_required_fields_and_safe_params():
    contracts = [{"symbol": "SA", "name": "纯碱"}]
    article = ResearchArticle(
        id="a1",
        title="纯碱库存下降",
        url="https://example.com/sa",
        source="fixture",
        fetched_at=utc_now_iso(),
        content="纯碱库存下降，开工率回落，基差走强。",
        symbols=["SA"],
    )
    ideas = generate_strategy_ideas([article], {"mode": "fallback"}, contracts)
    assert len(ideas) == 1
    idea = ideas[0]
    idea.validate()
    assert idea.symbols == ["SA"]
    assert set(idea.candidate_params).issubset(idea.SAFE_PARAM_KEYS)
    assert 0 <= idea.confidence <= 1


def test_empty_articles_do_not_crash():
    assert generate_strategy_ideas([], {"mode": "fallback"}, [{"symbol": "SA", "name": "纯碱"}]) == []
