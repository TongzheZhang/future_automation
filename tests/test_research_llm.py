import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.llm import parse_json_content, summarize_articles
from research.models import ResearchArticle, utc_now_iso


class FakeInvalidClient:
    def is_available(self):
        return True

    def chat_json(self, messages):
        raise RuntimeError("bad json")


def test_parse_json_content_accepts_strict_and_embedded_json():
    assert parse_json_content('{"ok": true}') == {"ok": True}
    assert parse_json_content('prefix {"ok": true, "n": 1} suffix') == {"ok": True, "n": 1}


def test_summarize_articles_invalid_ollama_falls_back():
    article = ResearchArticle(
        id="a",
        title="原油外盘波动",
        url="https://example.com/sc",
        source="fixture",
        fetched_at=utc_now_iso(),
        content="外盘原油波动抬升，国内开盘缺口需要过滤。",
        symbols=["SC"],
    )
    result = summarize_articles([article], {"llm": {"enabled": True}}, client=FakeInvalidClient())
    assert result["mode"] == "fallback"
    assert result["market_context"]
    assert "risks" in result
