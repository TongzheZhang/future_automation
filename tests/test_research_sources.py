import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.models import ResearchArticle, utc_now_iso
from research.sources import (
    article_fingerprint,
    collect_articles,
    dedupe_articles,
    normalize_url,
    parse_rss_feed,
)


def test_fingerprint_stable_ignores_tracking_params():
    url_a = "https://example.com/news/item?b=2&utm_source=x&a=1#frag"
    url_b = "https://example.com/news/item?a=1&b=2"
    assert normalize_url(url_a) == normalize_url(url_b)
    assert article_fingerprint("Title", url_a) == article_fingerprint("Title", url_b)


def test_dedupe_articles_removes_existing_ids():
    article = ResearchArticle(
        id="abc",
        title="测试",
        url="https://example.com/a",
        source="fixture",
        fetched_at=utc_now_iso(),
    )
    assert dedupe_articles([article, article]) == [article]
    assert dedupe_articles([article], existing_ids={"abc"}) == []


def test_parse_rss_feed_fixture():
    xml = """
    <rss><channel>
      <item>
        <title>纯碱库存下降</title>
        <link>https://example.com/sa</link>
        <description>库存变化可能影响期货价格。</description>
        <pubDate>Thu, 14 May 2026 08:00:00 GMT</pubDate>
      </item>
    </channel></rss>
    """
    articles = parse_rss_feed(xml, "fixture", "https://example.com/feed.xml")
    assert len(articles) == 1
    assert articles[0].title == "纯碱库存下降"
    assert articles[0].id


def test_collect_articles_dry_run_not_empty():
    contracts = [{"symbol": "SA", "name": "纯碱"}, {"symbol": "SC", "name": "原油"}]
    articles, errors = collect_articles({}, contracts, dry_run=True, limit=2)
    assert errors == []
    assert len(articles) == 2
    assert articles[0].symbols
