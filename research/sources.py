"""Network and fixture source collection for futures strategy research."""
from __future__ import annotations

import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

try:
    from loguru import logger
except ImportError:  # pragma: no cover - exercised only when optional deps are absent.
    import logging

    logger = logging.getLogger(__name__)

from research.models import ResearchArticle, utc_now_iso


DEFAULT_USER_AGENT = (
    "SparkResearchBot/0.1 (+local strategy research; no trading execution)"
)


def normalize_url(url: str) -> str:
    """Normalize URLs enough for stable deduplication."""
    if not url:
        return ""
    parsed = urllib.parse.urlsplit(url.strip())
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    query_items = [
        (k, v)
        for k, v in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in {"spm", "from"}
    ]
    query = urllib.parse.urlencode(sorted(query_items))
    return urllib.parse.urlunsplit((scheme, netloc, path, query, ""))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def article_fingerprint(title: str, url: str = "", published_at: str | None = None) -> str:
    """Return a stable article fingerprint."""
    base = normalize_url(url) if url else f"{normalize_text(title).lower()}|{published_at or ''}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


def dedupe_articles(
    articles: Iterable[ResearchArticle],
    existing_ids: set[str] | None = None,
) -> list[ResearchArticle]:
    seen = set(existing_ids or set())
    unique: list[ResearchArticle] = []
    for article in articles:
        if article.id in seen:
            continue
        seen.add(article.id)
        unique.append(article)
    return unique


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning(f"Skipping invalid JSONL line in {path}")
    return rows


def load_existing_article_ids(path: Path) -> set[str]:
    return {row.get("id") for row in read_jsonl(path) if row.get("id")}


def fetch_url(
    url: str,
    timeout: float = 12.0,
    user_agent: str = DEFAULT_USER_AGENT,
) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="replace")


class LinkAndTextParser(HTMLParser):
    """Tiny HTML parser for title, links, and readable text."""

    def __init__(self) -> None:
        super().__init__()
        self.in_title = False
        self.current_href: str | None = None
        self.current_link_text: list[str] = []
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self.in_title = True
        if tag.lower() == "a":
            attrs_dict = {k.lower(): v for k, v in attrs if v}
            self.current_href = attrs_dict.get("href")
            self.current_link_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self.in_title = False
        if tag.lower() == "a" and self.current_href:
            text = normalize_text(" ".join(self.current_link_text))
            if text:
                self.links.append((self.current_href, text))
            self.current_href = None
            self.current_link_text = []

    def handle_data(self, data: str) -> None:
        text = normalize_text(data)
        if not text:
            return
        if self.in_title:
            self.title_parts.append(text)
        if self.current_href:
            self.current_link_text.append(text)
        if len(text) > 1:
            self.text_parts.append(text)

    @property
    def title(self) -> str:
        return normalize_text(" ".join(self.title_parts))

    @property
    def text(self) -> str:
        return normalize_text(" ".join(self.text_parts))


def parse_html_page(url: str, source_name: str, html_text: str) -> ResearchArticle | None:
    parser = LinkAndTextParser()
    parser.feed(html_text)
    title = parser.title or source_name
    content = parser.text[:5000]
    if not title and not content:
        return None
    return ResearchArticle(
        id=article_fingerprint(title, url),
        title=title,
        url=url,
        source=source_name,
        fetched_at=utc_now_iso(),
        summary=content[:500],
        content=content,
    )


def parse_rss_feed(xml_text: str, source_name: str, feed_url: str, limit: int = 20) -> list[ResearchArticle]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning(f"RSS parse failed for {source_name}: {exc}")
        return []

    items = root.findall(".//item")
    if not items:
        items = root.findall(".//{http://www.w3.org/2005/Atom}entry")

    articles: list[ResearchArticle] = []
    for item in items[:limit]:
        title = _first_xml_text(item, ["title", "{http://www.w3.org/2005/Atom}title"])
        link = _first_xml_text(item, ["link"])
        if not link:
            atom_link = item.find("{http://www.w3.org/2005/Atom}link")
            link = atom_link.attrib.get("href", "") if atom_link is not None else ""
        published = _first_xml_text(
            item,
            ["pubDate", "published", "updated", "{http://www.w3.org/2005/Atom}updated"],
        )
        summary = _first_xml_text(
            item,
            ["description", "summary", "{http://www.w3.org/2005/Atom}summary"],
        )
        title = normalize_text(title)
        if not title:
            continue
        url = urllib.parse.urljoin(feed_url, link)
        article_id = article_fingerprint(title, url, published)
        articles.append(
            ResearchArticle(
                id=article_id,
                title=title,
                url=url,
                source=source_name,
                fetched_at=utc_now_iso(),
                published_at=normalize_text(published) or None,
                summary=normalize_text(summary),
                content=normalize_text(summary),
                metadata={"feed_url": feed_url},
            )
        )
    return articles


def _first_xml_text(node: ET.Element, names: list[str]) -> str:
    for name in names:
        child = node.find(name)
        if child is not None and child.text:
            return child.text
    return ""


def parse_search_results(html_text: str, query: str, source_name: str, base_url: str) -> list[ResearchArticle]:
    parser = LinkAndTextParser()
    parser.feed(html_text)
    articles: list[ResearchArticle] = []
    query_tokens = {token for token in re.split(r"\W+", query.lower()) if len(token) >= 2}

    for href, text in parser.links:
        url = _unwrap_search_url(urllib.parse.urljoin(base_url, href))
        if not url or _is_search_internal_url(url):
            continue
        lower_text = text.lower()
        if query_tokens and not any(token in lower_text for token in query_tokens):
            continue
        article_id = article_fingerprint(text, url)
        articles.append(
            ResearchArticle(
                id=article_id,
                title=text[:180],
                url=url,
                source=source_name,
                fetched_at=utc_now_iso(),
                summary=f"Search result for: {query}",
                metadata={"query": query},
            )
        )
    return dedupe_articles(articles)


def _unwrap_search_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    params = dict(urllib.parse.parse_qsl(parsed.query))
    for key in ("uddg", "url", "u"):
        if key in params and params[key].startswith(("http://", "https://")):
            return params[key]
    return url


def _is_search_internal_url(url: str) -> bool:
    host = urllib.parse.urlsplit(url).netloc.lower()
    return any(name in host for name in ("duckduckgo.com", "bing.com/search", "google.com/search"))


def build_queries(config: dict, contracts: list[dict]) -> list[str]:
    templates = config.get("query_templates", [])
    max_queries = int(config.get("max_queries_per_run", 8))
    queries: list[str] = []
    for template in templates:
        if "{name}" not in template and "{symbol}" not in template:
            queries.append(template)
            continue
        for contract in contracts:
            queries.append(
                template.format(
                    name=contract.get("name", contract.get("symbol", "")),
                    symbol=contract.get("symbol", ""),
                    exchange=contract.get("exchange", ""),
                )
            )
    return queries[:max_queries]


def collect_articles(
    config: dict,
    contracts: list[dict],
    limit: int | None = None,
    dry_run: bool = False,
) -> tuple[list[ResearchArticle], list[str]]:
    """Collect articles from configured sources, returning articles and errors."""
    source_cfg = config.get("sources", {})
    max_articles = limit or int(config.get("max_articles_per_run", 20))
    timeout = float(config.get("request_timeout_seconds", 12))
    user_agent = config.get("user_agent", DEFAULT_USER_AGENT)
    delay = float(config.get("rate_limit_seconds", 0.8))
    errors: list[str] = []

    if dry_run:
        return _dry_run_articles(contracts, max_articles), errors

    articles: list[ResearchArticle] = []

    for feed in source_cfg.get("rss_feeds", []):
        if len(articles) >= max_articles:
            break
        try:
            xml_text = fetch_url(feed["url"], timeout=timeout, user_agent=user_agent)
            articles.extend(parse_rss_feed(xml_text, feed.get("name", feed["url"]), feed["url"]))
            time.sleep(delay)
        except Exception as exc:  # noqa: BLE001 - source failures should not stop a run.
            msg = f"RSS source failed: {feed.get('name', feed.get('url'))}: {exc}"
            logger.warning(msg)
            errors.append(msg)

    for page in source_cfg.get("web_pages", []):
        if len(articles) >= max_articles:
            break
        try:
            html_text = fetch_url(page["url"], timeout=timeout, user_agent=user_agent)
            article = parse_html_page(page["url"], page.get("name", page["url"]), html_text)
            if article:
                articles.append(article)
            time.sleep(delay)
        except Exception as exc:  # noqa: BLE001
            msg = f"Web source failed: {page.get('name', page.get('url'))}: {exc}"
            logger.warning(msg)
            errors.append(msg)

    search_cfg = source_cfg.get("search", {})
    if search_cfg.get("enabled", True):
        url_template = search_cfg.get(
            "url_template",
            "https://duckduckgo.com/html/?q={query}",
        )
        source_name = search_cfg.get("name", "duckduckgo_html")
        base_url = search_cfg.get("base_url", "https://duckduckgo.com/")
        for query in build_queries(config, contracts):
            if len(articles) >= max_articles:
                break
            try:
                url = url_template.format(query=urllib.parse.quote_plus(query))
                html_text = fetch_url(url, timeout=timeout, user_agent=user_agent)
                articles.extend(parse_search_results(html_text, query, source_name, base_url))
                time.sleep(delay)
            except Exception as exc:  # noqa: BLE001
                msg = f"Search failed: {query}: {exc}"
                logger.warning(msg)
                errors.append(msg)

    return dedupe_articles(articles)[:max_articles], errors


def _dry_run_articles(contracts: list[dict], limit: int) -> list[ResearchArticle]:
    symbol_by_name = {c.get("name", c.get("symbol", "")): c.get("symbol", "") for c in contracts}
    samples = [
        (
            "纯碱库存下降叠加开工率波动，期限结构可能重新走强",
            "样本源",
            "https://example.local/research/sa-inventory",
            "纯碱现货库存连续下降，部分装置检修造成开工率回落。若基差同步走强，SA 可能出现短线趋势延续而非缺口回补。",
            ["SA"],
        ),
        (
            "原油外盘波动抬升，国内 SC 隔夜缺口需要分方向过滤",
            "样本源",
            "https://example.local/research/sc-overnight",
            "外盘原油波动放大时，国内原油开盘缺口更可能是信息传导，逆向回补策略需要提高做空阈值并缩短持仓观察窗口。",
            ["SC"],
        ),
        (
            "黑色链需求预期分化，螺纹和铁矿信号应加入宏观情绪过滤",
            "样本源",
            "https://example.local/research/black-chain",
            "地产和基建预期分化时，RB 与 I 的缺口回补表现可能不同。需要用成交量和趋势强度筛掉单边行情中的逆势信号。",
            ["RB", "I"],
        ),
    ]
    articles: list[ResearchArticle] = []
    for title, source, url, content, symbols in samples[:limit]:
        known_symbols = [s for s in symbols if s in symbol_by_name.values()]
        articles.append(
            ResearchArticle(
                id=article_fingerprint(title, url),
                title=title,
                url=url,
                source=source,
                fetched_at=utc_now_iso(),
                summary=content,
                content=content,
                symbols=known_symbols or symbols,
                metadata={"dry_run": True},
            )
        )
    return articles
