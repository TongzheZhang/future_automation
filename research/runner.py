"""Command-line entrypoint for the research pipeline."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

try:
    from loguru import logger
except ImportError:  # pragma: no cover - exercised only when optional deps are absent.
    import logging

    logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.idea_engine import generate_strategy_ideas
from research.llm import summarize_articles
from research.models import ResearchArticle, StrategyIdea
from research.reporting import append_jsonl, write_research_report, write_state
from research.sources import collect_articles, dedupe_articles, load_existing_article_ids


@dataclass
class ResearchRunResult:
    articles_fetched: int
    articles_new: int
    ideas_generated: int
    report_path: str
    articles_path: str
    ideas_path: str
    state_path: str
    dry_run: bool
    errors: list[str]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_contracts(path: Path) -> list[dict]:
    data = load_yaml(path)
    return data.get("contracts", [])


def run_research_cycle(
    contracts: list[dict],
    config: dict,
    dry_run: bool = False,
    limit: int | None = None,
) -> ResearchRunResult:
    """Run one research cycle and persist articles, ideas, report, and state."""
    output_dir = PROJECT_ROOT / config.get("output_dir", "research")
    cache_dir = output_dir / config.get("cache_dir", "cache")
    ideas_dir = output_dir / config.get("ideas_dir", "ideas")
    reports_dir = output_dir / config.get("reports_dir", "reports")
    state_path = output_dir / "state.json"
    articles_path = cache_dir / "articles.jsonl"
    ideas_path = ideas_dir / "strategy_ideas.jsonl"

    articles, errors = collect_articles(config, contracts, limit=limit, dry_run=dry_run)
    existing_ids = set() if dry_run else load_existing_article_ids(articles_path)
    new_articles = dedupe_articles(articles, existing_ids=existing_ids)

    analysis = summarize_articles(new_articles or articles, config)
    ideas = generate_strategy_ideas(new_articles or articles, analysis, contracts, config)

    append_jsonl(articles_path, [article.to_dict() for article in new_articles])
    append_jsonl(ideas_path, [idea.to_dict() for idea in ideas])

    report_name = f"research_{datetime.now().strftime('%Y%m%d')}.md"
    if dry_run:
        report_name = f"research_dry_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path = reports_dir / report_name
    write_research_report(report_path, new_articles or articles, ideas, analysis, errors, dry_run=dry_run)

    write_state(
        state_path,
        {
            "last_run_at": datetime.now().isoformat(timespec="seconds"),
            "dry_run": dry_run,
            "articles_fetched": len(articles),
            "articles_new": len(new_articles),
            "ideas_generated": len(ideas),
            "report_path": str(report_path),
            "analysis_mode": analysis.get("mode"),
            "errors": errors[-20:],
        },
    )

    logger.info(
        "Research cycle complete: "
        f"fetched={len(articles)}, new={len(new_articles)}, ideas={len(ideas)}, report={report_path}"
    )
    return ResearchRunResult(
        articles_fetched=len(articles),
        articles_new=len(new_articles),
        ideas_generated=len(ideas),
        report_path=str(report_path),
        articles_path=str(articles_path),
        ideas_path=str(ideas_path),
        state_path=str(state_path),
        dry_run=dry_run,
        errors=errors,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Spark futures strategy research cycle.")
    parser.add_argument("--once", action="store_true", help="Run one cycle. This is the default behavior.")
    parser.add_argument("--dry-run", action="store_true", help="Use built-in fixture sources and avoid network calls.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum articles to process in this run.")
    parser.add_argument("--config", default="config/research.yaml", help="Research config path.")
    parser.add_argument("--contracts", default="config/contracts.yaml", help="Contracts config path.")
    return parser


def main(argv: list[str] | None = None) -> ResearchRunResult:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_yaml(PROJECT_ROOT / args.config)
    contracts = load_contracts(PROJECT_ROOT / args.contracts)
    return run_research_cycle(
        contracts=contracts,
        config=config,
        dry_run=args.dry_run,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
