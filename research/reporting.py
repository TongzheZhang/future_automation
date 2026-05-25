"""Persistence and Markdown reporting for the research pipeline."""
from __future__ import annotations

import json
from pathlib import Path

from research.models import ResearchArticle, StrategyIdea, utc_now_iso


def append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_research_report(
    path: Path,
    articles: list[ResearchArticle],
    ideas: list[StrategyIdea],
    analysis: dict,
    errors: list[str],
    dry_run: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Spark 期货策略研究简报",
        "",
        f"- 生成时间: {utc_now_iso()}",
        f"- 模式: {'dry-run fixture' if dry_run else 'live sources'}",
        f"- 文章数: {len(articles)}",
        f"- 策略候选: {len(ideas)}",
        f"- 分析模式: {analysis.get('mode', 'unknown')}",
        "",
        "## 市场上下文",
        "",
        analysis.get("market_context", "暂无上下文。"),
        "",
    ]
    drivers = analysis.get("key_drivers") or []
    if drivers:
        lines.extend(["## 关键驱动", ""])
        lines.extend([f"- {driver}" for driver in drivers])
        lines.append("")

    if ideas:
        lines.extend(["## 策略候选", ""])
        for idea in ideas:
            lines.extend(
                [
                    f"### {idea.title}",
                    "",
                    f"- 品种: {', '.join(idea.symbols)}",
                    f"- 类型: {idea.strategy_family}",
                    f"- 状态: {idea.status}",
                    f"- 置信度: {idea.confidence:.0%}",
                    f"- 新颖度/可测性: {idea.novelty_score:.0%} / {idea.testability_score:.0%}",
                    f"- 假设: {idea.thesis}",
                    f"- 入场逻辑: {idea.entry_logic}",
                    f"- 出场逻辑: {idea.exit_logic}",
                    f"- 失效条件: {idea.invalidation}",
                    f"- 候选参数: `{json.dumps(idea.candidate_params, ensure_ascii=False, sort_keys=True)}`",
                    "- 来源: "
                    + "; ".join(f"[{ref.title}]({ref.url})" for ref in idea.source_refs),
                    "",
                ]
            )
    else:
        lines.extend(["## 策略候选", "", "本轮没有生成可测试候选。", ""])

    if articles:
        lines.extend(["## 来源摘要", ""])
        for article in articles[:12]:
            lines.append(f"- [{article.title}]({article.url}) - {article.source}")
        lines.append("")

    if errors:
        lines.extend(["## 采集错误", ""])
        lines.extend([f"- {error}" for error in errors[:20]])
        lines.append("")

    lines.extend(
        [
            "## 说明",
            "",
            "本报告只用于研究候选生成，不是交易建议，也不会自动进入生产信号。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")

