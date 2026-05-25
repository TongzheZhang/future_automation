"""Unified CLI for autonomous research, experiments, signals, and health checks."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from automation.candidates import ideas_to_candidates, load_candidates, load_ideas, save_candidate
from automation.experiments import make_dry_run_data, run_candidate_experiments
from automation.models import AutomationRunResult
from automation.reporting import write_experiment_report, write_health_report
from data.collectors.akshare_adapter import get_all_contracts_data
from main_v3 import generate_signals, load_config as load_main_config
from research.runner import load_yaml, run_research_cycle
from research.models import utc_now_iso
from signals.feishu_sender import send_feishu_message, send_signal_report
from signals.factors import create_v3_scorer


def load_automation_config(path: Path | None = None) -> dict[str, Any]:
    path = path or PROJECT_ROOT / "config" / "automation.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def output_paths(config: dict[str, Any]) -> dict[str, Path]:
    automation_dir = PROJECT_ROOT / config.get("output_dir", "automation")
    research_dir = PROJECT_ROOT / config.get("research_dir", "research")
    return {
        "automation_dir": automation_dir,
        "reports_dir": automation_dir / "reports",
        "state_path": automation_dir / "state.json",
        "ideas_path": research_dir / "ideas" / "strategy_ideas.jsonl",
        "candidates_dir": research_dir / "candidates",
        "experiments_path": research_dir / "experiments" / "candidate_experiments.jsonl",
    }


def save_run_state(paths: dict[str, Path], result: AutomationRunResult) -> None:
    state_path = paths["state_path"]
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def finish_result(
    paths: dict[str, Path],
    task: str,
    started_at: str,
    status: str,
    artifacts: list[str],
    errors: list[str],
    sent: bool = False,
) -> AutomationRunResult:
    result = AutomationRunResult(
        task=task,
        status=status,
        started_at=started_at,
        finished_at=utc_now_iso(),
        artifacts=artifacts,
        errors=errors,
        sent=sent,
    )
    save_run_state(paths, result)
    return result


def run_daily(config: dict[str, Any], dry_run: bool = False, send: bool = True) -> AutomationRunResult:
    started_at = utc_now_iso()
    paths = output_paths(config)
    artifacts: list[str] = []
    errors: list[str] = []
    sent = False
    try:
        _, contracts = load_main_config()
        data = make_dry_run_data(contracts) if dry_run else get_all_contracts_data(contracts, use_cache=True)
        scorer = create_v3_scorer()
        generator, signals = generate_signals(data, contracts, scorer)
        report = generator.format_report(signals)
        report_path = paths["reports_dir"] / f"daily_{datetime.now().strftime('%Y%m%d')}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        artifacts.append(str(report_path))
        if send and not dry_run:
            sent = send_signal_report(report)
        return finish_result(paths, "daily", started_at, "ok", artifacts, errors, sent)
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
        return finish_result(paths, "daily", started_at, "error", artifacts, errors, sent)


def run_research(config: dict[str, Any], dry_run: bool = False, send: bool = True) -> AutomationRunResult:
    started_at = utc_now_iso()
    paths = output_paths(config)
    artifacts: list[str] = []
    errors: list[str] = []
    sent = False
    try:
        _, contracts = load_main_config()
        research_cfg = load_yaml(PROJECT_ROOT / config.get("research_config", "config/research.yaml"))
        result = run_research_cycle(contracts, research_cfg, dry_run=dry_run)
        artifacts.extend([result.report_path, result.ideas_path])
        errors.extend(result.errors)
        if send and not dry_run:
            sent = send_feishu_message(
                f"Spark 研究任务完成\n新增文章: {result.articles_new}\n策略想法: {result.ideas_generated}\n报告: {result.report_path}"
            )
        return finish_result(paths, "research", started_at, "ok", artifacts, errors, sent)
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
        return finish_result(paths, "research", started_at, "error", artifacts, errors, sent)


def materialize_candidates(config: dict[str, Any], reset_existing: bool = False) -> list[str]:
    paths = output_paths(config)
    ideas = load_ideas(paths["ideas_path"])
    existing = [] if reset_existing else load_candidates(paths["candidates_dir"])
    test_window = config.get("experiments", {}).get("test_window", {})
    candidates = ideas_to_candidates(ideas, existing, test_window=test_window)
    saved: list[str] = []
    for candidate in candidates:
        saved.append(str(save_candidate(candidate, paths["candidates_dir"])))
    return saved


def run_experiment(config: dict[str, Any], dry_run: bool = False, send: bool = True) -> AutomationRunResult:
    started_at = utc_now_iso()
    paths = output_paths(config)
    artifacts: list[str] = []
    errors: list[str] = []
    sent = False
    try:
        saved = materialize_candidates(config, reset_existing=dry_run)
        artifacts.extend(saved)
        _, contracts = load_main_config()
        data = make_dry_run_data(contracts) if dry_run else get_all_contracts_data(contracts, use_cache=True)
        thresholds = config.get("experiments", {}).get("acceptance", {})
        results = run_candidate_experiments(
            paths["candidates_dir"],
            paths["experiments_path"],
            data,
            contracts,
            thresholds,
            dry_run=dry_run,
        )
        report_path = paths["reports_dir"] / f"experiment_{datetime.now().strftime('%Y%m%d')}.md"
        write_experiment_report(report_path, results)
        artifacts.extend([str(paths["experiments_path"]), str(report_path)])
        ready = sum(1 for result in results if result.status == "ready_for_review")
        if send and not dry_run:
            sent = send_feishu_message(
                f"Spark 候选实验完成\n新增候选: {len(saved)}\n已测试: {len(results)}\n可人工复核: {ready}\n报告: {report_path}"
            )
        return finish_result(paths, "experiment", started_at, "ok", artifacts, errors, sent)
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
        return finish_result(paths, "experiment", started_at, "error", artifacts, errors, sent)


def run_weekly(config: dict[str, Any], dry_run: bool = False, send: bool = True) -> AutomationRunResult:
    started_at = utc_now_iso()
    paths = output_paths(config)
    artifacts: list[str] = []
    errors: list[str] = []
    research_result = run_research(config, dry_run=dry_run, send=False)
    experiment_result = run_experiment(config, dry_run=dry_run, send=False)
    artifacts.extend(research_result.artifacts)
    artifacts.extend(experiment_result.artifacts)
    errors.extend(research_result.errors)
    errors.extend(experiment_result.errors)
    evolution_summary = "skipped"
    evolution_cfg = config.get("legacy_evolution", {})
    if evolution_cfg.get("enabled", True) and not dry_run:
        try:
            _, contracts = load_main_config()
            from evolution.engine import STATE_FILE, run_evolution_cycle

            changes = run_evolution_cycle(contracts)
            evolution_summary = f"changes={len(changes)}"
            artifacts.append(str(STATE_FILE))
        except Exception as exc:  # noqa: BLE001
            evolution_summary = f"error={exc}"
            errors.append(f"legacy_evolution: {exc}")
    status = "ok" if research_result.status == "ok" and experiment_result.status == "ok" else "error"
    report_path = paths["reports_dir"] / f"weekly_{datetime.now().strftime('%Y%m%d')}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Spark 周度自治报告",
                "",
                f"- 时间: {utc_now_iso()}",
                f"- 研究状态: {research_result.status}",
                f"- 实验状态: {experiment_result.status}",
                f"- 进化检查: {evolution_summary}",
                "",
                "## 产物",
                *[f"- {artifact}" for artifact in artifacts],
                "",
                "## 错误",
                *([f"- {error}" for error in errors] if errors else ["- 无"]),
                "",
                "说明：候选策略只进入人工复核，不自动启用生产信号。",
            ]
        ),
        encoding="utf-8",
    )
    artifacts.append(str(report_path))
    sent = False
    if send and not dry_run:
        sent = send_feishu_message(f"Spark 周度自治报告\n状态: {status}\n报告: {report_path}")
    return finish_result(paths, "weekly", started_at, status, artifacts, errors, sent)


def run_health(config: dict[str, Any], dry_run: bool = False, send: bool = True) -> AutomationRunResult:
    started_at = utc_now_iso()
    paths = output_paths(config)
    checks: dict[str, dict[str, Any]] = {}
    checks["openclaw"] = {
        "ok": bool(shutil.which("openclaw")) or dry_run,
        "detail": "openclaw command available" if shutil.which("openclaw") else "openclaw command not found",
    }
    try:
        from research.llm import OllamaClient

        client = OllamaClient()
        ok = client.is_available()
        checks["ollama"] = {"ok": ok or dry_run, "detail": "ollama available" if ok else "ollama unavailable"}
    except Exception as exc:  # noqa: BLE001
        checks["ollama"] = {"ok": dry_run, "detail": str(exc)}

    storage = PROJECT_ROOT / "data" / "storage"
    daily_files = list(storage.glob("*_daily.parquet")) if storage.exists() else []
    checks["data_cache"] = {
        "ok": bool(daily_files),
        "detail": f"{len(daily_files)} daily parquet files",
    }
    state_path = paths["state_path"]
    checks["automation_state"] = {
        "ok": state_path.exists() or dry_run,
        "detail": str(state_path) if state_path.exists() else "state missing",
    }
    report_path = paths["reports_dir"] / f"health_{datetime.now().strftime('%Y%m%d')}.md"
    write_health_report(report_path, checks)
    errors = [f"{k}: {v['detail']}" for k, v in checks.items() if not v.get("ok")]
    sent = False
    if send and errors and not dry_run:
        sent = send_feishu_message("Spark 健康检查异常\n" + "\n".join(errors))
    status = "ok" if not errors else "warning"
    return finish_result(paths, "health", started_at, status, [str(report_path)], errors, sent)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spark autonomous orchestration CLI.")
    parser.add_argument("task", choices=["daily", "research", "experiment", "weekly", "health"])
    parser.add_argument("--config", default="config/automation.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-send", action="store_true", help="Do not send Feishu messages.")
    return parser


def main(argv: list[str] | None = None) -> AutomationRunResult:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_automation_config(PROJECT_ROOT / args.config)
    send = not args.no_send
    if args.task == "daily":
        return run_daily(config, dry_run=args.dry_run, send=send)
    if args.task == "research":
        return run_research(config, dry_run=args.dry_run, send=send)
    if args.task == "experiment":
        return run_experiment(config, dry_run=args.dry_run, send=send)
    if args.task == "weekly":
        return run_weekly(config, dry_run=args.dry_run, send=send)
    if args.task == "health":
        return run_health(config, dry_run=args.dry_run, send=send)
    raise ValueError(args.task)


if __name__ == "__main__":
    main()
