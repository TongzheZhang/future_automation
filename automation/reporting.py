"""Markdown reports for automation runs."""
from __future__ import annotations

from pathlib import Path

from automation.models import ExperimentResult
from research.models import utc_now_iso


def write_experiment_report(path: Path, results: list[ExperimentResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Spark 候选策略实验报告",
        "",
        f"- 生成时间: {utc_now_iso()}",
        f"- 候选数: {len(results)}",
        "",
        "## 结果摘要",
        "",
    ]
    if not results:
        lines.append("本轮没有待测试候选。")
    for result in results:
        lines.extend(
            [
                f"### {result.candidate_id}",
                "",
                f"- 状态: {result.status}",
                f"- 结论: {result.reason}",
                f"- 交易数: {result.metrics.get('total_trades', 'N/A')}",
                f"- Sharpe: {result.metrics.get('sharpe', 'N/A')} (baseline {result.baseline_metrics.get('sharpe', 'N/A')})",
                f"- 胜率: {result.metrics.get('win_rate', 'N/A')}",
                f"- 最大回撤: {result.metrics.get('max_drawdown', 'N/A')}",
                "",
            ]
        )
    lines.extend(
        [
            "## 安全边界",
            "",
            "通过候选只标记为 ready_for_review，并发送提醒；不会自动进入生产信号。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_health_report(path: Path, checks: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Spark 自动化健康检查", "", f"- 时间: {utc_now_iso()}", ""]
    for name, check in checks.items():
        status = "OK" if check.get("ok") else "FAIL"
        lines.append(f"- {name}: {status} - {check.get('detail', '')}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
