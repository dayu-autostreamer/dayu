"""Analyze exported Dayu system logs from the command line."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

START_NODE = "_start"
END_NODE = "_end"
DEFAULT_SLO_SECONDS = 1.0
LATENCY_PERCENTILES = (50, 90, 95, 99)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a Dayu log export and print a compact execution summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log",
        required=True,
        metavar="LOG_FILE_PATH",
        help="Path to a log file exported from the Dayu frontend.",
    )
    parser.add_argument(
        "--output-format",
        choices=("text", "json", "full-json"),
        default="text",
        help="Choose a text summary, a compact JSON summary, or a full JSON report with per-task details.",
    )
    parser.add_argument(
        "--output-file",
        metavar="OUTPUT_PATH",
        help="Optional path to write the rendered report instead of printing the full content to stdout.",
    )
    parser.add_argument(
        "--slo-seconds",
        type=float,
        default=DEFAULT_SLO_SECONDS,
        help="End-to-end task latency SLO in seconds, used to compute compliance rate.",
    )
    return parser


def load_tasks(log_file: str | Path) -> list[dict[str, Any]]:
    log_path = Path(log_file).expanduser().resolve()

    if not log_path.exists():
        raise FileNotFoundError(f"Log file '{log_path}' does not exist.")
    if not log_path.is_file():
        raise ValueError(f"Log path '{log_path}' is not a file.")

    opener = gzip.open if log_path.suffix == ".gz" else open
    with opener(log_path, "rt", encoding="utf-8") as fh:
        sample = ""
        while True:
            char = fh.read(1)
            if not char or not char.isspace():
                sample = char
                break
        fh.seek(0)

        try:
            if sample == "[":
                records = json.load(fh)
            else:
                records = [json.loads(line) for line in fh if line.strip()]
        except json.JSONDecodeError as exc:
            raise ValueError(f"Log file '{log_path}' is not valid JSON.") from exc

    if not isinstance(records, list):
        raise ValueError(f"Log file '{log_path}' does not contain a task list.")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Log file '{log_path}' contains malformed task records.")

    return records


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _round_metric(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _percentile(values: Sequence[float], percentile: int) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(float(v) for v in values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight


def _build_latency_metrics(values: Sequence[float], *, slo_target_seconds: float | None = None) -> dict[str, Any]:
    values = [float(v) for v in values]
    metrics = {
        "count": len(values),
        "avg": 0.0,
        "min": 0.0,
        "max": 0.0,
    }
    for percentile in LATENCY_PERCENTILES:
        metrics[f"p{percentile}"] = 0.0

    if values:
        metrics.update(
            {
                "avg": _round_metric(sum(values) / len(values)),
                "min": _round_metric(min(values)),
                "max": _round_metric(max(values)),
            }
        )
        for percentile in LATENCY_PERCENTILES:
            metrics[f"p{percentile}"] = _round_metric(_percentile(values, percentile))

    if slo_target_seconds is not None:
        hit_count = sum(1 for value in values if value <= slo_target_seconds)
        metrics.update(
            {
                "slo_target_seconds": _round_metric(slo_target_seconds),
                "slo_hit_count": hit_count,
                "slo_miss_count": len(values) - hit_count,
                "slo_compliance_rate": round((hit_count / len(values)) if values else 0.0, 4),
            }
        )

    return metrics


def _iter_services(task: dict[str, Any]) -> list[dict[str, Any]]:
    dag = task.get("dag") or {}
    if not isinstance(dag, dict):
        raise ValueError("Task record is missing a valid 'dag' section.")

    services = []
    for service_name, node in dag.items():
        if service_name in {START_NODE, END_NODE}:
            continue

        service = (node or {}).get("service") or {}
        execute_data = service.get("execute_data") or {}
        execute_time = _safe_float(execute_data.get("execute_time"))
        real_execute_time = _safe_float(execute_data.get("real_execute_time"))
        transmit_time = _safe_float(execute_data.get("transmit_time"))
        services.append(
            {
                "name": service.get("service_name", service_name),
                "execute_device": service.get("execute_device") or "unknown",
                "execute_time": execute_time,
                "real_execute_time": real_execute_time,
                "transmit_time": transmit_time,
                "stage_latency": execute_time + transmit_time,
            }
        )

    return services


def build_task_details(tasks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    detailed_tasks = []

    for index, task in enumerate(tasks):
        services = _iter_services(task)
        execute_total = sum(service["execute_time"] for service in services)
        real_execute_total = sum(service["real_execute_time"] for service in services)
        transmit_total = sum(service["transmit_time"] for service in services)
        task_latency = execute_total + transmit_total

        task_detail = copy.deepcopy(task)
        task_detail["analysis"] = {
            "task_index": index,
            "service_count": len(services),
            "task_latency_seconds": _round_metric(task_latency),
            "execute_time_seconds": _round_metric(execute_total),
            "real_execute_time_seconds": _round_metric(real_execute_total),
            "transmit_time_seconds": _round_metric(transmit_total),
            "services": [
                {
                    **service,
                    "execute_time": _round_metric(service["execute_time"]),
                    "real_execute_time": _round_metric(service["real_execute_time"]),
                    "transmit_time": _round_metric(service["transmit_time"]),
                    "stage_latency": _round_metric(service["stage_latency"]),
                }
                for service in services
            ],
        }
        detailed_tasks.append(task_detail)

    return detailed_tasks


def summarize_tasks(tasks: Sequence[dict[str, Any]], slo_target_seconds: float = DEFAULT_SLO_SECONDS) -> dict[str, Any]:
    source_devices: Counter[str] = Counter()
    edge_devices: Counter[str] = Counter()
    service_rollup: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "occurrences": 0,
            "execute_times": [],
            "real_execute_times": [],
            "transmit_times": [],
            "stage_latencies": [],
            "execute_devices": Counter(),
        }
    )

    task_latencies: list[float] = []
    root_task_ids: set[str] = set()

    for index, task in enumerate(tasks):
        source_devices.update([str(task.get("source_device", "unknown"))])
        edge_devices.update(str(device) for device in task.get("all_edge_devices", []) if device)
        root_task_ids.add(str(task.get("root_uuid") or task.get("task_uuid") or f"task-{index}"))

        task_latency = 0.0
        for service in _iter_services(task):
            execute_time = service["execute_time"]
            real_execute_time = service["real_execute_time"]
            transmit_time = service["transmit_time"]
            stage_latency = service["stage_latency"]
            execute_device = service["execute_device"]

            task_latency += stage_latency

            rollup = service_rollup[service["name"]]
            rollup["occurrences"] += 1
            rollup["execute_times"].append(execute_time)
            rollup["real_execute_times"].append(real_execute_time)
            rollup["transmit_times"].append(transmit_time)
            rollup["stage_latencies"].append(stage_latency)
            rollup["execute_devices"].update([execute_device])

        task_latencies.append(task_latency)

    services: dict[str, dict[str, Any]] = {}
    for service_name in sorted(service_rollup):
        rollup = service_rollup[service_name]
        execute_metrics = _build_latency_metrics(rollup["execute_times"])
        real_execute_metrics = _build_latency_metrics(rollup["real_execute_times"])
        transmit_metrics = _build_latency_metrics(rollup["transmit_times"])
        stage_metrics = _build_latency_metrics(rollup["stage_latencies"])
        services[service_name] = {
            "occurrences": rollup["occurrences"],
            "avg_execute_time": execute_metrics["avg"],
            "avg_real_execute_time": real_execute_metrics["avg"],
            "avg_transmit_time": transmit_metrics["avg"],
            "avg_stage_latency": stage_metrics["avg"],
            "execute_devices": dict(sorted(rollup["execute_devices"].items())),
            "execute_time_seconds": execute_metrics,
            "real_execute_time_seconds": real_execute_metrics,
            "transmit_time_seconds": transmit_metrics,
            "stage_latency_seconds": stage_metrics,
        }

    task_latency_metrics = _build_latency_metrics(task_latencies, slo_target_seconds=slo_target_seconds)

    return {
        "task_count": len(tasks),
        "root_task_count": len(root_task_ids),
        "average_task_latency": task_latency_metrics["avg"],
        "slo_target_seconds": _round_metric(slo_target_seconds),
        "task_latency_seconds": task_latency_metrics,
        "source_devices": dict(sorted(source_devices.items())),
        "edge_devices": dict(sorted(edge_devices.items())),
        "services": services,
    }


def build_full_report(
    log_file: str | Path,
    tasks: Sequence[dict[str, Any]],
    *,
    slo_target_seconds: float = DEFAULT_SLO_SECONDS,
) -> dict[str, Any]:
    return {
        "log_file": str(Path(log_file).expanduser().resolve()),
        "summary": summarize_tasks(tasks, slo_target_seconds=slo_target_seconds),
        "tasks": build_task_details(tasks),
    }


def render_text_summary(log_file: str | Path, summary: dict[str, Any]) -> str:
    log_name = Path(log_file).name
    task_latency = summary["task_latency_seconds"]
    slo_rate = task_latency.get("slo_compliance_rate", 0.0) * 100.0
    lines = [
        "##################################################################",
        "###################### Dayu Log Analysis Tool ####################",
        f"Log file: {log_name}",
        f"Tasks analyzed: {summary['task_count']}",
        f"Unique root tasks: {summary['root_task_count']}",
        f"Average end-to-end task latency: {summary['average_task_latency']:.3f}s",
        (
            "Task latency percentiles: "
            f"P50={task_latency['p50']:.3f}s, "
            f"P90={task_latency['p90']:.3f}s, "
            f"P95={task_latency['p95']:.3f}s, "
            f"P99={task_latency['p99']:.3f}s"
        ),
        (
            "Task latency range: "
            f"min={task_latency['min']:.3f}s, "
            f"max={task_latency['max']:.3f}s"
        ),
        (
            "SLO compliance: "
            f"target={task_latency.get('slo_target_seconds', summary.get('slo_target_seconds', 0.0)):.3f}s, "
            f"rate={slo_rate:.2f}% "
            f"({task_latency.get('slo_hit_count', 0)}/{summary['task_count']})"
        ),
        "",
        "Source devices:",
    ]

    if summary["source_devices"]:
        lines.extend(f"  - {device}: {count}" for device, count in summary["source_devices"].items())
    else:
        lines.append("  - none")

    lines.extend(["", "Edge devices:"])
    if summary["edge_devices"]:
        lines.extend(f"  - {device}: {count}" for device, count in summary["edge_devices"].items())
    else:
        lines.append("  - none")

    lines.extend(["", "Service summary:"])
    if summary["services"]:
        for service_name, service_summary in summary["services"].items():
            devices = ", ".join(
                f"{device} ({count})" for device, count in service_summary["execute_devices"].items()
            )
            stage_metrics = service_summary["stage_latency_seconds"]
            lines.append(
                "  - "
                f"{service_name}: occurrences={service_summary['occurrences']}, "
                f"devices={devices or 'none'}, "
                f"avg_stage={service_summary['avg_stage_latency']:.3f}s, "
                f"p95_stage={stage_metrics['p95']:.3f}s, "
                f"avg_execute={service_summary['avg_execute_time']:.3f}s, "
                f"avg_real_execute={service_summary['avg_real_execute_time']:.3f}s, "
                f"avg_transmit={service_summary['avg_transmit_time']:.3f}s"
            )
    else:
        lines.append("  - none")

    lines.append("##################################################################")
    return "\n".join(lines)


def generate_report(
    log_file: str | Path,
    output_format: str = "text",
    *,
    slo_target_seconds: float = DEFAULT_SLO_SECONDS,
) -> str:
    tasks = load_tasks(log_file)
    summary = summarize_tasks(tasks, slo_target_seconds=slo_target_seconds)

    if output_format == "json":
        return json.dumps(summary, indent=2, sort_keys=True)
    if output_format == "full-json":
        return json.dumps(
            build_full_report(log_file, tasks, slo_target_seconds=slo_target_seconds),
            indent=2,
            ensure_ascii=False,
        )
    if output_format == "text":
        return render_text_summary(log_file, summary)

    raise ValueError(f"Unsupported output format: {output_format}")


def write_report(report: str, output_file: str | Path) -> Path:
    output_path = Path(output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        report = generate_report(
            args.log,
            output_format=args.output_format,
            slo_target_seconds=args.slo_seconds,
        )
        if args.output_file:
            output_path = write_report(report, args.output_file)
            print(f"Report written to {output_path}")
        else:
            print(report)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
