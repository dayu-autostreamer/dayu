"""Analyze exported Dayu system logs from the command line."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

START_NODE = "_start"
END_NODE = "_end"


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
        choices=("text", "json"),
        default="text",
        help="Choose a human-readable report or a machine-readable JSON summary.",
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
        services.append(
            {
                "name": service.get("service_name", service_name),
                "execute_device": service.get("execute_device") or "unknown",
                "execute_time": float(execute_data.get("execute_time") or 0.0),
                "real_execute_time": float(execute_data.get("real_execute_time") or 0.0),
                "transmit_time": float(execute_data.get("transmit_time") or 0.0),
            }
        )

    return services


def summarize_tasks(tasks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    source_devices: Counter[str] = Counter()
    edge_devices: Counter[str] = Counter()
    service_rollup: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "occurrences": 0,
            "execute_time_total": 0.0,
            "real_execute_time_total": 0.0,
            "transmit_time_total": 0.0,
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
            execute_device = service["execute_device"]

            task_latency += execute_time + transmit_time

            rollup = service_rollup[service["name"]]
            rollup["occurrences"] += 1
            rollup["execute_time_total"] += execute_time
            rollup["real_execute_time_total"] += real_execute_time
            rollup["transmit_time_total"] += transmit_time
            rollup["execute_devices"].update([execute_device])

        task_latencies.append(task_latency)

    services: dict[str, dict[str, Any]] = {}
    for service_name in sorted(service_rollup):
        rollup = service_rollup[service_name]
        occurrences = rollup["occurrences"] or 1
        services[service_name] = {
            "occurrences": rollup["occurrences"],
            "avg_execute_time": round(rollup["execute_time_total"] / occurrences, 3),
            "avg_real_execute_time": round(rollup["real_execute_time_total"] / occurrences, 3),
            "avg_transmit_time": round(rollup["transmit_time_total"] / occurrences, 3),
            "execute_devices": dict(sorted(rollup["execute_devices"].items())),
        }

    average_task_latency = round(sum(task_latencies) / len(task_latencies), 3) if task_latencies else 0.0

    return {
        "task_count": len(tasks),
        "root_task_count": len(root_task_ids),
        "average_task_latency": average_task_latency,
        "source_devices": dict(sorted(source_devices.items())),
        "edge_devices": dict(sorted(edge_devices.items())),
        "services": services,
    }


def render_text_summary(log_file: str | Path, summary: dict[str, Any]) -> str:
    log_name = Path(log_file).name
    lines = [
        "##################################################################",
        "###################### Dayu Log Analysis Tool ####################",
        f"Log file: {log_name}",
        f"Tasks analyzed: {summary['task_count']}",
        f"Unique root tasks: {summary['root_task_count']}",
        f"Average per-task service latency: {summary['average_task_latency']:.3f}s",
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
            lines.append(
                "  - "
                f"{service_name}: occurrences={service_summary['occurrences']}, "
                f"devices={devices or 'none'}, "
                f"avg_execute={service_summary['avg_execute_time']:.3f}s, "
                f"avg_real_execute={service_summary['avg_real_execute_time']:.3f}s, "
                f"avg_transmit={service_summary['avg_transmit_time']:.3f}s"
            )
    else:
        lines.append("  - none")

    lines.append("##################################################################")
    return "\n".join(lines)


def generate_report(log_file: str | Path, output_format: str = "text") -> str:
    tasks = load_tasks(log_file)
    summary = summarize_tasks(tasks)

    if output_format == "json":
        return json.dumps(summary, indent=2, sort_keys=True)

    return render_text_summary(log_file, summary)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        report = generate_report(args.log, output_format=args.output_format)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
