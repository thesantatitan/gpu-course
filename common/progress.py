from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
PROGRESS_DIR = "progress"
RUNS_FILE = "runs.jsonl"
SUMMARY_JSON_FILE = "summary.json"
SUMMARY_MD_FILE = "summary.md"

ASSIGNMENTS: dict[str, dict[str, str]] = {
    "smoke": {
        "id": "smoke",
        "title": "CUDA Smoke Test",
        "source": "tools/cuda-smoke/main.cu",
        "kind": "check",
    },
    "01": {
        "id": "01",
        "title": "Vector Add",
        "source": "assignments/01-vector-add/starter/main.cu",
        "kind": "assignment",
    },
    "02": {
        "id": "02",
        "title": "Grids And Matrix Multiply",
        "source": "assignments/02-grids-matmul/starter/main.cu",
        "kind": "assignment",
    },
    "03": {
        "id": "03",
        "title": "Memory Performance",
        "source": "assignments/03-memory-performance/starter/main.cu",
        "kind": "assignment",
    },
    "04": {
        "id": "04",
        "title": "Convolution And Stencil",
        "source": "assignments/04-convolution-stencil/starter/main.cu",
        "kind": "assignment",
    },
    "05": {
        "id": "05",
        "title": "Histogram And Reduction",
        "source": "assignments/05-histogram-reduction/starter/main.cu",
        "kind": "assignment",
    },
    "06": {
        "id": "06",
        "title": "Scan And Merge",
        "source": "assignments/06-scan-merge/starter/main.cu",
        "kind": "assignment",
    },
}

SOURCE_TO_ASSIGNMENT_ID = {
    assignment["source"]: assignment_id
    for assignment_id, assignment in ASSIGNMENTS.items()
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def git_metadata(root: Path) -> dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    commit = run_git(["rev-parse", "--short", "HEAD"])
    branch = run_git(["branch", "--show-current"])
    status = run_git(["status", "--porcelain"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
    }


def assignment_for_source(source: str) -> dict[str, str]:
    assignment_id = SOURCE_TO_ASSIGNMENT_ID.get(source)
    if assignment_id:
        return ASSIGNMENTS[assignment_id]

    return {
        "id": "custom",
        "title": Path(source).stem,
        "source": source,
        "kind": "custom",
    }


def parse_gpu_info(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {"raw": raw or ""}

    parts = [part.strip() for part in raw.split(",")]
    info: dict[str, Any] = {"raw": raw}
    if parts:
        info["name"] = parts[0]
    if len(parts) >= 2:
        info["driver_version"] = parts[1]
    if len(parts) >= 3:
        info["memory_total"] = parts[2]
        memory_number = parts[2].split()[0]
        if memory_number.isdigit():
            info["memory_total_mib"] = int(memory_number)
    return info


def timing_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    timings: dict[str, float] = {}
    for key, value in metrics.items():
        if key.endswith("_ms") and isinstance(value, (int, float)):
            timings[key] = float(value)
    return timings


def passed(record: dict[str, Any]) -> bool:
    metrics = record.get("metrics") or {}
    return record.get("assignment_exit_code") == 0 and metrics.get("ok") is True


def benchmark_signature(record: dict[str, Any]) -> str:
    payload = {
        "source": record.get("source"),
        "program_args": record.get("program_args", []),
        "gpu": (record.get("allocated_gpu") or {}).get("name")
        or record.get("requested_gpu"),
    }
    return json.dumps(payload, sort_keys=True)


def make_run_record(root: Path, modal_payload: dict[str, Any], run_stdout: str) -> dict[str, Any]:
    source = str(modal_payload.get("source") or "")
    assignment = assignment_for_source(source)
    assignment_exit_code = int(modal_payload.get("assignment_exit_code", -1))
    metrics = modal_payload.get("metrics") or {}
    program_args = modal_payload.get("program_args") or []
    recorded_at = now_utc()
    run_id_seed = json.dumps(
        {
            "recorded_at": recorded_at,
            "source": source,
            "program_args": program_args,
            "metrics": metrics,
            "stdout_hash": hashlib.sha256(run_stdout.encode()).hexdigest()[:16],
        },
        sort_keys=True,
    )
    run_id = hashlib.sha256(run_id_seed.encode()).hexdigest()[:12]

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "recorded_at": recorded_at,
        "runner": "modal",
        "assignment": assignment,
        "source": source,
        "program_args": program_args,
        "requested_gpu": modal_payload.get("requested_gpu"),
        "allocated_gpu": parse_gpu_info(modal_payload.get("allocated_gpu")),
        "cuda_image": modal_payload.get("cuda_image"),
        "remote_command": modal_payload.get("remote_command") or [],
        "assignment_exit_code": assignment_exit_code,
        "passed": assignment_exit_code == 0 and metrics.get("ok") is True,
        "metrics": metrics,
        "timings_ms": timing_metrics(metrics),
        "modal_run_url": modal_payload.get("modal_run_url"),
        "git": git_metadata(root),
    }
    record["benchmark_signature"] = benchmark_signature(record)
    return record


def read_runs(root: Path) -> list[dict[str, Any]]:
    runs_path = root / PROGRESS_DIR / RUNS_FILE
    if not runs_path.exists():
        return []

    runs: list[dict[str, Any]] = []
    for line in runs_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            runs.append(value)
    return runs


def append_run(root: Path, record: dict[str, Any]) -> None:
    progress_dir = root / PROGRESS_DIR
    progress_dir.mkdir(exist_ok=True)
    runs_path = progress_dir / RUNS_FILE
    with runs_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def compact_run(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": record.get("run_id"),
        "recorded_at": record.get("recorded_at"),
        "program_args": record.get("program_args", []),
        "requested_gpu": record.get("requested_gpu"),
        "allocated_gpu": record.get("allocated_gpu", {}),
        "assignment_exit_code": record.get("assignment_exit_code"),
        "passed": record.get("passed"),
        "metrics": record.get("metrics", {}),
        "timings_ms": record.get("timings_ms", {}),
        "git": record.get("git", {}),
        "modal_run_url": record.get("modal_run_url"),
    }


def build_summary(root: Path, runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in runs:
        assignment = record.get("assignment") or {}
        grouped[str(assignment.get("id", "custom"))].append(record)

    assignment_summaries: dict[str, Any] = {}
    assignment_ids = sorted(set(ASSIGNMENTS) | set(grouped))
    for assignment_id in assignment_ids:
        assignment_runs = grouped.get(assignment_id, [])
        if not assignment_runs:
            assignment = ASSIGNMENTS[assignment_id]
            assignment_summaries[assignment_id] = {
                "id": assignment_id,
                "title": assignment["title"],
                "kind": assignment["kind"],
                "source": assignment["source"],
                "done": False if assignment["kind"] == "assignment" else None,
                "runs": 0,
                "passed_runs": 0,
                "latest": {},
                "best_timings_ms": {},
            }
            continue

        assignment_runs = sorted(assignment_runs, key=lambda record: record.get("recorded_at", ""))
        latest = assignment_runs[-1]
        successful_runs = [record for record in assignment_runs if passed(record)]

        best_timings: dict[str, dict[str, Any]] = {}
        for record in successful_runs:
            for key, value in (record.get("timings_ms") or {}).items():
                existing = best_timings.get(key)
                if existing is None or value < existing["value"]:
                    best_timings[key] = {
                        "value": value,
                        "run_id": record.get("run_id"),
                        "program_args": record.get("program_args", []),
                        "recorded_at": record.get("recorded_at"),
                        "gpu": (record.get("allocated_gpu") or {}).get("name"),
                    }

        first = assignment_runs[0].get("assignment") or {}
        assignment_summaries[assignment_id] = {
            "id": assignment_id,
            "title": first.get("title", assignment_id),
            "kind": first.get("kind", "custom"),
            "source": first.get("source"),
            "done": bool(successful_runs) if first.get("kind") == "assignment" else None,
            "runs": len(assignment_runs),
            "passed_runs": len(successful_runs),
            "latest": compact_run(latest),
            "best_timings_ms": best_timings,
        }

    completed_assignments = sum(
        1
        for item in assignment_summaries.values()
        if item.get("kind") == "assignment" and item.get("done")
    )
    total_assignments = sum(1 for item in ASSIGNMENTS.values() if item.get("kind") == "assignment")

    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": now_utc(),
        "run_count": len(runs),
        "completed_assignments": completed_assignments,
        "total_assignments": total_assignments,
        "assignments": assignment_summaries,
    }


def format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def format_timings(timings: dict[str, Any]) -> str:
    if not timings:
        return "-"
    return ", ".join(f"{key}={float(value):.3f}" for key, value in sorted(timings.items()))


def format_best(best_timings: dict[str, Any]) -> str:
    if not best_timings:
        return "-"
    parts = []
    for key, value in sorted(best_timings.items()):
        parts.append(f"{key}={float(value['value']):.3f}")
    return ", ".join(parts)


def render_summary_markdown(summary: dict[str, Any]) -> str:
    rows = []
    for assignment_id, item in sorted(summary.get("assignments", {}).items()):
        latest = item.get("latest") or {}
        metrics = latest.get("metrics") or {}
        allocated_gpu = latest.get("allocated_gpu") or {}
        status = "done" if item.get("done") is True else "not done"
        if item.get("kind") != "assignment":
            if item.get("runs", 0) == 0:
                status = "not run"
            else:
                status = "check" if latest.get("passed") else "check failed"

        args = " ".join(str(arg) for arg in latest.get("program_args", [])) or "(defaults)"
        rows.append(
            "| {id} | {title} | {status} | {runs} | {passed} | {args} | {gpu} | {latest} | {best} |".format(
                id=assignment_id,
                title=item.get("title", ""),
                status=status,
                runs=item.get("runs", 0),
                passed=item.get("passed_runs", 0),
                args=args,
                gpu=allocated_gpu.get("name") or latest.get("requested_gpu") or "-",
                latest=format_timings(latest.get("timings_ms") or {}),
                best=format_best(item.get("best_timings_ms") or {}),
            )
        )

    lines = [
        "# Course Progress",
        "",
        f"Updated: `{summary.get('updated_at')}`",
        "",
        f"Assignments complete: **{summary.get('completed_assignments', 0)} / {summary.get('total_assignments', 0)}**",
        "",
        "| ID | Title | Status | Runs | Passed | Latest Args | GPU | Latest Timings (ms) | Best Timings (ms) |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |",
        *rows,
        "",
        "Raw run history is stored in `progress/runs.jsonl`. The machine-readable rollup is `progress/summary.json`.",
        "",
    ]
    return "\n".join(lines)


def write_summary(root: Path, summary: dict[str, Any]) -> None:
    progress_dir = root / PROGRESS_DIR
    progress_dir.mkdir(exist_ok=True)
    (progress_dir / SUMMARY_JSON_FILE).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (progress_dir / SUMMARY_MD_FILE).write_text(
        render_summary_markdown(summary),
        encoding="utf-8",
    )


def record_modal_run(root: Path, modal_payload: dict[str, Any], run_stdout: str) -> dict[str, Any]:
    record = make_run_record(root, modal_payload, run_stdout)
    append_run(root, record)
    summary = build_summary(root, read_runs(root))
    write_summary(root, summary)
    return record


def rebuild(root: Path) -> dict[str, Any]:
    summary = build_summary(root, read_runs(root))
    write_summary(root, summary)
    return summary


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    rebuilt = rebuild(repo_root)
    print(f"Rebuilt {PROGRESS_DIR}/{SUMMARY_MD_FILE} from {rebuilt['run_count']} runs.")
