"""Analyze TPU availability from jobman timeline files."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .utils import jobman_log_dir


def _parse_time(t: str) -> datetime:
    return datetime.fromisoformat(t)


def _analyze_worker(timeline_path: Path, now: datetime) -> dict | None:
    """Parse a single worker's timeline and return phase breakdown."""
    events = []
    for line in timeline_path.read_text().splitlines():
        if line.strip():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not events:
        return None

    accel = zone = None
    for e in events:
        if e.get("event") == "worker_started":
            accel = e.get("accelerator", "unknown")
            zone = e.get("zone", "unknown")
            break
    if not accel:
        return None

    phases: dict[str, float] = defaultdict(float)
    task_count = preemption_count = failure_count = 0
    cur_phase: str | None = None
    phase_start: datetime | None = None
    first_time = last_time = None

    def close_phase(t: datetime) -> None:
        nonlocal phase_start
        if cur_phase and phase_start:
            phases[cur_phase] += (t - phase_start).total_seconds()
        phase_start = t

    def set_phase(t: datetime, phase: str) -> None:
        nonlocal cur_phase
        close_phase(t)
        cur_phase = phase

    def _loop_exception_phase(error: str) -> str:
        err = error.lower()
        if "quota" in err or "resource_exhausted" in err:
            return "quota_error"
        if "httperror" in err or "network" in err or "connection" in err:
            return "api_error"
        return "error_recovery"

    for e in events:
        t = _parse_time(e["time"])
        if first_time is None:
            first_time = t
        last_time = t
        ev = e["event"]

        if ev == "worker_started":
            set_phase(t, "idle")
        elif ev == "tpu_requesting":
            set_phase(t, "requesting")
        elif ev == "tpu_status":
            status = e.get("status", "")
            if status == "WAITING_FOR_RESOURCES":
                set_phase(t, "waiting_for_resources")
            elif status in ("PROVISIONING", "CREATING"):
                set_phase(t, "provisioning")
            elif status in ("PREEMPTED", "SUSPENDED", "TERMINATED"):
                set_phase(t, "preempted")
                preemption_count += 1
            elif status == "FAILED":
                set_phase(t, "failed")
                failure_count += 1
        elif ev == "tpu_ready":
            set_phase(t, "ready_idle")
        elif ev == "bootstrap_started":
            set_phase(t, "bootstrapping")
        elif ev == "bootstrap_succeeded":
            set_phase(t, "ready_idle")
        elif ev == "bootstrap_failed":
            set_phase(t, "bootstrap_retry")
        elif ev == "task_started":
            set_phase(t, "running_task")
            task_count += 1
        elif ev in ("task_completed", "task_failed", "task_released"):
            set_phase(t, "ready_idle")
        elif ev == "tpu_delete_requested":
            set_phase(t, "deleting")
        elif ev == "tpu_deleted":
            set_phase(t, "deleted")
        elif ev == "worker_loop_exception":
            set_phase(t, _loop_exception_phase(str(e.get("error", ""))))

    # Close final phase up to now
    if cur_phase and phase_start:
        phases[cur_phase] += (now - phase_start).total_seconds()

    return {
        "accel": accel,
        "zone": zone,
        "phases": dict(phases),
        "tasks": task_count,
        "preemptions": preemption_count,
        "failures": failure_count,
        "first_seen": first_time.isoformat() if first_time else None,
    }


def compute_availability(
    worker_prefix: str | None = None,
    accelerator: str | None = None,
    zone: str | None = None,
) -> tuple[dict[str, dict], dict[tuple[str, str], dict[str, float]]]:
    """Compute per-worker and per-(accel, zone) availability stats.

    Returns (stats_by_worker, stats_by_accel).
    """
    logs_dir = Path(jobman_log_dir()) / "workers"
    now = datetime.now(timezone.utc)

    stats_by_worker: dict[str, dict] = {}
    stats_by_accel: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )

    if not logs_dir.is_dir():
        return stats_by_worker, dict(stats_by_accel)

    for worker_dir in sorted(logs_dir.iterdir()):
        if not worker_dir.is_dir():
            continue
        worker_id = worker_dir.name
        if worker_prefix and not worker_id.startswith(worker_prefix):
            continue

        timeline = worker_dir / "timeline.jsonl"
        if not timeline.exists():
            continue

        result = _analyze_worker(timeline, now)
        if result is None:
            continue

        if accelerator and result["accel"] != accelerator:
            continue
        if zone and result["zone"] != zone:
            continue

        stats_by_worker[worker_id] = result
        for phase, secs in result["phases"].items():
            stats_by_accel[(result["accel"], result["zone"])][phase] += secs

    return stats_by_worker, dict(stats_by_accel)


def format_report(
    stats_by_worker: dict[str, dict],
    stats_by_accel: dict[tuple[str, str], dict[str, float]],
) -> str:
    """Format availability stats into a human-readable report."""
    now = datetime.now(timezone.utc)
    lines: list[str] = []

    lines.append("=" * 90)
    lines.append("TPU AVAILABILITY PROFILE")
    lines.append(f"Analysis time: {now.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("=" * 90)

    for (accel, zone), phase_secs in sorted(stats_by_accel.items()):
        total = sum(phase_secs.values())
        if total == 0:
            continue

        task_time = phase_secs.get("running_task", 0)
        idle_ready = phase_secs.get("ready_idle", 0)
        available_for_tasks = task_time + idle_ready
        request_setup = (
            phase_secs.get("requesting", 0)
            + phase_secs.get("deleting", 0)
            + phase_secs.get("deleted", 0)
        )
        capacity_wait = phase_secs.get("waiting_for_resources", 0)
        provisioning = phase_secs.get("provisioning", 0)
        bootstrap_time = (
            phase_secs.get("bootstrapping", 0)
            + phase_secs.get("bootstrap_retry", 0)
        )
        fault_time = (
            phase_secs.get("preempted", 0)
            + phase_secs.get("failed", 0)
            + phase_secs.get("error_recovery", 0)
            + phase_secs.get("quota_error", 0)
            + phase_secs.get("api_error", 0)
        )
        other = phase_secs.get("idle", 0)

        workers = [
            w for w, s in stats_by_worker.items()
            if s["accel"] == accel and s["zone"] == zone
        ]
        total_tasks = sum(stats_by_worker[w]["tasks"] for w in workers)
        total_preemptions = sum(stats_by_worker[w]["preemptions"] for w in workers)
        available_pct = 100 * available_for_tasks / total if total > 0 else 0
        busy_pct = 100 * task_time / total if total > 0 else 0

        def pct(v: float) -> str:
            return f"{100 * v / total:5.1f}%" if total > 0 else "  N/A"

        lines.append("")
        lines.append(
            f"  {accel} in {zone}  |  {len(workers)} workers  |  "
            f"{total_tasks} tasks  |  {total_preemptions} preemptions"
        )
        lines.append(f"  {'─' * 75}")
        lines.append(f"  Total tracked:     {total / 3600:>8.1f} hrs")
        lines.append(f"  Running tasks:     {task_time / 3600:>8.1f} hrs  ({pct(task_time)})")
        lines.append(f"  Idle (ready):      {idle_ready / 3600:>8.1f} hrs  ({pct(idle_ready)})")
        lines.append(f"  Waiting/resources: {capacity_wait / 3600:>8.1f} hrs  ({pct(capacity_wait)})")
        lines.append(f"  Provisioning:      {provisioning / 3600:>8.1f} hrs  ({pct(provisioning)})")
        lines.append(f"  Request/setup:     {request_setup / 3600:>8.1f} hrs  ({pct(request_setup)})")
        lines.append(f"  Bootstrapping:     {bootstrap_time / 3600:>8.1f} hrs  ({pct(bootstrap_time)})")
        lines.append(f"  Faults/errors:     {fault_time / 3600:>8.1f} hrs  ({pct(fault_time)})")
        if phase_secs.get("quota_error", 0):
            lines.append(
                f"  Quota errors:      {phase_secs['quota_error'] / 3600:>8.1f} hrs  ({pct(phase_secs['quota_error'])})"
            )
        if phase_secs.get("api_error", 0):
            lines.append(
                f"  API/network errs:  {phase_secs['api_error'] / 3600:>8.1f} hrs  ({pct(phase_secs['api_error'])})"
            )
        lines.append(f"  Other/init:        {other / 3600:>8.1f} hrs  ({pct(other)})")
        lines.append(f"  >>> Available for tasks:   {available_pct:.1f}%")
        lines.append(f"  >>> Busy running tasks:    {busy_pct:.1f}%")

    return "\n".join(lines)
