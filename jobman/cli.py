"""Click CLI for jobman-lite."""

import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click

from .queue import Queue
from .tpu import TPU, DEFAULT_TPU_VERSION
from .utils import get_logger, jobman_dir

logger = get_logger(__name__)

PRICING_CHOICES = click.Choice(["spot", "preemptible", "standard"])
MODE_CHOICES = click.Choice(["tpu-vm", "queued-resources"])


# ---------------------------------------------------------------------------
# CLI root
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """jobman-lite: Lightweight TPU job orchestration."""


# ---------------------------------------------------------------------------
# jobman worker
# ---------------------------------------------------------------------------

@cli.group()
def worker():
    """Manage workers (persistent TPU holders)."""


@worker.command("start")
@click.option("--accelerator", required=True, help="TPU accelerator type, e.g. v4-8")
@click.option("--zone", required=True, help="GCP zone, e.g. us-central2-b")
@click.option("--tpu-name", default=None, help="TPU VM name (auto-generated if omitted)")
@click.option("--tpu-version", default=DEFAULT_TPU_VERSION, show_default=True,
              help="TPU runtime version")
@click.option("--pricing", default="spot", type=PRICING_CHOICES, show_default=True)
@click.option("--allocation-mode", default="tpu-vm", type=MODE_CHOICES, show_default=True)
def worker_start(accelerator, zone, tpu_name, tpu_version, pricing, allocation_mode):
    """Start a worker in a background tmux session."""
    if not tpu_name:
        tpu_name = _generate_tpu_name(accelerator, zone)
        click.echo(f"Auto-generated TPU name: {tpu_name}")

    state_dir = jobman_dir()
    os.makedirs(state_dir, exist_ok=True)

    session = f"jobman_{tpu_name}"

    # Check if session already exists
    result = subprocess.run(["tmux", "has-session", "-t", session],
                            capture_output=True)
    if result.returncode == 0:
        click.echo(f"tmux session '{session}' already exists. "
                   f"Attach with: tmux attach -t {session}")
        sys.exit(1)

    cmd = (
        f"python -m jobman "
        f"--tpu-name={tpu_name} "
        f"--accelerator={accelerator} "
        f"--zone={zone} "
        f"--tpu-version={tpu_version} "
        f"--pricing={pricing} "
        f"--allocation-mode={allocation_mode}"
    )
    tmux_cmd = ["tmux", "new-session", "-d", "-s", session, cmd]
    subprocess.run(tmux_cmd, check=True)

    click.echo(f"Worker started in tmux session '{session}'")
    click.echo(f"  TPU name    : {tpu_name}")
    click.echo(f"  Accelerator : {accelerator}")
    click.echo(f"  Zone        : {zone}")
    click.echo(f"  Pricing     : {pricing}")
    click.echo(f"  Mode        : {allocation_mode}")
    click.echo(f"Attach with: tmux attach -t {session}")


@worker.command("stop")
@click.argument("worker_ref")
def worker_stop(worker_ref):
    """Stop a worker (by name or index from 'jobman status')."""
    registry = _read_workers()
    tpu_name = _resolve_worker_name(worker_ref, registry)
    if tpu_name is None:
        click.echo(f"Worker '{worker_ref}' not found.", err=True)
        sys.exit(1)
    session = f"jobman_{tpu_name}"
    result = subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
    if result.returncode != 0:
        click.echo(f"No tmux session '{session}' found.")
        sys.exit(1)

    # Update registry
    _update_worker_status(tpu_name, "stopped")
    click.echo(f"Worker '{tpu_name}' stopped.")


@worker.command("stop-all")
def worker_stop_all():
    """Stop all registered workers."""
    registry = _read_workers()
    if not registry:
        click.echo("No workers registered.")
        return
    stopped, skipped = 0, 0
    for tpu_name in registry:
        session = f"jobman_{tpu_name}"
        result = subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
        if result.returncode == 0:
            _update_worker_status(tpu_name, "stopped")
            click.echo(f"Stopped: {tpu_name}")
            stopped += 1
        else:
            click.echo(f"Skipped (no session): {tpu_name}")
            skipped += 1
    click.echo(f"\n{stopped} stopped, {skipped} skipped.")


@worker.command("resume")
@click.argument("worker_ref")
def worker_resume(worker_ref):
    """Resume a dead worker (by name or index from 'jobman status')."""
    registry = _read_workers()
    tpu_name = _resolve_worker_name(worker_ref, registry)
    if tpu_name is None:
        click.echo(f"Worker '{worker_ref}' not found in registry.", err=True)
        sys.exit(1)
    w = registry[tpu_name]

    session = f"jobman_{tpu_name}"
    result = subprocess.run(["tmux", "has-session", "-t", session], capture_output=True)
    if result.returncode == 0:
        click.echo(f"Worker '{tpu_name}' is already running. "
                   f"Attach with: tmux attach -t {session}")
        sys.exit(1)

    cmd = (
        f"python -m jobman "
        f"--tpu-name={tpu_name} "
        f"--accelerator={w['accelerator']} "
        f"--zone={w['zone']} "
        f"--tpu-version={w.get('tpu_version', DEFAULT_TPU_VERSION)} "
        f"--pricing={w.get('pricing', 'spot')} "
        f"--allocation-mode={w.get('mode', 'tpu-vm')}"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", session, cmd], check=True)
    click.echo(f"Resumed worker '{tpu_name}' in tmux session '{session}'.")
    click.echo(f"Attach with: tmux attach -t {session}")


@worker.command("list")
def worker_list():
    """List registered workers with live TPU status from GCP."""
    registry = _read_workers()
    if not registry:
        click.echo("No workers registered.")
        return
    click.echo("Fetching TPU status from GCP...", err=True)
    statuses = _fetch_worker_statuses(registry)
    click.echo(f"{'WORKER':<28} {'ACCELERATOR':<12} {'ZONE':<20} {'PRICING':<12} {'PROCESS':<10} {'TPU'}")
    for w in registry.values():
        pstatus, tstatus = statuses.get(w["worker_id"], ("?", "?"))
        click.echo(f"{w['worker_id']:<28} {w['accelerator']:<12} {w['zone']:<20} "
                   f"{w.get('pricing',''):<12} {pstatus:<10} {tstatus}")


@worker.command("logs")
@click.argument("worker_ref")
@click.option("--lines", "-n", default=50, show_default=True, help="Number of tail lines")
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
def worker_logs(worker_ref, lines, follow):
    """Tail the log for a worker (by name or index from 'jobman status')."""
    registry = _read_workers()
    tpu_name = _resolve_worker_name(worker_ref, registry)
    if tpu_name is None:
        click.echo(f"Worker '{worker_ref}' not found.", err=True)
        sys.exit(1)
    log_file = os.path.join(jobman_dir(), "logs", "workers", tpu_name, "worker.log")
    if not os.path.exists(log_file):
        click.echo(f"No log found for worker {tpu_name}", err=True)
        sys.exit(1)
    cmd = ["tail", f"-n{lines}"]
    if follow:
        cmd.append("-f")
    cmd.append(log_file)
    subprocess.run(cmd)


# ---------------------------------------------------------------------------
# jobman submit
# ---------------------------------------------------------------------------

@cli.command("submit")
@click.argument("script", type=click.Path(exists=True, dir_okay=False))
@click.option("--name", default=None, help="Task name (overrides #JOBMAN header)")
@click.option("--accelerator", default=None, help="Override #JOBMAN --accelerator")
@click.option("--zone", default=None, help="Override #JOBMAN --zone")
@click.option("--tpu-version", default=None, help="Override #JOBMAN --tpu-version")
@click.option("--max-retries", default=None, type=int, help="Override #JOBMAN --max-retries")
@click.option("--pricing", default=None, type=PRICING_CHOICES,
              help="Pricing model (default: spot)")
def submit(script, name, accelerator, zone, tpu_version, max_retries, pricing):
    """Submit a script to the task queue."""
    script_abs = str(Path(script).resolve())
    headers = _parse_headers(script_abs)

    task_spec = {
        "script": script_abs,
        "name": name or headers.get("name") or Path(script).stem,
        "accelerator": accelerator or headers.get("accelerator"),
        "zone": zone or headers.get("zone"),
        "tpu_version": tpu_version or headers.get("tpu-version", DEFAULT_TPU_VERSION),
        "max_retries": max_retries if max_retries is not None
                       else int(headers.get("max-retries", 3)),
        "pricing": pricing or headers.get("pricing", "spot"),
    }

    if not task_spec["accelerator"]:
        raise click.UsageError("--accelerator required (or set #JOBMAN --accelerator=...)")
    if not task_spec["zone"]:
        raise click.UsageError("--zone required (or set #JOBMAN --zone=...)")

    q = Queue()
    task_id = q.submit(task_spec)
    click.echo(f"Submitted task {task_id} ({task_spec['name']})")
    click.echo(f"  Accelerator : {task_spec['accelerator']}")
    click.echo(f"  Zone        : {task_spec['zone']}")
    click.echo(f"  Max retries : {task_spec['max_retries']}")


# ---------------------------------------------------------------------------
# jobman cancel / reset / status / logs
# ---------------------------------------------------------------------------

@cli.command("cancel")
@click.argument("task_ref")
def cancel(task_ref):
    """Cancel a pending or running task (by ID or index from 'jobman status')."""
    q = Queue()
    task_id = _resolve_task_id(task_ref, q)
    if task_id is None:
        click.echo(f"Task '{task_ref}' not found.", err=True)
        sys.exit(1)
    if q.cancel(task_id):
        click.echo(f"Task {task_id} cancelled.")
    else:
        click.echo(f"Task {task_id} not found or already finished.", err=True)
        sys.exit(1)


@cli.command("reset")
@click.argument("task_ref")
def reset(task_ref):
    """Reset a failed/cancelled task back to pending (by ID or index from 'jobman status')."""
    q = Queue()
    task_id = _resolve_task_id(task_ref, q)
    if task_id is None:
        click.echo(f"Task '{task_ref}' not found.", err=True)
        sys.exit(1)
    if q.reset(task_id):
        click.echo(f"Task {task_id} reset to pending.")
    else:
        click.echo(f"Task {task_id} not found.", err=True)
        sys.exit(1)


@cli.command("status")
def status():
    """Show workers and task queue summary."""
    registry = _read_workers()
    click.echo("=== Workers ===")
    if registry:
        click.echo("Fetching TPU status from GCP...", err=True)
        statuses = _fetch_worker_statuses(registry)
        click.echo(f"{'#':<4} {'WORKER':<28} {'ACCELERATOR':<12} {'ZONE':<20} {'PROCESS':<10} {'TPU'}")
        for idx, w in enumerate(registry.values(), 1):
            pstatus, tstatus = statuses.get(w["worker_id"], ("?", "?"))
            click.echo(f"{idx:<4} {w['worker_id']:<28} {w['accelerator']:<12} {w['zone']:<20} "
                       f"{pstatus:<10} {tstatus}")
    else:
        click.echo("  (none)")

    click.echo("\n=== Tasks ===")
    q = Queue()
    tasks = q.list()
    if not tasks:
        click.echo("  (empty)")
        return
    click.echo(f"{'#':<4} {'ID':<18} {'NAME':<20} {'STATUS':<12} {'ACCELERATOR':<12} {'ZONE':<20} {'WORKER'}")
    for idx, t in enumerate(tasks, 1):
        worker_id = t.get("worker_id") or "-"
        click.echo(f"{idx:<4} {t['id']:<18} {t['name']:<20} {t['status']:<12} "
                   f"{t['accelerator']:<12} {t.get('zone',''):<20} {worker_id}")


@cli.command("logs")
@click.argument("task_ref")
@click.option("--lines", "-n", default=50, show_default=True, help="Number of tail lines")
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
def logs(task_ref, lines, follow):
    """Tail the log for a task (by ID or index from 'jobman status')."""
    q = Queue()
    task_id = _resolve_task_id(task_ref, q)
    if task_id is None:
        click.echo(f"Task '{task_ref}' not found.", err=True)
        sys.exit(1)
    log_file = os.path.join(jobman_dir(), "logs", "tasks", task_id, "task.log")
    if not os.path.exists(log_file):
        click.echo(f"No log found for task {task_id}", err=True)
        sys.exit(1)
    cmd = ["tail", f"-n{lines}"]
    if follow:
        cmd.append("-f")
    cmd.append(log_file)
    subprocess.run(cmd)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_status(w: dict) -> str:
    """Return worker process status: 'running', 'dead' (session gone), or 'stopped'."""
    stored = w.get("status", "")
    if stored == "running":
        session = f"jobman_{w['worker_id']}"
        result = subprocess.run(["tmux", "has-session", "-t", session], capture_output=True)
        if result.returncode != 0:
            return "dead"
    return stored


def _query_tpu_status(w: dict) -> str:
    """Query live TPU state from GCP. Returns e.g. READY, CREATING, PREEMPTED, NOT_FOUND."""
    try:
        tpu = TPU(
            name=w["worker_id"],
            zone=w["zone"],
            accelerator=w["accelerator"],
            mode=w.get("mode", "tpu-vm"),
        )
        return tpu.status()
    except Exception:
        return "UNKNOWN"


def _fetch_worker_statuses(workers: dict) -> dict:
    """Fetch process + TPU status for all workers in parallel.
    Returns {worker_id: (process_status, tpu_status)}.
    """
    results = {}

    def fetch_one(w):
        return w["worker_id"], _process_status(w), _query_tpu_status(w)

    with ThreadPoolExecutor(max_workers=min(len(workers), 8)) as ex:
        futures = {ex.submit(fetch_one, w): w["worker_id"] for w in workers.values()}
        for fut in as_completed(futures):
            wid, pstatus, tstatus = fut.result()
            results[wid] = (pstatus, tstatus)

    return results


def _generate_tpu_name(accelerator: str, zone: str) -> str:
    counter_path = os.path.join(jobman_dir(), "worker_counter")
    try:
        count = int(Path(counter_path).read_text().strip()) + 1
    except (FileNotFoundError, ValueError):
        count = 1
    Path(counter_path).write_text(str(count))
    return f"{accelerator}-{zone}-{count:05d}"


def _parse_headers(script_path: str) -> dict[str, str]:
    """Parse #JOBMAN headers from a script file."""
    headers: dict[str, str] = {}
    pattern = re.compile(r"^#JOBMAN\s+(.+)$")
    with open(script_path) as f:
        for line in f:
            m = pattern.match(line.strip())
            if not m:
                continue
            arg_str = m.group(1).strip()
            # Parse --key=value or --key value
            for match in re.finditer(r"--([a-zA-Z0-9_-]+)(?:=(\S+))?", arg_str):
                key = match.group(1)
                val = match.group(2) or "true"
                headers[key] = val
    return headers


def _read_workers() -> dict:
    path = os.path.join(jobman_dir(), "workers.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _resolve_task_id(ref: str, q: Queue) -> str | None:
    """Resolve a task reference: integer index (1-based) or task ID string."""
    if ref.isdigit():
        tasks = q.list()
        idx = int(ref) - 1
        if 0 <= idx < len(tasks):
            return tasks[idx]["id"]
        return None
    return ref


def _resolve_worker_name(ref: str, registry: dict) -> str | None:
    """Resolve a worker reference: integer index (1-based) or TPU name string."""
    if ref.isdigit():
        workers = list(registry.values())
        idx = int(ref) - 1
        if 0 <= idx < len(workers):
            return workers[idx]["worker_id"]
        return None
    return ref  # pass through unknown names; callers decide if that's valid


def _update_worker_status(tpu_name: str, status: str) -> None:
    path = os.path.join(jobman_dir(), "workers.json")
    if not os.path.exists(path):
        return
    try:
        with open(path) as f:
            registry = json.load(f)
        if tpu_name in registry:
            registry[tpu_name]["status"] = status
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(registry, f, indent=2)
        os.replace(tmp, path)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to update worker status: %s", e)
