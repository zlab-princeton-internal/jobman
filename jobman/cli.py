"""Click CLI for jobman-lite."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from fnmatch import fnmatch
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click

from .queue import Queue
from .tpu import TPU, DEFAULT_TPU_VERSION, resolve_tpu_version
from .utils import (
    BREVO_DOCS_URL,
    brevo_config_path,
    get_logger,
    jobman_dir,
    jobman_log_dir,
    load_brevo_config,
    save_brevo_config,
)

logger = get_logger(__name__)

PRICING_CHOICES = click.Choice(["spot", "preemptible", "standard"])
MODE_CHOICES = click.Choice(["tpu-vm", "queued-resources"])
OWNER_FILE = ".jobman_owner"
BREVO_FILE = ".jobman_brevo.json"
MAX_OWNER_LENGTH = 20
ACCELERATOR_PATTERN = re.compile(r"^v[a-z0-9]+-\d+$", re.IGNORECASE)
ZONE_PATTERN = re.compile(r"^[a-z]+(?:-[a-z0-9]+)+\d-[a-z]$")
MAIL_TYPE_ORDER = ("BEGIN", "END", "FAIL")
MAIL_TYPE_CHOICES = {"BEGIN", "END", "FAIL", "ALL", "NONE"}


# ---------------------------------------------------------------------------
# CLI root
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """jobman-lite: Lightweight TPU job orchestration."""


@cli.command("test")
def run_tests():
    """Run the full test suite and print a compact summary."""
    from .test_runner import run_all_tests

    sys.exit(run_all_tests())


# ---------------------------------------------------------------------------
# jobman worker
# ---------------------------------------------------------------------------

@cli.group()
def worker():
    """Manage workers (persistent TPU holders)."""


@worker.command("start")
@click.option("--accelerator", "-a", required=True, help="TPU accelerator type, e.g. v4-8")
@click.option("--zone", "-z", required=True, help="GCP zone, e.g. us-central2-b")
@click.option("--tpu-name", "-n", default=None, help="TPU VM name (auto-generated if omitted)")
@click.option("--pricing", "-p", default="spot", type=PRICING_CHOICES, show_default=True)
@click.option("--allocation-mode", "-m", default="queued-resources", type=MODE_CHOICES, show_default=True)
@click.option("--startup-script", "-s", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None, help="Optional bootstrap script to run on all TPU hosts before claiming tasks")
@click.option("--debug", is_flag=True, default=False,
              help="Run interactively in the foreground with live output and no log files")
def worker_start(accelerator, zone, tpu_name, pricing, allocation_mode, startup_script, debug):
    """Start a worker in a background tmux session."""
    _ensure_owner_name()
    if not tpu_name:
        tpu_name = _generate_tpu_name(accelerator)
        click.echo(f"Auto-generated TPU name: {tpu_name}")

    tpu_version = resolve_tpu_version(accelerator)
    state_dir = jobman_dir()
    os.makedirs(state_dir, exist_ok=True)
    startup_script = str(startup_script.resolve()) if startup_script else None

    args = [
        "python", "-m", "jobman",
        f"--tpu-name={tpu_name}",
        f"--accelerator={accelerator}",
        f"--zone={zone}",
        f"--tpu-version={tpu_version}",
        f"--pricing={pricing}",
        f"--allocation-mode={allocation_mode}",
    ]
    if startup_script:
        args.append(f"--startup-script={startup_script}")
    if debug:
        args.append("--debug")

    if debug:
        click.echo(f"Starting worker '{tpu_name}' in debug mode (Ctrl-C to stop)...")
        os.execvp(args[0], args)
        return  # unreachable

    session = f"jobman_{tpu_name}"

    # Check if session already exists
    result = subprocess.run(["tmux", "has-session", "-t", session],
                            capture_output=True)
    if result.returncode == 0:
        click.echo(f"tmux session '{session}' already exists. "
                   f"Attach with: tmux attach -t {session}")
        sys.exit(1)

    tmux_cmd = ["tmux", "new-session", "-d", "-s", session] + args
    subprocess.run(tmux_cmd, check=True)

    click.echo(f"Worker started in tmux session '{session}'")
    click.echo(f"  TPU name    : {tpu_name}")
    click.echo(f"  Accelerator : {accelerator}")
    click.echo(f"  Zone        : {zone}")
    click.echo(f"  Pricing     : {pricing}")
    click.echo(f"  Mode        : {allocation_mode}")
    if startup_script:
        click.echo(f"  Setup       : {startup_script}")
    if debug:
        click.echo("  Debug       : enabled")
    click.echo(f"Attach with: tmux attach -t {session}")


@worker.command("stop")
@click.argument("worker_refs", nargs=-1)
@click.option("--all", "stop_all", is_flag=True, help="Stop all registered workers")
@click.option("--accelerator", "-a", default=None, help="Stop workers matching accelerator type, e.g. v4-8")
@click.option("--zone", "-z", default=None, help="Stop workers matching GCP zone, e.g. us-central2-b")
def worker_stop(worker_refs, stop_all, accelerator, zone):
    """Stop workers by explicit ref, or by --all / --accelerator / --zone filter."""
    registry = _read_workers()
    worker_names = _select_workers(worker_refs, registry, stop_all, accelerator, zone)
    if worker_names is None:
        return
    failed = False
    for ref, tpu_name in worker_names:
        if tpu_name is None:
            click.echo(f"Worker '{ref}' not found.", err=True)
            failed = True
            continue
        session = f"jobman_{tpu_name}"
        result = subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
        if result.returncode != 0:
            click.echo(f"No tmux session '{session}' found.", err=True)
            failed = True
            continue
        _update_worker_status(tpu_name, "stopped")
        click.echo(f"Worker '{tpu_name}' stopped.")
    if failed:
        sys.exit(1)


@worker.command("resume")
@click.argument("worker_refs", nargs=-1)
@click.option("--all", "resume_all", is_flag=True, help="Resume all registered workers")
@click.option("--accelerator", "-a", default=None, help="Resume workers matching accelerator type, e.g. v4-8")
@click.option("--zone", "-z", default=None, help="Resume workers matching GCP zone, e.g. us-central2-b")
@click.option("--debug", is_flag=True, default=False,
              help="Run interactively in the foreground with live output and no log files")
def worker_resume(worker_refs, resume_all, accelerator, zone, debug):
    """Resume workers by explicit ref, or by --all / --accelerator / --zone filter."""
    registry = _read_workers()
    worker_names = _select_workers(worker_refs, registry, resume_all, accelerator, zone)
    if worker_names is None:
        return

    failed = False
    for ref, tpu_name in worker_names:
        if tpu_name is None or tpu_name not in registry:
            click.echo(f"Worker '{ref}' not found in registry.", err=True)
            failed = True
            continue

        args = _worker_run_args(registry[tpu_name])
        if debug:
            args.append("--debug")

        if debug:
            if len(worker_names) != 1:
                raise click.UsageError("--debug can only be used when resuming exactly one worker.")
            click.echo(f"Resuming worker '{tpu_name}' in debug mode (Ctrl-C to stop)...")
            os.execvp(args[0], args)
            return  # unreachable

        session = f"jobman_{tpu_name}"
        result = subprocess.run(["tmux", "has-session", "-t", session], capture_output=True)
        if result.returncode == 0:
            click.echo(f"Worker '{tpu_name}' is already running. "
                       f"Attach with: tmux attach -t {session}")
            failed = True
            continue

        try:
            subprocess.run(["tmux", "new-session", "-d", "-s", session] + args, check=True)
        except subprocess.CalledProcessError:
            click.echo(f"Failed to resume worker '{tpu_name}'.", err=True)
            failed = True
            continue

        click.echo(f"Resumed worker '{tpu_name}' in tmux session '{session}'.")
        click.echo(f"Attach with: tmux attach -t {session}")

    if failed:
        sys.exit(1)


@worker.command("reboot")
@click.argument("worker_refs", nargs=-1)
@click.option("--all", "reboot_all", is_flag=True, help="Reboot all registered workers")
@click.option("--accelerator", "-a", default=None, help="Reboot workers matching accelerator type, e.g. v4-8")
@click.option("--zone", "-z", default=None, help="Reboot workers matching GCP zone, e.g. us-central2-b")
def worker_reboot(worker_refs, reboot_all, accelerator, zone):
    """Reboot workers by explicit ref, or by --all / --accelerator / --zone filter."""
    registry = _read_workers()
    worker_names = _select_workers(worker_refs, registry, reboot_all, accelerator, zone)
    if worker_names is None:
        return
    failed = False

    for ref, tpu_name in worker_names:
        if tpu_name is None or tpu_name not in registry:
            click.echo(f"Worker '{ref}' not found in registry.", err=True)
            failed = True
            continue

        session = f"jobman_{tpu_name}"
        subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
        _update_worker_status(tpu_name, "stopped")

        try:
            subprocess.run(["tmux", "new-session", "-d", "-s", session] + _worker_run_args(registry[tpu_name]),
                           check=True)
        except subprocess.CalledProcessError:
            click.echo(f"Failed to reboot worker '{tpu_name}'.", err=True)
            failed = True
            continue

        click.echo(f"Worker '{tpu_name}' rebooted.")

    if failed:
        sys.exit(1)


@worker.command("delete")
@click.argument("worker_refs", nargs=-1)
@click.option("--all", "delete_all", is_flag=True, help="Delete all registered workers")
@click.option("--accelerator", "-a", default=None, help="Delete workers matching accelerator type, e.g. v4-8")
@click.option("--zone", "-z", default=None, help="Delete workers matching GCP zone, e.g. us-central2-b")
def worker_delete(worker_refs, delete_all, accelerator, zone):
    """Delete workers by explicit ref, or by --all / --accelerator / --zone filter."""
    registry = _read_workers()
    worker_names = _select_workers(worker_refs, registry, delete_all, accelerator, zone)
    if worker_names is None:
        return
    queue = Queue()
    failed = False
    delete_targets: list[tuple[str, dict]] = []

    for ref, tpu_name in worker_names:
        if tpu_name is None or tpu_name not in registry:
            click.echo(f"Worker '{ref}' not found in registry.", err=True)
            failed = True
            continue

        worker_cfg = registry[tpu_name]
        session = f"jobman_{tpu_name}"
        result = subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
        if result.returncode == 0:
            click.echo(f"Worker '{tpu_name}' stopped.")
        else:
            click.echo(f"No tmux session '{session}' found; continuing with delete.")
        _update_worker_status(tpu_name, "stopped")

        released = queue.release_worker_tasks(tpu_name)
        if released:
            click.echo(f"Released tasks: {', '.join(released)}")

        delete_targets.append((tpu_name, worker_cfg))

    def delete_one(tpu_name: str, worker_cfg: dict) -> tuple[str, Exception | None]:
        try:
            TPU(
                name=worker_cfg["worker_id"],
                zone=worker_cfg["zone"],
                accelerator=worker_cfg["accelerator"],
                version=worker_cfg.get("tpu_version") or resolve_tpu_version(worker_cfg["accelerator"]),
                pricing=worker_cfg.get("pricing", "spot"),
                mode=worker_cfg.get("mode", "tpu-vm"),
            ).delete()
            return tpu_name, None
        except Exception as e:
            return tpu_name, e

    if delete_targets:
        with ThreadPoolExecutor(max_workers=min(len(delete_targets), 8)) as ex:
            futures = {
                ex.submit(delete_one, tpu_name, worker_cfg): tpu_name
                for tpu_name, worker_cfg in delete_targets
            }
            with click.progressbar(length=len(futures), label="Deleting TPUs", show_pos=True) as bar:
                for fut in as_completed(futures):
                    tpu_name, err = fut.result()
                    if err is not None:
                        click.echo(f"Failed to delete TPU for worker '{tpu_name}': {err}", err=True)
                        failed = True
                    else:
                        _remove_worker(tpu_name)
                        click.echo(f"Worker '{tpu_name}' deleted.")
                    bar.update(1)

    if failed:
        sys.exit(1)


@worker.command("ssh")
@click.argument("worker_ref")
@click.option("--worker-index", "-w", default=0, show_default=True,
              help="TPU worker index (for multi-host TPUs)")
def worker_ssh(worker_ref, worker_index):
    """Open an interactive SSH session to a worker (by name or index from 'jobman status')."""
    registry = _read_workers()
    tpu_name = _resolve_worker_name(worker_ref, registry)
    if tpu_name is None:
        click.echo(f"Worker '{worker_ref}' not found.", err=True)
        sys.exit(1)
    w = registry.get(tpu_name, {})
    zone = w.get("zone")
    if not zone:
        click.echo(f"No zone found for worker '{tpu_name}'.", err=True)
        sys.exit(1)
    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "ssh", tpu_name,
        f"--zone={zone}",
        f"--worker={worker_index}",
        "--ssh-flag=-o StrictHostKeyChecking=no",
        "--ssh-flag=-o UserKnownHostsFile=/dev/null",
    ]
    os.execvp("gcloud", cmd)


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
    log_file = os.path.join(jobman_log_dir(), "workers", tpu_name, "worker.log")
    if not os.path.exists(log_file):
        click.echo(f"No log found for worker {tpu_name}", err=True)
        sys.exit(1)
    cmd = ["tail", f"-n{lines}"]
    if follow:
        cmd.append("-f")
    cmd.append(log_file)
    subprocess.run(cmd)


@worker.command("show")
@click.argument("worker_ref")
def worker_show(worker_ref):
    """Show detailed information for one worker (by name or index from 'jobman status')."""
    registry = _read_workers()
    tpu_name = _resolve_worker_name(worker_ref, registry)
    if tpu_name is None or tpu_name not in registry:
        click.echo(f"Worker '{worker_ref}' not found.", err=True)
        sys.exit(1)

    w = registry[tpu_name]
    click.echo(f"Worker            : {w['worker_id']}")
    click.echo(f"tmux session      : jobman_{w['worker_id']}")
    click.echo(f"TPU name          : {w['worker_id']}")
    click.echo(f"Accelerator       : {w['accelerator']}")
    click.echo(f"Zone              : {w['zone']}")
    click.echo(f"Pricing           : {w.get('pricing', 'spot')}")
    click.echo(f"Mode              : {w.get('mode', 'tpu-vm')}")
    startup_script = w.get("startup_script")
    if startup_script:
        snapshot_path = os.path.join(jobman_log_dir(), "workers", w["worker_id"], os.path.basename(startup_script))
        click.echo(f"Setup             : {snapshot_path}")
    process_status = _process_status(w)
    vm_status, qr_status = _query_tpu_statuses(w)
    click.echo(f"Process status    : {process_status}")
    click.echo(f"VM status         : {vm_status}")
    click.echo(f"QR status         : {qr_status}")
    click.echo(f"Attach with       : tmux attach -t jobman_{w['worker_id']}")
    tpu_url = _tpu_console_url(w)
    if tpu_url:
        click.echo(f"URL               : {tpu_url}")


# ---------------------------------------------------------------------------
# jobman task
# ---------------------------------------------------------------------------

@cli.group()
def task():
    """Manage tasks (queue entries)."""


@task.command("submit")
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
    resolved_mail_user = headers.get("mail-user")
    resolved_mail_types = _normalize_mail_types(headers.get("mail-type"))

    task_spec = {
        "script": script_abs,
        "name": name or headers.get("name") or Path(script).stem,
        "accelerator": accelerator or headers.get("accelerator"),
        "zone": zone or headers.get("zone"),
        "tpu_version": tpu_version or headers.get("tpu-version") or None,  # resolved below
        "max_retries": max_retries if max_retries is not None
                       else int(headers.get("max-retries", 3)),
        "pricing": pricing or headers.get("pricing", "spot"),
        "mail_user": None,
        "mail_types": [],
        "mail_config_path": None,
    }

    if not task_spec["accelerator"]:
        raise click.UsageError("--accelerator required (or set #JOBMAN --accelerator=...)")
    _validate_accelerator(task_spec["accelerator"])
    if not task_spec["tpu_version"]:
        task_spec["tpu_version"] = resolve_tpu_version(task_spec["accelerator"])
    if not task_spec["zone"]:
        raise click.UsageError("--zone required (or set #JOBMAN --zone=...)")
    _validate_zone(task_spec["zone"])
    if resolved_mail_user and not resolved_mail_types:
        raise click.UsageError("#JOBMAN --mail-type required when #JOBMAN --mail-user is set.")
    if resolved_mail_types and not resolved_mail_user:
        raise click.UsageError("#JOBMAN --mail-user required when #JOBMAN --mail-type is set.")
    if resolved_mail_user:
        task_spec["mail_user"] = _validate_email(resolved_mail_user, option_name="#JOBMAN --mail-user")
        task_spec["mail_types"] = resolved_mail_types
        task_spec["mail_config_path"] = str(_ensure_brevo_config())
        brevo_cfg = load_brevo_config(task_spec["mail_config_path"])
        if brevo_cfg.get("disabled"):
            click.echo(
                "Warning: task requests email notifications, but local Brevo sending is disabled "
                f"in {task_spec['mail_config_path']}. No emails will be sent.",
                err=True,
            )

    q = Queue()
    task_id = q.submit(task_spec)
    click.echo(f"Submitted task {task_id} ({task_spec['name']})")
    click.echo(f"  Accelerator : {task_spec['accelerator']}")
    click.echo(f"  Zone        : {task_spec['zone']}")
    click.echo(f"  Max retries : {task_spec['max_retries']}")
    if task_spec["mail_user"]:
        click.echo(f"  Mail user   : {task_spec['mail_user']}")
        click.echo(f"  Mail type   : {','.join(task_spec['mail_types'])}")


# ---------------------------------------------------------------------------
# jobman task / status
# ---------------------------------------------------------------------------

@task.command("delete")
@click.argument("task_refs", nargs=-1)
@click.option("--all", "delete_all", is_flag=True, help="Delete all tasks")
@click.option("--accelerator", "-a", default=None, help="Delete tasks matching accelerator type, e.g. v4-8")
@click.option("--zone", "-z", default=None, help="Delete tasks matching GCP zone, e.g. us-central2-b")
@click.option("--pattern", "-p", default=None, help="Delete tasks whose names match a glob pattern")
def delete(task_refs, delete_all, accelerator, zone, pattern):
    """Delete tasks by explicit ref, or by --all / --accelerator / --zone / --pattern filter."""
    q = Queue()
    task_ids = _select_tasks(task_refs, q, delete_all, accelerator, zone, pattern)
    if task_ids is None:
        return
    failed = False
    for ref, task_id in task_ids:
        if task_id is None:
            click.echo(f"Task '{ref}' not found.", err=True)
            failed = True
            continue
        if q.cancel(task_id):
            click.echo(f"Task {task_id} deleted from queue.")
        else:
            click.echo(f"Task {task_id} not found or not deletable.", err=True)
            failed = True
    if failed:
        sys.exit(1)


@task.command("pause")
@click.argument("task_refs", nargs=-1)
@click.option("--all", "pause_all", is_flag=True, help="Pause all tasks")
@click.option("--accelerator", "-a", default=None, help="Pause tasks matching accelerator type, e.g. v4-8")
@click.option("--zone", "-z", default=None, help="Pause tasks matching GCP zone, e.g. us-central2-b")
@click.option("--pattern", "-p", default=None, help="Pause tasks whose names match a glob pattern")
def pause(task_refs, pause_all, accelerator, zone, pattern):
    """Pause tasks by explicit ref, or by --all / --accelerator / --zone / --pattern filter."""
    q = Queue()
    task_ids = _select_tasks(task_refs, q, pause_all, accelerator, zone, pattern)
    if task_ids is None:
        return
    failed = False
    for ref, task_id in task_ids:
        if task_id is None:
            click.echo(f"Task '{ref}' not found.", err=True)
            failed = True
            continue
        if q.pause(task_id):
            click.echo(f"Task {task_id} paused.")
        else:
            click.echo(f"Task {task_id} not found or not pausable.", err=True)
            failed = True
    if failed:
        sys.exit(1)


@task.command("requeue")
@click.argument("task_refs", nargs=-1)
@click.option("--all", "requeue_all", is_flag=True, help="Requeue all tasks")
@click.option("--accelerator", "-a", default=None, help="Requeue tasks matching accelerator type, e.g. v4-8")
@click.option("--zone", "-z", default=None, help="Requeue tasks matching GCP zone, e.g. us-central2-b")
@click.option("--pattern", "-p", default=None, help="Requeue tasks whose names match a glob pattern")
def requeue(task_refs, requeue_all, accelerator, zone, pattern):
    """Requeue tasks by explicit ref, or by --all / --accelerator / --zone / --pattern filter."""
    q = Queue()
    task_ids = _select_tasks(task_refs, q, requeue_all, accelerator, zone, pattern)
    if task_ids is None:
        return
    failed = False
    for ref, task_id in task_ids:
        if task_id is None:
            click.echo(f"Task '{ref}' not found.", err=True)
            failed = True
            continue
        if q.reset(task_id):
            click.echo(f"Task {task_id} reset to pending.")
        else:
            click.echo(f"Task {task_id} not found.", err=True)
            failed = True
    if failed:
        sys.exit(1)


@cli.command("status")
@click.option("--accelerator", "-a", default=None, help="Filter by accelerator type, e.g. v4-8")
@click.option("--zone", "-z", default=None, help="Filter by GCP zone, e.g. us-central2-b")
@click.option("--live-only", "-lo", is_flag=True, help="Show only workers with queued-resource status ACTIVE")
@click.option("--workers-only", "-wo", is_flag=True, help="Show only the worker table")
@click.option("--task-only", "-to", is_flag=True, help="Show only the task table")
def status(accelerator, zone, live_only, workers_only, task_only):
    """Show workers and task queue summary."""
    if workers_only and task_only:
        raise click.UsageError("--workers-only and --task-only cannot be used together")

    registry = _read_workers()
    all_workers = list(registry.values())
    filtered_workers = [
        (idx, w) for idx, w in enumerate(all_workers, 1)
        if (accelerator is None or w.get("accelerator") == accelerator)
        and (zone is None or w.get("zone") == zone)
    ]

    if not task_only:
        click.echo("=== Workers ===")
        if filtered_workers:
            click.echo("Fetching TPU status from GCP...", err=True)
            statuses = _fetch_worker_statuses({w["worker_id"]: w for _, w in filtered_workers})
            click.echo(f"{'#':<4} {'WORKER':<28} {'ACCELERATOR':<12} {'ZONE':<20} {'PROCESS':<10} {'VM':<10} {'QR'}")
            rows = []
            for idx, w in filtered_workers:
                pstatus, vm_status, qr_status = statuses.get(w["worker_id"], ("?", "?", "?"))
                if live_only and qr_status.upper() != "ACTIVE":
                    continue
                rows.append((idx, w, pstatus, vm_status, qr_status))
            if rows:
                for idx, w, pstatus, vm_status, qr_status in rows:
                    click.echo(f"{idx:<4} {w['worker_id']:<28} {w['accelerator']:<12} {w['zone']:<20} "
                               f"{pstatus:<10} {vm_status:<10} {qr_status}")
            else:
                click.echo("  (none)")
        else:
            click.echo("  (none)")

    if workers_only:
        return

    if not task_only:
        click.echo("\n=== Tasks ===")
    else:
        click.echo("=== Tasks ===")
    q = Queue()
    all_tasks = q.list()
    tasks = [
        (idx, t) for idx, t in enumerate(all_tasks, 1)
        if (accelerator is None or t.get("accelerator") == accelerator)
        and (zone is None or t.get("zone") == zone)
    ]
    if not tasks:
        click.echo("  (empty)")
        return
    click.echo(f"{'#':<4} {'ID':<18} {'NAME':<40} {'STATUS':<12} {'RETRY':<7} {'ACCELERATOR':<12} {'ZONE':<20} {'WORKER'}")
    for idx, t in tasks:
        worker_id = t.get("worker_id") or "-"
        retry = f"{t.get('fail_count', 0)}/{t.get('max_retries', 3)}"
        click.echo(f"{idx:<4} {t['id']:<18} {t['name']:<40} {t['status']:<12} {retry:<7} "
                   f"{t['accelerator']:<12} {t.get('zone',''):<20} {worker_id}")


@task.command("show")
@click.argument("task_ref")
def task_show(task_ref):
    """Show detailed information for one task (by ID or index from 'jobman status')."""
    q = Queue()
    task_id = _resolve_task_id(task_ref, q)
    if task_id is None:
        click.echo(f"Task '{task_ref}' not found.", err=True)
        sys.exit(1)

    t = q.get(task_id)
    if t is None:
        click.echo(f"Task '{task_ref}' not found.", err=True)
        sys.exit(1)

    click.echo(f"Task              : {t['id']}")
    click.echo(f"Name              : {t.get('name', t['id'])}")
    click.echo(f"Status            : {t.get('status', '')}")
    click.echo(f"Accelerator       : {t.get('accelerator', '')}")
    click.echo(f"Zone              : {t.get('zone', '')}")
    click.echo(f"TPU version       : {t.get('tpu_version', '')}")
    click.echo(f"Pricing           : {t.get('pricing', 'spot')}")
    click.echo(f"Worker            : {t.get('worker_id') or '-'}")
    click.echo(f"Retries           : {t.get('fail_count', 0)}/{t.get('max_retries', 3)}")
    click.echo(f"Run count         : {t.get('run_count', 0)}")
    click.echo(f"Mail user         : {t.get('mail_user') or '-'}")
    click.echo(f"Mail type         : {','.join(t.get('mail_types', [])) or '-'}")
    click.echo(f"Submitted         : {t.get('submitted') or '-'}")
    click.echo(f"Started           : {t.get('started') or '-'}")
    click.echo(f"Ended             : {t.get('ended') or '-'}")
    click.echo(f"Script            : {t.get('script') or '-'}")
    log_file = _latest_task_log(task_id)
    if log_file:
        click.echo(f"Latest log        : {log_file}")


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


def _query_tpu_statuses(w: dict) -> tuple[str, str]:
    """Query live TPU VM and queued-resource states from GCP."""
    try:
        tpu = TPU(
            name=w["worker_id"],
            zone=w["zone"],
            accelerator=w["accelerator"],
            mode=w.get("mode", "tpu-vm"),
        )
        return tpu.vm_status(), tpu.queued_resource_status()
    except Exception:
        return "UNKNOWN", "UNKNOWN"


def _fetch_worker_statuses(workers: dict) -> dict:
    """Fetch process + TPU status for all workers in parallel.
    Returns {worker_id: (process_status, vm_status, qr_status)}.
    """
    results = {}

    def fetch_one(w):
        vm_status, qr_status = _query_tpu_statuses(w)
        return w["worker_id"], _process_status(w), vm_status, qr_status

    with ThreadPoolExecutor(max_workers=min(len(workers), 8)) as ex:
        futures = {ex.submit(fetch_one, w): w["worker_id"] for w in workers.values()}
        with click.progressbar(length=len(futures), label="Fetching TPU status", show_pos=True) as bar:
            for fut in as_completed(futures):
                wid, pstatus, vm_status, qr_status = fut.result()
                results[wid] = (pstatus, vm_status, qr_status)
                bar.update(1)

    return results


@lru_cache(maxsize=1)
def _gcloud_project() -> str | None:
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None

    if result.returncode != 0:
        return None
    project = (result.stdout or "").strip()
    if not project or project == "(unset)":
        return None
    return project


def _tpu_console_url(worker: dict) -> str:
    zone = worker.get("zone")
    worker_id = worker.get("worker_id")
    mode = worker.get("mode", "tpu-vm")
    if not zone or not worker_id:
        return ""

    if mode == "queued-resources":
        path = f"/compute/tpus/queuedResources/details/{zone}/{worker_id}"
    else:
        path = f"/compute/tpus/details/{zone}/{worker_id}"

    project = _gcloud_project()
    query_parts = []
    query_parts.append(f"authuser=1")
    if project:
        query_parts.append(f"project={project}")
    query = f"?{'&'.join(query_parts)}" if query_parts else ""
    return f"https://console.cloud.google.com{path}{query}"


def _owner_file_path() -> Path:
    return Path.cwd() / OWNER_FILE


def _validate_owner_name(value: str) -> str:
    owner = value.strip().lower()
    if not owner:
        raise click.UsageError("Owner name cannot be empty.")
    if " " in owner:
        raise click.UsageError("Owner name cannot contain spaces.")
    if len(owner) > MAX_OWNER_LENGTH:
        raise click.UsageError(f"Owner name must be at most {MAX_OWNER_LENGTH} characters.")
    if not re.fullmatch(r"[a-z0-9-]+", owner):
        raise click.UsageError("Owner name may contain only lowercase letters, digits, and hyphens.")
    return owner


def _ensure_owner_name() -> str:
    owner_path = _owner_file_path()
    if owner_path.exists():
        try:
            return _validate_owner_name(owner_path.read_text().strip())
        except (OSError, click.UsageError) as exc:
            raise click.ClickException(
                f"Invalid owner file at {owner_path}: {exc}. "
                f"Fix or remove it and rerun 'jobman worker start'."
            ) from exc

    owner = _prompt_owner_name()
    owner_path.write_text(owner + "\n")
    click.echo(f"Saved owner name to {owner_path}")
    return owner


def _prompt_owner_name() -> str:
    while True:
        value = click.prompt(f"Enter owner name (lowercase, no spaces, <= {MAX_OWNER_LENGTH} chars)")
        try:
            return _validate_owner_name(value)
        except click.UsageError as exc:
            click.echo(str(exc), err=True)


def _validate_email(value: str, option_name: str = "email") -> str:
    email = value.strip()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise click.UsageError(f"Invalid email for {option_name}: {value}")
    return email


def _normalize_mail_types(value: str | None) -> list[str]:
    if value is None:
        return []
    raw_items = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not raw_items:
        return []
    invalid = [item for item in raw_items if item not in MAIL_TYPE_CHOICES]
    if invalid:
        raise click.UsageError(
            f"Invalid --mail-type value(s): {', '.join(invalid)}. "
            "Use BEGIN, END, FAIL, ALL, or NONE."
        )
    if "NONE" in raw_items:
        if len(raw_items) != 1:
            raise click.UsageError("NONE cannot be combined with other --mail-type values.")
        return []
    expanded = []
    for item in raw_items:
        if item == "ALL":
            expanded.extend(MAIL_TYPE_ORDER)
        else:
            expanded.append(item)
    return [item for item in MAIL_TYPE_ORDER if item in expanded]


def _ensure_brevo_config() -> Path:
    cfg_path = brevo_config_path()
    if cfg_path.name != BREVO_FILE:
        raise click.ClickException(f"Unexpected Brevo config path: {cfg_path}")
    try:
        config = load_brevo_config(str(cfg_path))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise click.ClickException(
            f"Invalid Brevo config at {cfg_path}: {exc}. Fix or remove it and retry."
        ) from exc
    if config.get("disabled"):
        return cfg_path
    api_key = config.get("api_key", "")
    sender_email = config.get("sender_email", "")
    if api_key and sender_email:
        return cfg_path

    click.echo("Email notifications use Brevo.")
    click.echo(f"Brevo setup guide: {BREVO_DOCS_URL}")
    if click.confirm("Skip Brevo setup for now and disable local email sending?", default=False):
        save_brevo_config("", "", str(cfg_path), disabled=True)
        click.echo(f"Saved disabled Brevo config to {cfg_path}")
        return cfg_path
    api_key = click.prompt("Enter Brevo API key", hide_input=True).strip()
    sender_email = _validate_email(
        click.prompt("Enter verified Brevo sender email").strip(),
        option_name="Brevo sender email",
    )
    save_brevo_config(api_key, sender_email, str(cfg_path))
    click.echo(f"Saved Brevo config to {cfg_path}")
    return cfg_path


def _generate_tpu_name(accelerator: str) -> str:
    owner = _ensure_owner_name()
    counter_path = os.path.join(jobman_dir(), "worker_counter")
    try:
        count = int(Path(counter_path).read_text().strip()) + 1
    except (FileNotFoundError, ValueError):
        count = 1
    Path(counter_path).write_text(str(count))
    return f"{owner}-{accelerator}-{count:05d}"


def _worker_run_args(worker: dict) -> list[str]:
    """Build the worker process argv from a registry entry."""
    args = [
        "python", "-m", "jobman",
        f"--tpu-name={worker['worker_id']}",
        f"--accelerator={worker['accelerator']}",
        f"--zone={worker['zone']}",
        f"--tpu-version={worker.get('tpu_version') or resolve_tpu_version(worker['accelerator'])}",
        f"--pricing={worker.get('pricing', 'spot')}",
        f"--allocation-mode={worker.get('mode', 'tpu-vm')}",
    ]
    if worker.get("startup_script"):
        args.append(f"--startup-script={worker['startup_script']}")
    return args


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


def _validate_accelerator(value: str) -> str:
    accelerator = value.strip()
    if not ACCELERATOR_PATTERN.fullmatch(accelerator):
        raise click.UsageError(
            f"Invalid accelerator '{value}'. Expected values like v4-8, v5e-16, or v6e-32."
        )
    return accelerator


def _validate_zone(value: str) -> str:
    zone = value.strip()
    if not ZONE_PATTERN.fullmatch(zone):
        raise click.UsageError(
            f"Invalid zone '{value}'. Expected a GCP zone like us-central2-b."
        )
    return zone


def _read_workers() -> dict:
    path = os.path.join(jobman_dir(), "workers.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _latest_task_log(task_id: str) -> str | None:
    """Return the most recently modified run log for a task from known log roots, or None."""
    log_dirs = [
        Path(jobman_log_dir()) / "tasks" / task_id,
        Path(jobman_dir()) / "logs" / "tasks" / task_id,
    ]
    candidates: list[Path] = []
    for log_dir in log_dirs:
        if not log_dir.is_dir():
            continue
        candidates.extend(log_dir.glob("run_*.log"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


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


def _resolve_task_ids(refs: tuple[str, ...], q: Queue) -> list[tuple[str, str | None]]:
    """Resolve multiple task references against a stable snapshot."""
    tasks = q.list()
    resolved = []
    for ref in refs:
        if ref.isdigit():
            idx = int(ref) - 1
            task_id = tasks[idx]["id"] if 0 <= idx < len(tasks) else None
        else:
            task_id = ref
        resolved.append((ref, task_id))
    return resolved


def _select_tasks(
    task_refs: tuple[str, ...],
    q: Queue,
    select_all: bool,
    accelerator: str | None,
    zone: str | None,
    pattern: str | None,
) -> list[tuple[str, str | None]] | None:
    """Select tasks by explicit refs or by --all / --accelerator / --zone / --pattern filters."""
    tasks = q.list()
    if not tasks:
        click.echo("No tasks in queue.")
        return None

    using_selector = select_all or accelerator is not None or zone is not None or pattern is not None
    if task_refs and using_selector:
        raise click.UsageError("Pass task refs, or use --all / --accelerator / --zone / --pattern, not both.")
    if not task_refs and not using_selector:
        raise click.UsageError("Pass task refs, or select tasks with --all / --accelerator / --zone / --pattern.")

    if select_all:
        return [(t["id"], t["id"]) for t in tasks]

    if accelerator is not None or zone is not None or pattern is not None:
        task_ids = [
            (t["id"], t["id"])
            for t in tasks
            if (accelerator is None or t.get("accelerator") == accelerator)
            and (zone is None or t.get("zone") == zone)
            and (pattern is None or fnmatch(t.get("name", ""), pattern))
        ]
        if not task_ids:
            click.echo("No tasks matched the requested filters.")
            return None
        return task_ids

    return _resolve_task_ids(task_refs, q)


def _resolve_worker_names(refs: tuple[str, ...], registry: dict) -> list[tuple[str, str | None]]:
    """Resolve multiple worker references against a stable snapshot."""
    workers = list(registry.values())
    resolved = []
    for ref in refs:
        if ref.isdigit():
            idx = int(ref) - 1
            worker_name = workers[idx]["worker_id"] if 0 <= idx < len(workers) else None
        else:
            worker_name = ref
        resolved.append((ref, worker_name))
    return resolved


def _select_workers(
    worker_refs: tuple[str, ...],
    registry: dict,
    select_all: bool,
    accelerator: str | None,
    zone: str | None,
) -> list[tuple[str, str | None]] | None:
    """Select workers by explicit refs or by --all / --accelerator / --zone filters."""
    if not registry:
        click.echo("No workers registered.")
        return None

    using_selector = select_all or accelerator is not None or zone is not None
    if worker_refs and using_selector:
        raise click.UsageError("Pass worker refs, or use --all / --accelerator / --zone, not both.")
    if not worker_refs and not using_selector:
        raise click.UsageError("Pass worker refs, or select workers with --all / --accelerator / --zone.")

    if select_all:
        return [(w["worker_id"], w["worker_id"]) for w in registry.values()]
    if accelerator is not None or zone is not None:
        worker_names = [
            (w["worker_id"], w["worker_id"])
            for w in registry.values()
            if (accelerator is None or w.get("accelerator") == accelerator)
            and (zone is None or w.get("zone") == zone)
        ]
        if not worker_names:
            click.echo("No workers matched the requested filters.")
            return None
        return worker_names
    return _resolve_worker_names(worker_refs, registry)


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


def _remove_worker(tpu_name: str) -> None:
    path = os.path.join(jobman_dir(), "workers.json")
    if not os.path.exists(path):
        return
    try:
        with open(path) as f:
            registry = json.load(f)
        registry.pop(tpu_name, None)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(registry, f, indent=2)
        os.replace(tmp, path)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to remove worker from registry: %s", e)
