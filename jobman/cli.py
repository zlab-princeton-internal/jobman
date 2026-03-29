"""Click CLI for jobman-lite."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from fnmatch import fnmatch
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click

# from analytics import availability as analytics_availability
# from analytics import cost as analytics_cost
# from analytics import storage as analytics_storage
from .queue import Queue
from .tpu import TPU, DEFAULT_TPU_VERSION, resolve_tpu_version
from .utils import (
    BREVO_DOCS_URL,
    brevo_config_path,
    dir_lock,
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
ANALYTICS_OUTPUT_CHOICES = click.Choice(["table", "json"])
ANALYTICS_OWNERSHIP_CHOICES = click.Choice(["all", "mine", "not_mine", "unknown"])
BILLING_TABLE_PATTERN = re.compile(r"^[A-Za-z0-9_:\-\.]+$")
DEFAULT_BILLING_EXPORT_TABLE = "billing_export_us.gcp_billing_export_resource_v1_017331_B5D939_87F923"


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


@cli.group()
def analytics():
    """Historical analytics and reporting."""


@analytics.command("availability")
def analytics_availability_cmd():
    """Show availability analytics."""
    click.echo("Availability analytics is not implemented yet.")


@analytics.command("cost")
@click.option(
    "--date",
    "target_date",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Billing date to query, in YYYY-MM-DD.",
)
@click.option(
    "--billing-table",
    envvar="JOBMAN_BILLING_EXPORT_TABLE",
    default=DEFAULT_BILLING_EXPORT_TABLE,
    show_default=True,
    help="BigQuery billing export table, e.g. project.dataset.table.",
)
@click.option("--limit", default=50, show_default=True, type=int, help="Maximum rows to return.")
@click.option(
    "--location",
    default=None,
    help="Optional location filter, matched against billing location.location.",
)
@click.option(
    "--network-only",
    is_flag=True,
    help="Only show rows whose SKU matches network data transfer.",
)
@click.option(
    "--output",
    "output_format",
    default="table",
    show_default=True,
    type=ANALYTICS_OUTPUT_CHOICES,
    help="Render output as a compact table or raw JSON.",
)
@click.option(
    "--ownership",
    default="all",
    show_default=True,
    type=ANALYTICS_OWNERSHIP_CHOICES,
    help="Filter by conservative ownership classification.",
)
def analytics_cost_cmd(target_date, billing_table, limit, location, network_only, output_format, ownership):
    """Show a daily billing breakdown from BigQuery billing export."""
    _validate_billing_table(billing_table)
    if limit <= 0:
        raise click.UsageError("--limit must be greater than 0.")

    where_clauses = [
        "cost > 0",
        "cost_type = 'regular'",
        "DATE(usage_start_time) = @target_date",
    ]
    parameters = [f"target_date:DATE:{target_date.date().isoformat()}"]
    if location:
        where_clauses.append("LOWER(location.location) = LOWER(@location)")
        parameters.append(f"location:STRING:{location}")
    if network_only:
        where_clauses.append("sku.description LIKE '%Network %Data Transfer%'")
    if ownership != "all":
        where_clauses.append("ownership = @ownership")
        parameters.append(f"ownership:STRING:{ownership}")

    query = f"""
WITH classified AS (
SELECT
  DATE(usage_start_time) AS usage_date,
  service.description AS service,
  sku.description AS sku,
  location.location AS location,
  project.id AS project_id,
  resource.name AS resource_name,
  resource.global_name AS global_name,
  CASE
    WHEN LOWER(COALESCE(resource.name, '')) LIKE 'llm_pruning%' THEN 'mine'
    WHEN LOWER(COALESCE(resource.global_name, '')) LIKE '%llm_pruning%' THEN 'mine'
    WHEN LOWER(COALESCE(resource.name, '')) LIKE 'yufeng%' THEN 'mine'
    WHEN LOWER(COALESCE(resource.global_name, '')) LIKE '%/yufeng%' THEN 'mine'
    WHEN service.description = 'Cloud Storage'
      AND COALESCE(resource.name, '') != ''
      AND LOWER(COALESCE(resource.name, '')) NOT LIKE 'llm_pruning%' THEN 'not_mine'
    WHEN service.description = 'Cloud Storage'
      AND COALESCE(resource.global_name, '') != ''
      AND LOWER(COALESCE(resource.global_name, '')) NOT LIKE '%llm_pruning%' THEN 'not_mine'
    ELSE 'unknown'
  END AS ownership,
  ROUND(SUM(cost), 2) AS cost,
  ROUND(SUM(usage.amount), 2) AS usage_amount,
  ANY_VALUE(usage.unit) AS usage_unit
FROM `{billing_table}`
WHERE
  {' AND '.join(where_clauses[:4] if ownership == 'all' and not network_only and not location else [c for c in where_clauses if c not in ('ownership = @ownership',)])}
GROUP BY usage_date, service, sku, location, project_id, resource_name, global_name, ownership
)
SELECT *
FROM classified
WHERE
  {' AND '.join([c for c in where_clauses[4:] if c != "sku.description LIKE '%Network %Data Transfer%'"] + (["sku LIKE '%Network %Data Transfer%'"] if network_only else []) or ["TRUE"])}
ORDER BY cost DESC
LIMIT {limit}
""".strip()

    rows = _run_bq_query(query, parameters)
    _emit_analytics_rows(rows, output_format=output_format)


@analytics.command("egress")
@click.option(
    "--date",
    "target_date",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Billing date to query, in YYYY-MM-DD.",
)
@click.option(
    "--billing-table",
    envvar="JOBMAN_BILLING_EXPORT_TABLE",
    default=DEFAULT_BILLING_EXPORT_TABLE,
    show_default=True,
    help="BigQuery billing export table, e.g. project.dataset.table.",
)
@click.option("--limit", default=50, show_default=True, type=int, help="Maximum rows to return.")
@click.option(
    "--kind",
    type=click.Choice(["internet", "inter-zone", "all"]),
    default="all",
    show_default=True,
    help="Which transfer SKU family to include.",
)
@click.option(
    "--location",
    default=None,
    help="Optional location filter, matched against billing location.location.",
)
@click.option(
    "--output",
    "output_format",
    default="table",
    show_default=True,
    type=ANALYTICS_OUTPUT_CHOICES,
    help="Render output as a compact table or raw JSON.",
)
@click.option(
    "--ownership",
    default="all",
    show_default=True,
    type=ANALYTICS_OWNERSHIP_CHOICES,
    help="Filter by conservative ownership classification.",
)
def analytics_egress_cmd(target_date, billing_table, limit, kind, location, output_format, ownership):
    """Show daily network transfer billing rows from BigQuery billing export."""
    _validate_billing_table(billing_table)
    if limit <= 0:
        raise click.UsageError("--limit must be greater than 0.")

    where_clauses = [
        "cost > 0",
        "cost_type = 'regular'",
        "DATE(usage_start_time) = @target_date",
    ]
    parameters = [f"target_date:DATE:{target_date.date().isoformat()}"]
    if location:
        where_clauses.append("LOWER(location.location) = LOWER(@location)")
        parameters.append(f"location:STRING:{location}")
    if ownership != "all":
        where_clauses.append("ownership = @ownership")
        parameters.append(f"ownership:STRING:{ownership}")
    if kind == "internet":
        sku_filter = "sku LIKE '%Network Internet Data Transfer Out%'"
    elif kind == "inter-zone":
        sku_filter = "sku LIKE '%Network Inter Zone Data Transfer%'"
    else:
        sku_filter = (
            "("
            "sku LIKE '%Network Internet Data Transfer Out%' "
            "OR sku LIKE '%Network Inter Zone Data Transfer%'"
            ")"
        )

    query = f"""
WITH classified AS (
SELECT
  DATE(usage_start_time) AS usage_date,
  service.description AS service,
  sku.description AS sku,
  location.location AS location,
  project.id AS project_id,
  resource.name AS resource_name,
  resource.global_name AS global_name,
  CASE
    WHEN LOWER(COALESCE(resource.name, '')) LIKE 'llm_pruning%' THEN 'mine'
    WHEN LOWER(COALESCE(resource.global_name, '')) LIKE '%llm_pruning%' THEN 'mine'
    WHEN LOWER(COALESCE(resource.name, '')) LIKE 'yufeng%' THEN 'mine'
    WHEN LOWER(COALESCE(resource.global_name, '')) LIKE '%/yufeng%' THEN 'mine'
    WHEN service.description = 'Cloud Storage'
      AND COALESCE(resource.name, '') != ''
      AND LOWER(COALESCE(resource.name, '')) NOT LIKE 'llm_pruning%' THEN 'not_mine'
    WHEN service.description = 'Cloud Storage'
      AND COALESCE(resource.global_name, '') != ''
      AND LOWER(COALESCE(resource.global_name, '')) NOT LIKE '%llm_pruning%' THEN 'not_mine'
    ELSE 'unknown'
  END AS ownership,
  ROUND(SUM(cost), 2) AS cost,
  ROUND(SUM(usage.amount), 2) AS usage_amount,
  ANY_VALUE(usage.unit) AS usage_unit
FROM `{billing_table}`
WHERE
  {' AND '.join([c for c in where_clauses if c != 'ownership = @ownership'])}
GROUP BY usage_date, service, sku, location, project_id, resource_name, global_name, ownership
)
SELECT *
FROM classified
WHERE
  {sku_filter}
  {"AND ownership = @ownership" if ownership != "all" else ""}
ORDER BY cost DESC
LIMIT {limit}
""".strip()

    rows = _run_bq_query(query, parameters)
    _emit_analytics_rows(rows, output_format=output_format)


@analytics.command("storage")
def analytics_storage_cmd():
    """Show storage analytics."""
    click.echo("Storage analytics is not implemented yet.")


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
@click.option("--ssh-user", "-u", default="yx3038", help="SSH username for connecting to TPU VMs (default: gcloud default)")
@click.option("--debug", is_flag=True, default=False,
              help="Run interactively in the foreground with live output and no log files")
def worker_start(accelerator, zone, tpu_name, pricing, allocation_mode, startup_script, ssh_user, debug):
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
    if ssh_user:
        args.append(f"--ssh-user={ssh_user}")
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
@click.option("--ssh-user", "-u", default="yx3038", help="SSH username for connecting to the TPU VM")
def worker_ssh(worker_ref, worker_index, ssh_user):
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
    ssh_target = f"{ssh_user}@{tpu_name}" if ssh_user else tpu_name
    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "ssh", ssh_target,
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


@worker.command("sync")
@click.option("--zone", "-z", multiple=True, default=None,
              help="GCP zone(s) to query (default: all zones from existing workers + standard TPU zones)")
@click.option("--startup-script", "-s", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None, help="Set startup script for workers that don't have one")
def worker_sync(zone, startup_script):
    """Rebuild workers.json from tmux sessions, timeline logs, and GCP queued-resources."""
    startup_script = str(startup_script.resolve()) if startup_script else None
    existing = _read_workers()
    recovered_tmux = _sync_workers_from_tmux()

    # Determine zones to query
    if zone:
        zones = list(zone)
    else:
        zones = sorted({
            w["zone"] for w in existing.values() if w.get("zone")
        } | {
            w["zone"] for w in recovered_tmux.values() if w.get("zone")
        })
    if not zones:
        zones = ["us-central1-b", "us-central2-b", "us-east5-b"]

    # Discover resources from GCP
    owner = _get_owner_prefix()
    gcp_workers: dict = {}
    if owner:
        click.echo(f"Querying GCP queued-resources in {', '.join(zones)}...", err=True)
        gcp_workers = _discover_workers_from_gcp(zones, owner)
        click.echo(f"Found {len(gcp_workers)} queued resources on GCP for owner '{owner}'.", err=True)
    else:
        click.echo("No .jobman_owner found; skipping GCP discovery.", err=True)

    # Merge priority: existing > tmux/logs > GCP
    merged: dict = {}

    # Start with GCP-discovered resources (lowest priority)
    for worker_id, entry in gcp_workers.items():
        merged[worker_id] = entry

    # Override with tmux-recovered entries (have richer config)
    for worker_id, entry in recovered_tmux.items():
        merged[worker_id] = entry

    # Override with existing entries (most accurate)
    for worker_id, entry in existing.items():
        if worker_id in merged:
            merged[worker_id] = entry
            # Update status to running if tmux session exists
            if worker_id in recovered_tmux:
                merged[worker_id]["status"] = "running"

    if not merged:
        click.echo("No workers found from tmux sessions or GCP.")
        return

    if startup_script:
        applied = 0
        for w in merged.values():
            if not w.get("startup_script"):
                worker_log_dir = os.path.join(jobman_log_dir(), "workers", w["worker_id"])
                os.makedirs(worker_log_dir, exist_ok=True)
                dst = os.path.join(worker_log_dir, os.path.basename(startup_script))
                if not os.path.exists(dst):
                    shutil.copy2(startup_script, dst)
                w["startup_script"] = dst
                applied += 1
        if applied:
            click.echo(f"Applied startup script to {applied} workers without one.")

    _write_workers(merged)
    new_from_tmux = len(set(recovered_tmux) - set(existing))
    new_from_gcp = len(set(gcp_workers) - set(existing) - set(recovered_tmux))
    click.echo(f"Synced {len(merged)} workers ({new_from_tmux} from tmux, {new_from_gcp} from GCP).")
    for worker_id in sorted(merged):
        w = merged[worker_id]
        source = ""
        if worker_id not in existing:
            if worker_id in recovered_tmux:
                source = " (from tmux)"
            elif worker_id in gcp_workers:
                source = " (from GCP)"
        click.echo(f"  {worker_id}: {w['accelerator']} in {w['zone']} [{w.get('status', '?')}]{source}")


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
    click.echo(f"Host0 IP          : {_host0_ip(w)}")
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
            running_tasks_by_worker = _running_tasks_by_worker()
            click.echo(f"{'#':<4} {'WORKER':<28} {'ACCELERATOR':<12} {'ZONE':<20} {'STATUS':<10} {'VM':<10} {'QR'}")
            rows = []
            for idx, w in filtered_workers:
                pstatus, vm_status, qr_status = statuses.get(w["worker_id"], ("?", "?", "?"))
                if live_only and qr_status.upper() != "ACTIVE":
                    continue
                display_status = _worker_display_status(
                    w,
                    process_status=pstatus,
                    vm_status=vm_status,
                    qr_status=qr_status,
                    has_running_task=w["worker_id"] in running_tasks_by_worker,
                )
                rows.append((idx, w, display_status, vm_status, qr_status))
            if rows:
                for idx, w, display_status, vm_status, qr_status in rows:
                    click.echo(f"{idx:<4} {w['worker_id']:<28} {w['accelerator']:<12} {w['zone']:<20} "
                               f"{display_status:<10} {vm_status:<10} {qr_status}")
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


@cli.command("availability")
@click.option("--accelerator", "-a", default=None, help="Filter by accelerator type, e.g. v4-8")
@click.option("--zone", "-z", default=None, help="Filter by GCP zone, e.g. us-central2-b")
@click.option("--prefix", "-p", default=None, help="Filter workers by name prefix, e.g. yufeng-")
def availability(accelerator, zone, prefix):
    """Show TPU availability profile from timeline data."""
    from .availability import compute_availability, format_report

    stats_by_worker, stats_by_accel = compute_availability(
        worker_prefix=prefix,
        accelerator=accelerator,
        zone=zone,
    )
    if not stats_by_worker:
        click.echo("No timeline data found.")
        return
    click.echo(format_report(stats_by_worker, stats_by_accel))


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
    """Return worker process status: 'running', 'setup', 'dead', or 'stopped'."""
    stored = w.get("status", "")
    if stored == "running":
        session = f"jobman_{w['worker_id']}"
        result = subprocess.run(["tmux", "has-session", "-t", session], capture_output=True)
        if result.returncode != 0:
            return "dead"
        # Check timeline to distinguish setup vs running
        timeline = os.path.join(
            jobman_log_dir(), "workers", w["worker_id"], "timeline.jsonl"
        )
        try:
            with open(timeline, "rb") as f:
                # Seek to the last line efficiently
                f.seek(0, 2)
                pos = f.tell()
                if pos == 0:
                    return stored
                # Read backwards to find the last newline
                pos -= 1
                while pos > 0:
                    f.seek(pos)
                    if f.read(1) == b"\n":
                        break
                    pos -= 1
                else:
                    f.seek(0)
                last_line = f.readline().decode().strip()
                if last_line:
                    event = json.loads(last_line)
                    if event.get("event") == "bootstrap_started":
                        return "setup"
        except (OSError, ValueError):
            pass
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


def _host0_ip(worker: dict) -> str:
    """Return the host0 TPU VM external IP address, or '-' when unavailable."""
    try:
        result = subprocess.run(
            [
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "describe",
                worker["worker_id"],
                f"--zone={worker['zone']}",
                "--format=json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, KeyError):
        return "-"

    if result.returncode != 0:
        return "-"

    try:
        info = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return "-"

    endpoints = info.get("networkEndpoints")
    if isinstance(endpoints, list) and endpoints:
        first = endpoints[0]
        if isinstance(first, dict):
            access_config = first.get("accessConfig")
            ip = None
            if isinstance(access_config, dict):
                ip = access_config.get("externalIp")
            if not ip:
                candidate = first.get("ipAddress")
                if candidate and not str(candidate).startswith("10."):
                    ip = candidate
            if ip:
                return str(ip)

    return "-"


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


def _running_tasks_by_worker() -> set[str]:
    """Return worker IDs that currently own a running task."""
    q = Queue()
    return {
        task["worker_id"]
        for task in q.list()
        if task.get("status") == "running" and task.get("worker_id")
    }


def _worker_display_status(
    worker: dict,
    *,
    process_status: str,
    vm_status: str,
    qr_status: str,
    has_running_task: bool,
) -> str:
    """Return the user-facing worker status shown by `jobman status`."""
    process = (process_status or "").lower()
    vm = (vm_status or "").upper()
    qr = (qr_status or "").upper()

    if process in {"dead", "stopped"}:
        return process
    if qr != "ACTIVE" or vm in {"", "UNKNOWN", "CREATING", "NOT_FOUND"}:
        return "pending"
    if process == "setup":
        return "setup"
    if process == "running":
        return "running" if has_running_task else "idle"
    stored = str(worker.get("status", "") or "").lower()
    if stored in {"pending", "setup", "idle", "running"}:
        return stored
    return process or "unknown"


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
    if worker.get("ssh_user"):
        args.append(f"--ssh-user={worker['ssh_user']}")
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


def _validate_billing_table(value: str) -> str:
    table = value.strip()
    if not table:
        raise click.UsageError("Billing table cannot be empty.")
    if not BILLING_TABLE_PATTERN.fullmatch(table):
        raise click.UsageError(
            f"Invalid billing table '{value}'. Expected something like project.dataset.table."
        )
    return table


def _run_bq_query(query: str, parameters: list[str]) -> list[dict]:
    cmd = ["bq", "query", "--use_legacy_sql=false", "--format=json"]
    for param in parameters:
        cmd.append(f"--parameter={param}")
    cmd.append(query)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise click.ClickException(
            f"Failed to run 'bq query': {exc}. Is the BigQuery CLI installed and on PATH?"
        ) from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip() or "unknown error"
        raise click.ClickException(f"BigQuery query failed: {detail}")

    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Failed to parse BigQuery JSON output: {exc}") from exc

    if not isinstance(payload, list):
        raise click.ClickException("Unexpected BigQuery response format; expected a JSON list.")
    return payload


def _emit_analytics_rows(rows: list[dict], *, output_format: str) -> None:
    if output_format == "json":
        click.echo(json.dumps(rows, indent=2, sort_keys=True))
        return

    if not rows:
        click.echo("No billing rows matched the requested filters.")
        return

    click.echo(
        f"{'#':<4} {'DATE':<10} {'OWNER':<10} {'COST':>8} {'USAGE':>12} {'UNIT':<10} {'LOCATION':<14} "
        f"{'SERVICE':<18} {'RESOURCE':<28} SKU"
    )
    for idx, row in enumerate(rows, start=1):
        usage_date = str(row.get("usage_date", ""))[:10]
        ownership = str(row.get("ownership", "") or "-")[:10]
        cost = _format_decimal_like(row.get("cost"))
        usage_amount = _format_decimal_like(row.get("usage_amount"))
        usage_unit = str(row.get("usage_unit", "") or "")[:10]
        location = str(row.get("location", "") or "")[:14]
        service = str(row.get("service", "") or "")[:18]
        resource = str(row.get("resource_name") or row.get("global_name") or "-")[:28]
        sku = str(row.get("sku", "") or "")
        click.echo(
            f"{idx:<4} {usage_date:<10} {ownership:<10} {cost:>8} {usage_amount:>12} {usage_unit:<10} "
            f"{location:<14} {service:<18} {resource:<28} {sku}"
        )


def _format_decimal_like(value) -> str:
    if value in (None, ""):
        return "-"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _read_workers() -> dict:
    path = os.path.join(jobman_dir(), "workers.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _sync_workers_from_tmux() -> dict:
    """Rebuild the worker registry from running tmux sessions + timeline logs.

    For each ``jobman_*`` tmux session, reads the corresponding
    ``timeline.jsonl`` to recover the worker config (accelerator, zone, etc.).
    Returns the rebuilt registry dict.
    """
    result = subprocess.run(["tmux", "ls"], capture_output=True, text=True)
    if result.returncode != 0:
        return {}

    worker_names: list[str] = []
    for line in result.stdout.strip().split("\n"):
        session = line.split(":")[0]
        if session.startswith("jobman_"):
            worker_names.append(session[len("jobman_"):])

    log_root = os.path.join(jobman_log_dir(), "workers")
    registry: dict = {}
    for worker_id in sorted(worker_names):
        timeline = os.path.join(log_root, worker_id, "timeline.jsonl")
        if not os.path.exists(timeline):
            continue
        try:
            with open(timeline) as f:
                first = json.loads(f.readline())
        except (json.JSONDecodeError, OSError):
            continue
        if first.get("event") != "worker_started":
            continue

        accelerator = first["accelerator"]
        zone = first["zone"]

        # Find startup script snapshot in the log directory
        startup_script = None
        worker_log_dir = os.path.join(log_root, worker_id)
        try:
            for fname in os.listdir(worker_log_dir):
                if fname.endswith("_bootstrap.sh") or fname == "setup.sh":
                    startup_script = os.path.join(os.path.abspath(worker_log_dir), fname)
                    break
        except OSError:
            pass

        entry = {
            "worker_id": worker_id,
            "tpu_name": worker_id,
            "accelerator": accelerator,
            "zone": zone,
            "tpu_version": resolve_tpu_version(accelerator),
            "pricing": first.get("pricing", "spot"),
            "mode": first.get("allocation_mode", "queued-resources"),
            "startup_script": startup_script,
            "status": "running",
            "registered": first["time"],
            "pid": 0,
            "ssh_user": "yx3038",
        }
        registry[worker_id] = entry

    return registry


def _get_owner_prefix() -> str | None:
    """Read the owner name from .jobman_owner, or None if not set."""
    owner_path = _owner_file_path()
    if not owner_path.exists():
        return None
    try:
        return owner_path.read_text().strip()
    except OSError:
        return None


def _discover_workers_from_gcp(zones: list[str], owner: str) -> dict:
    """Query GCP queued-resources list across *zones* and return a registry
    of resources whose name starts with *owner*.
    """
    registry: dict = {}

    def query_zone(zone: str) -> list[dict]:
        try:
            result = subprocess.run(
                ["gcloud", "compute", "tpus", "queued-resources", "list",
                 f"--zone={zone}", "--format=json"],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                return []
            return json.loads(result.stdout.strip() or "[]")
        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
            return []

    with ThreadPoolExecutor(max_workers=min(len(zones), 8)) as ex:
        futures = {ex.submit(query_zone, z): z for z in zones}
        for fut in as_completed(futures):
            zone = futures[fut]
            for item in fut.result():
                qr_name = item["name"].split("/")[-1]
                if not qr_name.startswith(owner):
                    continue
                # Extract node-id (the TPU VM name)
                try:
                    node_id = item["tpu"]["nodeSpec"][0]["nodeId"]
                except (KeyError, IndexError, TypeError):
                    node_id = qr_name
                # Extract accelerator
                try:
                    accelerator = item["tpu"]["nodeSpec"][0]["node"]["acceleratorType"]
                except (KeyError, IndexError, TypeError):
                    accelerator = ""
                # Extract state
                state = item.get("state", {})
                if isinstance(state, dict):
                    state = state.get("state", "UNKNOWN")
                state = str(state).upper()
                # Extract runtime version
                try:
                    tpu_version = item["tpu"]["nodeSpec"][0]["node"]["runtimeVersion"]
                except (KeyError, IndexError, TypeError):
                    tpu_version = resolve_tpu_version(accelerator) if accelerator else ""
                # Determine pricing from presence of "spot" key
                pricing = "spot" if "spot" in item else "standard"

                registry[qr_name] = {
                    "worker_id": qr_name,
                    "tpu_name": node_id,
                    "accelerator": accelerator,
                    "zone": zone,
                    "tpu_version": tpu_version,
                    "pricing": pricing,
                    "mode": "queued-resources",
                    "startup_script": None,
                    "status": state.lower(),
                    "registered": item.get("createTime", ""),
                    "pid": 0,
                }

    return registry


def _write_workers(registry: dict) -> None:
    """Write the worker registry to workers.json with file locking."""
    path = os.path.join(jobman_dir(), "workers.json")
    lock_dir = path + ".d.lock"
    os.makedirs(jobman_dir(), exist_ok=True)
    with dir_lock(lock_dir):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(registry, f, indent=2)
        os.replace(tmp, path)


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
    lock_dir = path + ".d.lock"
    try:
        with dir_lock(lock_dir):
            if not os.path.exists(path):
                return
            with open(path) as f:
                registry = json.load(f)
            if tpu_name not in registry:
                return
            registry[tpu_name]["status"] = status
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(registry, f, indent=2)
            os.replace(tmp, path)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to update worker status: %s", e)


def _remove_worker(tpu_name: str) -> None:
    path = os.path.join(jobman_dir(), "workers.json")
    lock_dir = path + ".d.lock"
    try:
        with dir_lock(lock_dir):
            if not os.path.exists(path):
                return
            with open(path) as f:
                registry = json.load(f)
            registry.pop(tpu_name, None)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(registry, f, indent=2)
            os.replace(tmp, path)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to remove worker from registry: %s", e)
