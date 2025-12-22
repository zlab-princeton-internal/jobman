# jobman/cli.py
import os
import click
import subprocess
from pathlib import Path
from omegaconf import OmegaConf

from jobman.jobman import JobMan

from jobman.utils import setup_logger
from jobman.profilers.billing_report import main as run_billing_report
from jobman.profilers.quota_report import main as run_quota_report
from jobman.profilers.storage_report import main as run_storage_report

def get_cfg(job_id):
    jm = JobMan()  
    user = os.environ.get("USER")
    
    with jm.with_meta_lock() as meta:
        if job_id not in meta:
            raise KeyError(f"Job ID not found: {job_id}")
        
    job_meta = meta[job_id]
    owner = job_meta['user']
    if owner != user:
        raise PermissionError(f"Meta owner mismatch for {job_id}: owner={owner}, current_user={user}")
    
    config_path = job_meta['config_path']
    cfg = OmegaConf.load(config_path)
    
    return cfg
        
@click.group()
def cli():
    """JobMan CLI: manage TPU jobs."""
    pass

@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
def create(config_path):
    jm = JobMan()  
    job_id = jm.create_job(config_path)
    jm.start_job(job_id)
    
@cli.command(name="resume")
@click.argument("job_id", type=str)
def resume(job_id):
    """Cancel a running job."""
    jm = JobMan()
    jm.start_job(job_id)
    
@cli.command(name="stop")
@click.argument("job_ids", type=str, nargs=-1, required=True)
def stop(job_ids):
    """Stop one or more running jobs."""
    from jobman.queue import Queue

    jm = JobMan()

    for job_id in job_ids:
        # Check if this is a queue mode job and release the task
        try:
            cfg = get_cfg(job_id)
            queue_cfg = getattr(cfg.job, 'queue', None)
            if queue_cfg and queue_cfg.get('enabled', False):
                queue_config_path = queue_cfg.get('config')
                if queue_config_path:
                    from pathlib import Path
                    if not Path(queue_config_path).is_absolute():
                        base_dir = Path(cfg.job.dir).parent.parent.parent
                        queue_config_path = base_dir / queue_config_path

                    logger = setup_logger(stdout=False)
                    q = Queue(queue_config_path, logger)
                    task_id = q.find_task_by_job(job_id)
                    if task_id:
                        q.release_task(task_id, job_id, status='interrupted')
                        print(f"Released task '{task_id}' (marked as interrupted)")
        except Exception:
            pass  # If we can't release the task, still stop the job

        jm.stop_job(job_id)
    
@cli.command(name="reboot")
@click.argument("job_ids", type=str, nargs=-1, required=True)
@click.option("--force", is_flag=True, help="Force re-run ALL setup steps (SSH, gcsfuse, venv, prefix)")
@click.option("--force-prefix", is_flag=True, help="Force re-run only the prefix scripts")
def reboot(job_ids, force, force_prefix):
    """Stop and restart one or more jobs."""
    jm = JobMan()
    for job_id in job_ids:
        jm.stop_job(job_id)
        jm.start_job(job_id, force=force, force_prefix=force_prefix)

@cli.command(name="requeue")
@click.argument("job_id", type=str)
def requeue(job_id):
    """Stop a queue job, mark current task as interrupted, restart to pick new task."""
    from jobman.queue import Queue

    jm = JobMan()
    cfg = get_cfg(job_id)

    # Check if this is a queue mode job
    queue_cfg = getattr(cfg.job, 'queue', None)
    if not queue_cfg or not queue_cfg.get('enabled', False):
        print(f"Job {job_id} is not in queue mode. Use 'jobman reboot' instead.")
        return

    # Get the queue config path
    queue_config_path = queue_cfg.get('config')
    if not queue_config_path:
        print(f"Job {job_id} has no queue config specified.")
        return

    # Resolve path
    from pathlib import Path
    if not Path(queue_config_path).is_absolute():
        base_dir = Path(cfg.job.dir).parent.parent.parent
        queue_config_path = base_dir / queue_config_path

    logger = setup_logger(stdout=True)
    q = Queue(queue_config_path, logger)

    # Find what task this job is running
    task_id = q.find_task_by_job(job_id)
    if task_id:
        print(f"Job {job_id} is running task '{task_id}'")
        q.release_task(task_id, job_id, status='interrupted')
        print(f"Task '{task_id}' marked as interrupted - will be retried")
    else:
        print(f"Job {job_id} is not currently running any task")

    # Stop and restart
    jm.stop_job(job_id)
    jm.start_job(job_id)
    print(f"Job {job_id} restarted - will pick next available task")
    
@cli.command(name="delete")
@click.argument("job_ids", type=str, nargs=-1, required=True)
def delete(job_ids):
    """Delete one or more jobs (stops job and deletes TPU)."""
    jm = JobMan()
    for job_id in job_ids:
        jm.delete_job(job_id)

@cli.command(name="clean")
@click.argument("job_ids", type=str, nargs=-1, required=True)
def clean(job_ids):
    """Clean one or more jobs (deletes job, TPU, and logs)."""
    jm = JobMan()
    for job_id in job_ids:
        jm.clean_job(job_id)

@cli.command(name="list")
def list_jobs():
    """List all jobs and their status."""
    jm = JobMan()
    jm.list_jobs()
    
@cli.command()
def billing():
    """Run billing report profiler."""
    run_billing_report()

@cli.command()
def quota():
    """Run quota usage profiler."""
    run_quota_report()

@cli.command()
def storage():
    """Run storage usage profiler."""
    run_storage_report()

@cli.command(name="queue")
@click.argument("config_path", type=click.Path(), default="queue_machines/tasks.yaml", required=False)
@click.option("--reset", type=str, default=None, help="Reset a specific task (mark as not done, clear running)")
@click.option("--reset-all", is_flag=True, help="Reset all tasks in the queue")
@click.option("--reset-failed", is_flag=True, help="Reset all failed tasks (mark as not done so they retry)")
@click.option("--prioritize", type=str, default=None, help="Mark a task as priority (will be picked first)")
@click.option("--task", type=str, default=None, help="Show details for a specific task")
@click.option("--cleanup", is_flag=True, help="Remove stale 'running' entries for jobs that are not actually running")
@click.option("--done", is_flag=True, help="Show all completed tasks (including hidden)")
def queue_status(config_path, reset, reset_all, reset_failed, prioritize, task, cleanup, done):
    """View or manage task queue status."""
    from tabulate import tabulate
    from jobman.queue import Queue

    if not Path(config_path).exists():
        print(f"Queue config not found: {config_path}")
        return

    logger = setup_logger(stdout=True)
    q = Queue(config_path, logger)

    if cleanup:
        # Find jobs that are actually running (have active tmux sessions AND exist in metadata)
        from jobman.jobman import JobMan
        jm = JobMan()
        running_job_ids = set()
        known_job_ids = set()  # All jobs in metadata (running or not)
        try:
            with jm.with_meta_lock() as meta:
                for job_id, job_meta in meta.items():
                    known_job_ids.add(job_id)
                    session_name = job_meta.get('session_name', f'job_{job_id}')
                    if jm.check_tmux_session(session_name):
                        running_job_ids.add(job_id)
        except Exception:
            pass

        # Clean up stale entries:
        # 1. Jobs that exist in metadata but don't have active tmux sessions
        # 2. Jobs that don't exist in metadata at all (deleted jobs)
        cleaned_count = 0
        with q._with_state_lock() as state:
            tasks_state = state.get('tasks', {})
            for task_id, task_state in tasks_state.items():
                running = task_state.get('running', [])
                stale = []
                for r in running:
                    job_id = r.get('job_id')
                    # Stale if: job not running OR job not in metadata (deleted)
                    if job_id not in running_job_ids or job_id not in known_job_ids:
                        stale.append(r)

                for entry in stale:
                    running.remove(entry)
                    # Add to history as interrupted
                    history = task_state.setdefault('history', [])
                    from datetime import datetime
                    job_id = entry.get('job_id')
                    reason = "deleted" if job_id not in known_job_ids else "stopped"
                    history.append({
                        'job_id': job_id,
                        'started': entry.get('started'),
                        'ended': datetime.now().isoformat(),
                        'status': 'stale_cleanup'
                    })
                    print(f"  Cleaned: task '{task_id}' was claimed by {reason} job {job_id}")
                    cleaned_count += 1
                task_state['running'] = running

        if cleaned_count > 0:
            print(f"\nCleaned up {cleaned_count} stale entry/entries.")
        else:
            print("No stale entries found.")
        return

    if done:
        # Show all completed tasks (including hidden ones)
        from tabulate import tabulate
        with q._with_state_lock() as state:
            tasks_state = state.get('tasks', {})

            rows = []
            for accel, tasks in q.tasks_by_accelerator.items():
                for task in tasks:
                    task_id = task['id']
                    is_hidden = task.get('hide', False)
                    task_state = tasks_state.get(task_id, {})

                    if not task_state.get('done', False):
                        continue

                    # Get last status from history
                    history = task_state.get('history', [])
                    if history:
                        last_entry = history[-1]
                        status = last_entry.get('status', 'completed').upper()
                        ended = last_entry.get('ended', '?')
                        job_id = last_entry.get('job_id', '?')
                    else:
                        status = "DONE"
                        ended = "?"
                        job_id = "?"

                    run_count = task_state.get('run_count', 0)
                    fail_count = task_state.get('fail_count', 0)
                    hidden_marker = "(H)" if is_hidden else ""

                    rows.append([
                        f"{task_id} {hidden_marker}".strip(),
                        accel,
                        status,
                        job_id,
                        run_count,
                        fail_count if fail_count > 0 else "-",
                        ended[:19] if ended != "?" else "?"  # Trim timestamp
                    ])

            if rows:
                print(f"\n=== Completed Tasks ({len(rows)} total) ===\n")
                headers = ["Task ID", "Accelerator", "Status", "Last Job", "Runs", "Fails", "Completed At"]
                print(tabulate(rows, headers=headers, tablefmt="github"))
            else:
                print("\nNo completed tasks found.")
        return

    if reset_all:
        with q._with_state_lock() as state:
            state['tasks'] = {}
        print("All tasks have been reset.")
        return

    if reset_failed:
        reset_count = 0
        with q._with_state_lock() as state:
            tasks_state = state.get('tasks', {})
            for task_id, task_state in tasks_state.items():
                # Check if task is done and last status was 'failed'
                if task_state.get('done', False):
                    history = task_state.get('history', [])
                    if history and history[-1].get('status') == 'failed':
                        task_state['done'] = False
                        task_state['running'] = []
                        task_state['fail_count'] = 0  # Reset fail count too
                        reset_count += 1
                        print(f"  Reset: {task_id}")
        print(f"\nReset {reset_count} failed task(s).")
        return

    if prioritize:
        if q.prioritize_task(prioritize):
            print(f"Task '{prioritize}' marked as priority - will be picked first.")
        else:
            print(f"Task '{prioritize}' not found.")
        return

    if reset:
        with q._with_state_lock() as state:
            tasks_state = state.get('tasks', {})
            if reset in tasks_state:
                tasks_state[reset] = {'done': False, 'running': [], 'history': []}
                print(f"Task '{reset}' has been reset.")
            elif reset in q.all_tasks:
                # Task exists in config but not in state yet
                print(f"Task '{reset}' has no state to reset.")
            else:
                print(f"Task '{reset}' not found.")
        return

    if task:
        # Show details for a specific task
        task_config = q.all_tasks.get(task)
        if not task_config:
            print(f"Task '{task}' not found in config.")
            return

        print(f"\n=== Task: {task} ===\n")

        # Task configuration
        print("Configuration:")
        print(f"  Accelerator: {task_config.get('accelerator', 'default')}")
        print(f"  Repo: {task_config.get('repo', 'N/A')}")
        print(f"  Branch: {task_config.get('branch', 'main')}")
        print(f"  Env: {task_config.get('env', 'N/A')}")
        print(f"  Workdir: {task_config.get('workdir', 'N/A')}")
        print(f"  Run: {task_config.get('run', 'N/A')}")
        print(f"  Max Jobs: {task_config.get('max_jobs', 1)}")
        print(f"  Max Retries: {task_config.get('max_retries') or q.max_retries}")
        depends_on = task_config.get('depends_on')
        if depends_on:
            if isinstance(depends_on, str):
                depends_on = [depends_on]
            print(f"  Depends On: {', '.join(depends_on)}")

        # Task state
        with q._with_state_lock() as state:
            tasks_state = state.get('tasks', {})
            task_state = tasks_state.get(task, {})

            print("\nState:")
            done = task_state.get('done', False)
            running = task_state.get('running', [])
            run_count = task_state.get('run_count', 0)
            fail_count = task_state.get('fail_count', 0)
            priority = task_state.get('priority', False)

            if done:
                history = task_state.get('history', [])
                if history:
                    last_status = history[-1].get('status', 'completed')
                    print(f"  Status: DONE ({last_status})")
                else:
                    print(f"  Status: DONE")
            elif running:
                print(f"  Status: RUNNING ({len(running)}/{task_config.get('max_jobs', 1)})")
            elif priority:
                print(f"  Status: PRIORITY (pending)")
            else:
                print(f"  Status: PENDING")

            print(f"  Run Count: {run_count}")
            print(f"  Fail Count: {fail_count}")
            if priority:
                print(f"  Priority: Yes")

            if running:
                print(f"\nCurrently Running:")
                for r in running:
                    print(f"  - Job: {r.get('job_id', '?')}, Started: {r.get('started_at', '?')}")

            history = task_state.get('history', [])
            if history:
                print(f"\nHistory (last 10 runs):")
                for entry in history[-10:]:
                    job_id = entry.get('job_id', '?')
                    status = entry.get('status', '?')
                    started = entry.get('started_at', '?')
                    ended = entry.get('ended_at', '?')
                    print(f"  - [{status.upper()}] Job {job_id}: {started} -> {ended}")

        # Task logs location
        from jobman.jobman import JobMan
        jm = JobMan()
        try:
            with jm.with_meta_lock() as meta:
                # Find any job to get the jobs directory
                for job_id, job_meta in meta.items():
                    job_dir = job_meta.get('job_dir')
                    if job_dir:
                        tasks_log_dir = Path(job_dir).parent.parent / "tasks" / task
                        if tasks_log_dir.exists():
                            log_files = sorted(tasks_log_dir.glob("run_*.log"))
                            print(f"\nLogs: {tasks_log_dir}")
                            if log_files:
                                print(f"  Files: {len(log_files)} run log(s)")
                                # Show last 3 log files
                                for lf in log_files[-3:]:
                                    print(f"    - {lf.name}")
                        else:
                            print(f"\nLogs: No logs yet (would be at jobs/tasks/{task}/)")
                        break
        except Exception:
            pass

        return

    # Get all jobman jobs to count TPU machines per accelerator
    # Include jobs with active tmux (RUNNING) and stopped jobs (IDLE) - but not deleted ones
    from jobman.jobman import JobMan
    from jobman.job import Job
    jm = JobMan()
    tpu_machines_by_accel = {}  # accelerator -> list of job_ids
    running_jobs = set()  # Jobs with active tmux sessions AND TPU ready
    queueing_jobs = set()  # Jobs with active tmux but waiting for TPU
    try:
        with jm.with_meta_lock() as meta:
            for job_id, job_meta in meta.items():
                job_dir = job_meta.get('job_dir')
                if not job_dir:
                    continue
                config_path_job = Path(job_dir) / 'config.yaml'
                if config_path_job.exists():
                    from omegaconf import OmegaConf
                    job_cfg = OmegaConf.load(config_path_job)
                    # Only count queue-enabled jobs
                    if job_cfg.job.get('queue', {}).get('enabled', False):
                        accel = job_cfg.tpu.accelerator
                        if accel not in tpu_machines_by_accel:
                            tpu_machines_by_accel[accel] = []
                        tpu_machines_by_accel[accel].append(job_id)
                        # Check job status from .job_status file
                        session_name = job_meta.get('session_name', f'job_{job_id}')
                        has_tmux = jm.check_tmux_session(session_name)
                        if has_tmux:
                            job_status = Job.read_status(job_dir)
                            if job_status in {"RUNNING"}:
                                running_jobs.add(job_id)
                            elif job_status in {"QUEUEING", "PREEMPTED"} or job_status is None:
                                # QUEUEING, PREEMPTED, or no status file = waiting for TPU
                                queueing_jobs.add(job_id)
                            else:
                                # Other statuses with tmux = assume queueing
                                queueing_jobs.add(job_id)
    except Exception:
        pass  # If we can't read meta, just skip TPU count

    # Display queue status grouped by accelerator
    with q._with_state_lock() as state:
        tasks_state = state.get('tasks', {})

        for accel, tasks in q.tasks_by_accelerator.items():
            rows = []
            # Stats for this accelerator
            total_tasks = len(tasks)
            done_count = 0
            running_count = 0
            pending_count = 0
            waiting_count = 0
            failed_count = 0
            hidden_done = 0
            hidden_pending = 0
            hidden_waiting = 0
            active_jobs = set()  # Unique job IDs currently running tasks

            for task in tasks:
                task_id = task['id']
                max_jobs = task.get('max_jobs', 1)
                is_hidden = task.get('hide', False)

                task_state = tasks_state.get(task_id, {})
                done = task_state.get('done', False)
                # Filter stale entries: only show jobs that are actually running
                running = [r for r in task_state.get('running', [])
                          if r.get('job_id') in running_jobs]
                run_count = task_state.get('run_count', 0)
                fail_count = task_state.get('fail_count', 0)

                # Determine status
                if done:
                    done_count += 1
                    # Check last history entry to show more specific status
                    history = task_state.get('history', [])
                    if history:
                        last_status = history[-1].get('status', 'completed')
                        if last_status == 'failed':
                            status = "FAILED"
                            failed_count += 1
                        elif last_status == 'interrupted':
                            status = "INTERRUPTED"
                        else:
                            status = "DONE"
                    else:
                        status = "DONE"
                elif running:
                    running_count += 1
                    status = f"RUNNING ({len(running)}/{max_jobs})"
                    for r in running:
                        active_jobs.add(r.get('job_id', '?'))
                else:
                    priority = task_state.get('priority', False)
                    max_retries = task.get('max_retries') or q.max_retries

                    # Check if waiting on dependencies
                    depends_on = task.get('depends_on')
                    waiting_on_dep = False
                    if depends_on:
                        if isinstance(depends_on, str):
                            depends_on = [depends_on]
                        for dep_id in depends_on:
                            dep_state = tasks_state.get(dep_id, {})
                            if not dep_state.get('done', False):
                                waiting_on_dep = True
                                break
                            # Also check if dependency failed (won't run)
                            dep_history = dep_state.get('history', [])
                            if dep_history and dep_history[-1].get('status') == 'failed':
                                waiting_on_dep = True
                                break

                    if priority:
                        status = "PRIORITY"
                        pending_count += 1
                    elif waiting_on_dep:
                        status = "WAITING"
                        waiting_count += 1
                    elif fail_count > 0:
                        status = f"RETRY ({fail_count}/{max_retries})"
                        pending_count += 1
                    else:
                        status = "PENDING"
                        pending_count += 1

                # Decide whether to show this task
                # Hidden tasks only show if RUNNING or FAILED
                show_task = True
                if is_hidden:
                    if status == "DONE" or status == "INTERRUPTED":
                        show_task = False
                        hidden_done += 1
                    elif status == "WAITING":
                        show_task = False
                        hidden_waiting += 1
                    elif status == "PENDING":
                        show_task = False
                        hidden_pending += 1
                    # RUNNING, FAILED, RETRY, PRIORITY always show

                if show_task:
                    running_jobs_str = ", ".join(r.get('job_id', '?') for r in running) if running else "-"
                    fails_str = str(fail_count) if fail_count > 0 else "-"
                    rows.append([
                        task_id,
                        status,
                        running_jobs_str,
                        run_count,
                        fails_str
                    ])

            # Also check for jobs running tasks that were removed from config
            # (tasks in state but not in current config)
            for task_id, task_state in tasks_state.items():
                if task_id not in q.all_tasks:  # Task removed from config
                    running = [r for r in task_state.get('running', [])
                              if r.get('job_id') in running_jobs]
                    for r in running:
                        active_jobs.add(r.get('job_id', '?'))

            if rows or hidden_done > 0 or hidden_pending > 0 or hidden_waiting > 0:
                print(f"\n=== {accel} ===")
                machine_ids = set(tpu_machines_by_accel.get(accel, []))
                accel_running = running_jobs & machine_ids
                accel_queueing = queueing_jobs & machine_ids
                # Working = TPU ready AND running a task
                working_ids = sorted(active_jobs & accel_running)
                # Idle = TPU ready but NOT running a task
                idle_ids = sorted(accel_running - active_jobs)
                # Waiting for TPU = has tmux but TPU not ready (queueing/preempted)
                waiting_tpu_ids = sorted(accel_queueing)
                # Stopped = no tmux session (job was stopped but may still be in metadata)
                stopped_ids = sorted(machine_ids - accel_running - accel_queueing)
                print(f"Working: {' '.join(working_ids) if working_ids else 'none'} ({len(working_ids)})")
                print(f"Idle: {' '.join(idle_ids) if idle_ids else 'none'} ({len(idle_ids)})")
                if waiting_tpu_ids:
                    print(f"Waiting for TPU: {' '.join(waiting_tpu_ids)} ({len(waiting_tpu_ids)})")
                if stopped_ids:
                    print(f"Stopped: {' '.join(stopped_ids)} ({len(stopped_ids)})")
                print(f"Tasks: {done_count} done, {running_count} running, {pending_count} pending, {waiting_count} waiting ({total_tasks} total)")
                if rows:
                    headers = ["Task ID", "Status", "Running Jobs", "Runs", "Fails"]
                    print(tabulate(rows, headers=headers, tablefmt="github"))
                if hidden_done > 0 or hidden_pending > 0 or hidden_waiting > 0:
                    hidden_parts = []
                    if hidden_done > 0:
                        hidden_parts.append(f"{hidden_done} done")
                    if hidden_pending > 0:
                        hidden_parts.append(f"{hidden_pending} pending")
                    if hidden_waiting > 0:
                        hidden_parts.append(f"{hidden_waiting} waiting")
                    print(f"Hidden: {', '.join(hidden_parts)} (use --task <id> to view)")

        # Show tasks that were removed from config but still running (once, outside accel loop)
        removed_running_tasks = []
        for task_id, task_state in tasks_state.items():
            if task_id not in q.all_tasks:
                running = [r for r in task_state.get('running', [])
                          if r.get('job_id') in running_jobs]
                if running:
                    job_ids = [r.get('job_id', '?') for r in running]
                    removed_running_tasks.append((task_id, job_ids))
        if removed_running_tasks:
            print(f"\n=== Removed tasks still running ===")
            for task_id, job_ids in removed_running_tasks:
                print(f"  {task_id}: {' '.join(job_ids)}")

    # Show state file location
    print(f"\nState file: {q.state_path}")
    
@cli.command(name="ssh")
@click.argument("job_id")
def ssh(job_id):
    cfg = get_cfg(job_id)
    remote_user = cfg.job.remote_user
    ips = cfg.tpu.get('ips', [])
    if not ips:
        raise ValueError(f'Host0 IP for Job {job_id} not found!')
    host0_ip = next(ip.get("external_ip") for ip in ips if ip.worker == 0)
    ssh_cmd = ["ssh", f"{remote_user}@{host0_ip}"]
    subprocess.run(ssh_cmd)
    
@cli.command("run")
@click.argument("job_id")
@click.option("--cmd-only", is_flag=True, help="Run the main command only")
@click.option("--force", is_flag=True, help="Force re-run ALL setup steps (SSH, gcsfuse, venv, prefix)")
@click.option("--force-prefix", is_flag=True, help="Force re-run only the prefix scripts")
def run(job_id, cmd_only, force, force_prefix):
    """Run a job by job_id using job.py's argparse main."""
    from jobman.job import Job
    cfg = get_cfg(job_id)
    job = Job(cfg)
    if cmd_only:
        job.execute()
    else:
        job.run(force=force, force_prefix=force_prefix)

# @cli.command(name="tpu")
# @click.argument("job_id")
# def tpu(job_id):
#     """Request a TPU for a given job_id."""
#     # 直接调用 tpu.main()，并把参数传进去
#     from jobman.tpu import TPU
#     cfg = get_cfg(job_id)
#     tpu = TPU(cfg)
#     tpu.request()
    
# @cli.command(name="ssh")
# @click.argument("job_id")
# def ssh(job_id):
#     """Request a TPU for a given job_id."""
#     from jobman.ssh import SSH
#     cfg = get_cfg(job_id)
#     ssh = SSH(cfg)
#     ssh.setup()
    
# @cli.command(name="gcsfuse")
# @click.argument("job_id")
# def gcsfuse(job_id):
#     """Request a TPU for a given job_id."""
#     from jobman.gcsfuse import GCSFUSE
#     cfg = get_cfg(job_id)
#     gcsfuse = GCSFUSE(cfg)
#     gcsfuse.setup()
    
# @cli.command(name="docker")
# @click.argument("job_id")
# def docker(job_id):
#     """Request a TPU for a given job_id."""
#     from jobman.envs.docker import DOCKER
#     cfg = get_cfg(job_id)
#     logger = setup_logger(log_file=Path(cfg.job.dir)/'logs'/'job.log')
#     docker = DOCKER(cfg, logger)
#     docker.setup()
  
# @cli.command(name="conda")
# @click.argument("job_id")
# def conda(job_id):
#     """Request a TPU for a given job_id."""
#     from jobman.envs.conda import CONDA
#     cfg = get_cfg(job_id)
#     logger = setup_logger(log_file=Path(cfg.job.dir)/'logs'/'job.log')
#     conda = CONDA(cfg, logger)
#     conda.setup()  

# @cli.command(name="venv")
# @click.argument("job_id")
# def venv(job_id):
#     """Request a TPU for a given job_id."""
#     from jobman.envs.venv import VENV
#     cfg = get_cfg(job_id)
#     logger = setup_logger(log_file=Path(cfg.job.dir)/'logs'/'job.log')
#     venv = VENV(cfg, logger)
#     venv.setup()  