# jobman/queue.py
"""
Queue mode for jobman: allows jobs to pick tasks from a shared queue
instead of running a fixed script.
"""

import os
import fcntl
import time
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from omegaconf import OmegaConf

from jobman.runner import MultiWorkerRunner


class Queue:
    """Manages a shared task queue with file-based state."""

    def __init__(self, config_path, logger):
        self.config_path = Path(config_path).resolve()
        self.state_path = self.config_path.with_name(
            self.config_path.stem + "_state.yaml"
        )
        self.lock_path = self.config_path.with_name(
            self.config_path.stem + ".lock"
        )
        self.logger = logger
        self._load_config()

    def _load_config(self):
        """Load the queue configuration file."""
        self.config = OmegaConf.load(self.config_path)
        self.repos = dict(self.config.get('repos', {}))
        self.defaults = dict(self.config.get('defaults', {}))

        # Global max_retries (default 3)
        self.max_retries = self.defaults.get('max_retries', 3)

        # Parse grouped format: accelerators -> tasks
        self.tasks_by_accelerator = {}
        self.all_tasks = {}  # task_id -> task dict with accelerator

        accelerators = self.config.get('accelerators', {})
        for accel, accel_config in accelerators.items():
            accel_config = dict(accel_config) if accel_config else {}
            tasks = accel_config.get('tasks', [])
            accel_defaults = {k: v for k, v in accel_config.items() if k != 'tasks'}

            self.tasks_by_accelerator[accel] = []
            for task in tasks:
                task = dict(task)
                task['accelerator'] = accel
                # Apply defaults: global -> accelerator-level -> task-level
                for key in ['env', 'repo', 'branch', 'workdir', 'max_jobs', 'max_retries']:
                    if key not in task:
                        task[key] = accel_defaults.get(key, self.defaults.get(key))

                self.tasks_by_accelerator[accel].append(task)
                self.all_tasks[task['id']] = task

    def get_tasks_for_accelerator(self, accelerator, zone=None):
        """Get ordered list of tasks for a specific accelerator and zone.

        Zone matching is strict: task zone must match job zone.
        Tasks without zone specified use the default zone from config.
        """
        tasks = self.tasks_by_accelerator.get(accelerator, [])
        if zone is None:
            return tasks
        default_zone = self.defaults.get('zone')
        # Strict zone matching: task zone (or default) must match job zone
        # If task has no zone and no default, it won't match any zone filter
        return [t for t in tasks if t.get('zone', default_zone) == zone]

    @contextmanager
    def _with_state_lock(self):
        """Context manager for locked state file access."""
        self.lock_path.touch(exist_ok=True)
        with open(self.lock_path, 'r+') as lock_fp:
            fcntl.flock(lock_fp, fcntl.LOCK_EX)
            try:
                if self.state_path.exists():
                    state = OmegaConf.load(self.state_path)
                    state = OmegaConf.to_container(state, resolve=True) or {}
                else:
                    state = {}
                yield state
                OmegaConf.save(OmegaConf.create(state), self.state_path)
            finally:
                fcntl.flock(lock_fp, fcntl.LOCK_UN)

    def cleanup_stale_entries(self, job_id):
        """Remove any stale running entries for this job from previous crashed runs."""
        with self._with_state_lock() as state:
            tasks_state = state.get('tasks', {})
            for task_id, task_state in tasks_state.items():
                running = task_state.get('running', [])
                stale = [r for r in running if r.get('job_id') == job_id]
                for entry in stale:
                    running.remove(entry)
                    history = task_state.setdefault('history', [])
                    history.append({
                        'job_id': job_id,
                        'started': entry.get('started'),
                        'ended': datetime.now().isoformat(),
                        'status': 'stale_cleanup'
                    })
                    self.logger.warning(
                        f"Cleaned up stale entry for task '{task_id}', job {job_id}"
                    )
                task_state['running'] = running

    def prioritize_task(self, task_id):
        """Mark a task as priority - will be picked up first."""
        if task_id not in self.all_tasks:
            self.logger.error(f"Task '{task_id}' not found")
            return False

        with self._with_state_lock() as state:
            tasks_state = state.setdefault('tasks', {})
            task_state = tasks_state.setdefault(task_id, {
                'done': False,
                'running': [],
                'history': []
            })
            task_state['priority'] = True
            self.logger.info(f"Task '{task_id}' marked as priority")
            return True

    def find_task_by_job(self, job_id):
        """Find which task a job is currently running."""
        with self._with_state_lock() as state:
            tasks_state = state.get('tasks', {})
            for task_id, task_state in tasks_state.items():
                running = task_state.get('running', [])
                for entry in running:
                    if entry.get('job_id') == job_id:
                        return task_id
        return None

    def find_and_claim_task(self, accelerator, job_id, zone=None, valid_job_ids=None):
        """
        Find the next available task for the given accelerator/zone and claim it.
        Priority tasks are checked first.

        Args:
            accelerator: TPU accelerator type (e.g., 'v6e-64')
            job_id: ID of the job claiming the task
            zone: TPU zone for strict zone matching
            valid_job_ids: Set of job IDs that are actually running. If provided,
                          running entries for jobs not in this set are ignored.

        Returns the task dict or None if no task available.
        """
        tasks = self.get_tasks_for_accelerator(accelerator, zone)
        if not tasks:
            self.logger.warning(f"No tasks defined for accelerator '{accelerator}'" +
                              (f" zone '{zone}'" if zone else ""))
            return None

        with self._with_state_lock() as state:
            tasks_state = state.setdefault('tasks', {})

            # Clean up stale running entries if valid_job_ids provided
            if valid_job_ids is not None:
                for tid, tstate in tasks_state.items():
                    running = tstate.get('running', [])
                    stale = [r for r in running if r.get('job_id') not in valid_job_ids]
                    for entry in stale:
                        running.remove(entry)
                        history = tstate.setdefault('history', [])
                        history.append({
                            'job_id': entry.get('job_id'),
                            'started': entry.get('started'),
                            'ended': datetime.now().isoformat(),
                            'status': 'stale_cleanup'
                        })
                        self.logger.warning(f"Auto-cleaned stale entry for task '{tid}', job {entry.get('job_id')}")
                    tstate['running'] = running

            # First pass: check priority tasks
            for task in tasks:
                task_id = task['id']
                task_state = tasks_state.get(task_id, {})
                if task_state.get('priority') and not task_state.get('done', False):
                    # Found priority task - try to claim it
                    result = self._try_claim_task(task, task_state, job_id, tasks_state)
                    if result:
                        # Clear priority flag after claiming
                        task_state['priority'] = False
                        return result

            # Second pass: normal order
            for task in tasks:
                task_id = task['id']
                task_state = tasks_state.setdefault(task_id, {
                    'done': False,
                    'running': [],
                    'history': []
                })
                result = self._try_claim_task(task, task_state, job_id, tasks_state)
                if result:
                    return result

        return None

    def _try_claim_task(self, task, task_state, job_id, tasks_state=None):
        """Try to claim a task. Returns task dict if claimed, None otherwise."""
        task_id = task['id']

        # Skip if done
        if task_state.get('done', False):
            return None

        # Check dependencies - all depends_on tasks must be completed
        depends_on = task.get('depends_on')
        if depends_on and tasks_state is not None:
            # Normalize to list
            if isinstance(depends_on, str):
                depends_on = [depends_on]
            for dep_id in depends_on:
                dep_state = tasks_state.get(dep_id, {})
                if not dep_state.get('done', False):
                    return None  # Dependency not done yet
                # Also check it wasn't a failure
                history = dep_state.get('history', [])
                if history and history[-1].get('status') == 'failed':
                    return None  # Dependency failed, don't run

        # Check max_jobs limit
        max_jobs = task.get('max_jobs', 1)
        running = task_state.get('running', [])

        # Check if this job is already running this task
        already_running = any(r.get('job_id') == job_id for r in running)
        if already_running:
            self.logger.info(f"Resuming task '{task_id}' (already claimed)")
            return dict(task)

        if len(running) >= max_jobs:
            return None

        # Found a task! Claim it.
        running.append({
            'job_id': job_id,
            'started': datetime.now().isoformat()
        })
        task_state['running'] = running

        # Update run count
        run_count = task_state.get('run_count', 0) + 1
        task_state['run_count'] = run_count

        self.logger.info(
            f"Claimed task '{task_id}' (run #{run_count}, "
            f"{len(running)}/{max_jobs} concurrent)"
        )
        return dict(task)

    def release_task(self, task_id, job_id, status='interrupted'):
        """
        Release a task (remove from running, update status).

        Status can be:
        - 'completed': task finished successfully (exit code 0)
        - 'failed': task errored (non-zero exit code)
        - 'interrupted': task was preempted/killed
        - 'stale_cleanup': cleaned up from previous crashed run

        For 'failed' status, the task will be retried up to max_retries times.
        On 'completed', the fail count resets.
        """
        with self._with_state_lock() as state:
            tasks_state = state.setdefault('tasks', {})
            task_state = tasks_state.get(task_id)
            if not task_state:
                return

            # Find and remove from running
            running = task_state.get('running', [])
            entry = next((r for r in running if r.get('job_id') == job_id), None)
            if entry:
                running.remove(entry)
                # Add to history
                history = task_state.setdefault('history', [])
                history.append({
                    'job_id': job_id,
                    'started': entry.get('started'),
                    'ended': datetime.now().isoformat(),
                    'status': status
                })

            task_state['running'] = running

            if status == 'completed':
                task_state['done'] = True
                task_state['fail_count'] = 0  # Reset fail count on success
                self.logger.info(f"Task '{task_id}' completed by job {job_id}")
            elif status == 'failed':
                # Get max_retries from task config (default to global, then 3)
                task_config = self.all_tasks.get(task_id, {})
                max_retries = task_config.get('max_retries') or self.max_retries

                # Increment fail count and check if we should retry
                fail_count = task_state.get('fail_count', 0) + 1
                task_state['fail_count'] = fail_count

                if fail_count >= max_retries:
                    task_state['done'] = True
                    self.logger.error(f"Task '{task_id}' FAILED by job {job_id} (attempt {fail_count}/{max_retries}) - max retries reached, giving up")
                else:
                    # Don't mark as done - will be retried
                    self.logger.warning(f"Task '{task_id}' FAILED by job {job_id} (attempt {fail_count}/{max_retries}) - will retry")
            else:
                # interrupted - will be retried (doesn't count as a failure)
                self.logger.info(f"Task '{task_id}' interrupted by job {job_id} - will retry")

    def release_all_for_job(self, job_id):
        """
        Release all tasks currently claimed by a job.
        Called when a job is stopped/deleted to prevent stale entries.
        Returns list of released task_ids.
        """
        released = []
        with self._with_state_lock() as state:
            tasks_state = state.get('tasks', {})
            for task_id, task_state in tasks_state.items():
                running = task_state.get('running', [])
                entry = next((r for r in running if r.get('job_id') == job_id), None)
                if entry:
                    running.remove(entry)
                    # Add to history
                    history = task_state.setdefault('history', [])
                    history.append({
                        'job_id': job_id,
                        'started': entry.get('started'),
                        'ended': datetime.now().isoformat(),
                        'status': 'interrupted'
                    })
                    task_state['running'] = running
                    released.append(task_id)
        return released

    def get_repo_info(self, repo_key):
        """Get full repo URL and branch from short key."""
        repo = self.repos.get(repo_key)
        if repo is None:
            return None, None
        if isinstance(repo, str):
            return repo, self.defaults.get('branch', 'main')
        return repo.get('url'), repo.get('branch', self.defaults.get('branch', 'main'))

    def get_task_workdir(self, task):
        """Get the working directory for a task."""
        return task.get('workdir') or self.defaults.get('workdir') or task.get('repo')


class QueueRunner(MultiWorkerRunner):
    """Runs queue tasks on TPU workers."""

    def __init__(self, cfg, logger, queue, env_wrapper=None, prefix_list=None):
        super().__init__(cfg, logger, action='queue')
        self.queue = queue
        self.env_wrapper = env_wrapper
        # prefix_list is a list of dicts: [{'name': 'x', 'workers': 0 or 'all', 'run': '...'}, ...]
        self.prefix_list = prefix_list or []
        self.current_task = None
        self.job_id = cfg.job.id
        self.workers = [0]
        self.num_workers = cfg.tpu.num_workers

        # Job directory for marker files
        self.job_dir = Path(cfg.job.dir)

        # Check if prefix was already completed successfully (persisted across reboots)
        self._prefix_done_marker = self.job_dir / '.prefix_done'
        self.prefix_done = self._prefix_done_marker.exists()

        # Task logs directory: jobs/tasks/<task_id>/ (sibling to user folders)
        # Structure: jobs/<user>/<job_id> -> tasks dir is jobs/tasks/
        self.tasks_log_dir = self.job_dir.parent.parent / "tasks"  # jobs/tasks/
        self.tasks_log_dir.mkdir(parents=True, exist_ok=True)

        # Machine info for logging
        self.machine_info = {
            'job_id': cfg.job.id,
            'tpu_name': cfg.tpu.name,
            'accelerator': cfg.tpu.accelerator,
            'zone': cfg.tpu.zone,
        }

        # Job-level environment variables (exported before each task)
        # Resolve any OmegaConf interpolations (e.g., ${tpu.name})
        env_cfg = getattr(cfg.job, 'environment', None)
        if env_cfg:
            self.environment = OmegaConf.to_container(env_cfg, resolve=True)
        else:
            self.environment = {}

        # Queue logger will be set by job.py after execute_queue creates it
        self.queue_logger = None

    def _qlog(self, msg):
        """Log to queue.log if available."""
        if self.queue_logger:
            self.queue_logger.info(msg)

    def _get_task_log_dir(self, task_id):
        """Get or create task-specific log directory.

        Hidden tasks (hide: true) go in tasks/hide/<task_id>/
        Normal tasks go in tasks/<task_id>/
        """
        # Check if task is hidden
        task_config = self.queue.all_tasks.get(task_id, {})
        is_hidden = task_config.get('hide', False)

        if is_hidden:
            task_dir = self.tasks_log_dir / "hide" / task_id
        else:
            task_dir = self.tasks_log_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    def _get_run_number(self, task_id):
        """Get the next run number for this task (counts existing run files)."""
        if hasattr(self, '_current_run_number'):
            return self._current_run_number

        task_dir = self._get_task_log_dir(task_id)
        # Count existing run_* directories or log files
        existing_runs = list(task_dir.glob("run_*_worker_0.log"))
        self._current_run_number = len(existing_runs) + 1
        return self._current_run_number

    def _per_worker_log(self, i):
        """Override to log task output to task-specific directory."""
        if self.current_task:
            task_id = self.current_task['id']
            task_dir = self._get_task_log_dir(task_id)
            run_num = self._get_run_number(task_id)
            # Format: run_001_job100001_worker_0.log (run number first for sorting)
            return task_dir / f"run_{run_num:03d}_job{self.job_id}_worker_{i}.log"
        else:
            # Fallback for prefix runs (no task yet)
            return Path(self.cfg.job.dir) / "logs" / f"{self.action}_worker_{i}.log"

    def _write_task_metadata(self, task_id, status='started'):
        """Write machine info to task log directory."""
        task_dir = self._get_task_log_dir(task_id)
        run_num = self._get_run_number(task_id)
        meta_file = task_dir / f"run_{run_num:03d}_job{self.job_id}_meta.yaml"

        from datetime import datetime
        metadata = {
            'run_number': run_num,
            **self.machine_info,
            'status': status,
            'timestamp': datetime.now().isoformat(),
        }
        OmegaConf.save(OmegaConf.create(metadata), meta_file)

    def run_prefix(self, force=False):
        """
        Run prefix scripts once (SSH keys, gcloud config, etc).
        Each prefix in prefix_list runs in order on its specified workers.

        If a prefix has 'required: true' and fails, the job stops.
        Otherwise, errors are non-fatal and job continues.

        Returns True if all prefixes ran successfully, False if any had errors.
        Raises RuntimeError if a required prefix fails.
        """
        if self.prefix_done and not force:
            self.logger.info("[Prefix] Already executed, skipping (use --force to re-run)")
            return True

        if not self.prefix_list:
            self.logger.info("[Prefix] No prefix configured, skipping")
            self.prefix_done = True
            return True

        self.logger.info(f"[Prefix] Running {len(self.prefix_list)} prefix script(s)...")

        all_success = True
        for i, prefix_cfg in enumerate(self.prefix_list):
            prefix_name = prefix_cfg.get('name', f'prefix_{i+1}')
            prefix_script_content = prefix_cfg.get('run', '')
            prefix_workers = prefix_cfg.get('workers', 0)
            is_required = prefix_cfg.get('required', False)

            if not prefix_script_content.strip():
                self.logger.info(f"[Prefix:{prefix_name}] Empty script, skipping")
                continue

            # Determine which workers to run on
            if prefix_workers == 'all':
                self.workers = list(range(self.num_workers))
                workers_desc = f"all {self.num_workers} workers"
            elif isinstance(prefix_workers, list):
                self.workers = prefix_workers
                workers_desc = f"workers {prefix_workers}"
            else:
                self.workers = [int(prefix_workers)]
                workers_desc = f"worker {prefix_workers}"

            self.logger.info(f"[Prefix:{prefix_name}] Running on {workers_desc}..." + (" (required)" if is_required else ""))

            # For required prefixes, don't wrap with error suppression
            if is_required:
                prefix_script = f"""
echo "========================================"
echo "[Prefix:{prefix_name}] Running on $(hostname)..."
echo "========================================"

{prefix_script_content}

echo "========================================"
echo "[Prefix:{prefix_name}] Completed on $(hostname)"
echo "========================================"
"""
            else:
                # Non-required: wrap with error handling - continue on error
                prefix_script = f"""
set +e  # Don't exit on error
echo "========================================"
echo "[Prefix:{prefix_name}] Running on $(hostname)..."
echo "========================================"

{prefix_script_content}

PREFIX_EXIT=$?
if [ $PREFIX_EXIT -ne 0 ]; then
    echo "========================================"
    echo "[Prefix:{prefix_name}] WARNING: Had errors (exit code $PREFIX_EXIT)"
    echo "[Prefix:{prefix_name}] Continuing anyway..."
    echo "========================================"
fi
exit 0  # Always succeed - prefix errors are non-fatal
"""
            self.full_cmd = prefix_script

            # Track per-worker results
            self._worker_results = {}
            try:
                result = self.setup()

                # Log per-worker status summary
                success_workers = [w for w, ok in self._worker_results.items() if ok]
                failed_workers = [w for w, ok in self._worker_results.items() if not ok]

                if success_workers:
                    self.logger.info(f"[Prefix:{prefix_name}] Success on workers: {success_workers}")
                if failed_workers:
                    self.logger.error(f"[Prefix:{prefix_name}] Failed on workers: {failed_workers}")

                if result:
                    self.logger.info(f"[Prefix:{prefix_name}] Completed successfully on all workers")
                else:
                    if is_required:
                        self.logger.error(f"[Prefix:{prefix_name}] FAILED (required) - stopping job")
                        raise RuntimeError(f"Required prefix '{prefix_name}' failed on workers: {failed_workers}")
                    else:
                        self.logger.warning(f"[Prefix:{prefix_name}] Had issues but continuing...")
                        all_success = False
            except RuntimeError:
                # Re-raise RuntimeError for required prefix failures
                raise
            except Exception as e:
                if is_required:
                    self.logger.error(f"[Prefix:{prefix_name}] FAILED with exception (required) - stopping job: {e}")
                    raise RuntimeError(f"Required prefix '{prefix_name}' failed: {e}")
                else:
                    self.logger.warning(f"[Prefix:{prefix_name}] Failed with exception: {e}, continuing anyway...")
                    all_success = False

        self.prefix_done = True
        # Write marker file so prefix is skipped on reboot
        self._prefix_done_marker.write_text(datetime.now().isoformat())
        if all_success:
            self.logger.info("[Prefix] All prefix scripts completed successfully")
        else:
            self.logger.warning("[Prefix] Some prefix scripts had issues but continuing...")
        return all_success

    def _build_env_exports(self):
        """Build export statements and gcloud config for job-level environment."""
        lines = ["# Job environment setup (errors are non-fatal)"]
        lines.append("set +e  # Don't exit on gcloud/env errors")

        # Set gcloud zone config (avoids "gcloud config set compute/zone" errors)
        zone = getattr(self.cfg.tpu, 'zone', None)
        if zone:
            lines.append(f"gcloud config set compute/zone {zone} 2>/dev/null || true")

        # Set gcloud project if available
        project = getattr(self.cfg.tpu, 'project', None)
        if project:
            lines.append(f"gcloud config set project {project} 2>/dev/null || true")

        # Export job environment variables
        if self.environment:
            lines.append("# Environment variables")
            for key, value in self.environment.items():
                # Escape single quotes in values
                escaped_value = str(value).replace("'", "'\\''")
                lines.append(f"export {key}='{escaped_value}'")

        lines.append("set -e  # Re-enable exit on error for actual task")
        return "\n".join(lines)

    def run_task(self, task):
        """Run a single task: clone repo and execute command."""
        self.current_task = task
        task_id = task['id']

        # Clear cached run number for new task
        if hasattr(self, '_current_run_number'):
            del self._current_run_number

        # Write metadata at start
        self._write_task_metadata(task_id, status='running')

        # Get repo info
        repo_key = task.get('repo')
        repo_url, default_branch = self.queue.get_repo_info(repo_key)
        if not repo_url:
            self.logger.error(f"Unknown repo key: {repo_key}")
            self._qlog(f"    ERROR: Unknown repo key '{repo_key}'")
            return False

        branch = task.get('branch') or default_branch
        workdir = self.queue.get_task_workdir(task)
        run_cmd = task.get('run', '')
        env_name = task.get('env')

        # Log to queue.log
        self._qlog(f"    Cloning {repo_key} (branch: {branch})...")
        self._qlog(f"    Running: {run_cmd[:80]}{'...' if len(run_cmd) > 80 else ''}")

        # Build environment exports
        env_exports = self._build_env_exports()

        # Build the full command:
        # 1. Export job environment variables
        # 2. Delete old repo and clone fresh
        # 3. cd to workdir and run the command
        # (prefix already ran once during setup)

        clone_script = f"""
set -e
echo "========================================"
echo "[Queue] Task: {task_id}"
echo "[Queue] Repo: {repo_key} -> {repo_url}"
echo "[Queue] Branch: {branch}"
echo "[Queue] Workdir: ~/{workdir}"
echo "========================================"

{env_exports}

echo "[Queue] Removing old repo directory..."
rm -rf ~/{workdir}

echo "[Queue] Cloning repository..."
git clone --branch {branch} {repo_url} ~/{workdir}

echo "[Queue] Entering workdir..."
cd ~/{workdir}

echo "[Queue] Running task command..."
# Capture output and exit code (multihost_runner may exit 0 even on errors)
set +e
TASK_OUTPUT_FILE=$(mktemp)
{run_cmd} 2>&1 | tee "$TASK_OUTPUT_FILE"
TASK_EXIT_CODE=${{PIPESTATUS[0]}}

# Keep set +e for all checks to avoid early exit
# Check for SUCCESS first - if multihost_runner says success, trust it
if grep -q "Main command completed successfully" "$TASK_OUTPUT_FILE" 2>/dev/null || \
   grep -q "Multihost runner finished successfully" "$TASK_OUTPUT_FILE" 2>/dev/null; then
    echo "========================================"
    echo "[Queue] Task '{task_id}' finished successfully"
    echo "========================================"
    rm -f "$TASK_OUTPUT_FILE" 2>/dev/null
    exit 0
fi

# Check for explicit failure - multihost_runner exits 0 but prints error messages
if grep -q "Main command failed" "$TASK_OUTPUT_FILE" 2>/dev/null || \
   grep -q "Main command finished with errors" "$TASK_OUTPUT_FILE" 2>/dev/null || \
   grep -q "failed with error code" "$TASK_OUTPUT_FILE" 2>/dev/null; then
    echo "========================================"
    echo "[Queue] Task '{task_id}' FAILED (detected error in output)"
    echo "========================================"
    rm -f "$TASK_OUTPUT_FILE" 2>/dev/null
    exit 1
fi

# Check for preemption - only if not success and not explicit failure
if grep -q "TPU .* PREEMPTED" "$TASK_OUTPUT_FILE" 2>/dev/null || \
   grep -q "going through a maintenance event" "$TASK_OUTPUT_FILE" 2>/dev/null || \
   grep -q "TPU is being preempted" "$TASK_OUTPUT_FILE" 2>/dev/null; then
    echo "========================================"
    echo "[Queue] Task '{task_id}' INTERRUPTED (TPU preempted)"
    echo "========================================"
    rm -f "$TASK_OUTPUT_FILE" 2>/dev/null
    exit 143  # Special code for preemption (SIGTERM)
fi
rm -f "$TASK_OUTPUT_FILE" 2>/dev/null

if [ $TASK_EXIT_CODE -ne 0 ]; then
    echo "========================================"
    echo "[Queue] Task '{task_id}' FAILED with exit code $TASK_EXIT_CODE"
    echo "========================================"
    exit $TASK_EXIT_CODE
fi

echo "========================================"
echo "[Queue] Task '{task_id}' finished successfully"
echo "========================================"
"""

        # Wrap with environment activation if needed
        if self.env_wrapper and env_name:
            full_cmd = self.env_wrapper(clone_script, env_name)
        else:
            full_cmd = clone_script

        self.full_cmd = full_cmd

        # Run on specified workers
        task_workers = task.get('workers', [0])
        if task_workers == 'all':
            self.workers = list(range(self.cfg.tpu.num_workers))
        elif isinstance(task_workers, int):
            self.workers = [task_workers]
        else:
            self.workers = list(task_workers)

        self._qlog(f"    Executing on {len(self.workers)} worker(s)...")

        # Run the task and get exit code
        start_time = time.time()
        success = self.setup()
        elapsed = time.time() - start_time
        task_id = task['id']

        if success:
            self._write_task_metadata(task_id, status='completed')
            self._qlog(f"    Finished in {elapsed/60:.1f}min")
            return 'completed'

        # Check if failure was due to SSH/connection error (likely preemption)
        # Key indicator: SSH retries were attempted in runner.py
        # This means the connection itself failed, not the command
        if hasattr(self, '_retry_attempted') and self._retry_attempted:
            self._write_task_metadata(task_id, status='interrupted')
            self._qlog(f"    INTERRUPTED (SSH failed) after {elapsed/60:.1f}min")
            return 'interrupted'

        # Exit code 143 = preemption detected in script (SIGTERM equivalent)
        # Exit code 255 = SSH connection failure
        if hasattr(self, '_last_exit_code') and self._last_exit_code in (143, 255):
            self._write_task_metadata(task_id, status='interrupted')
            self._qlog(f"    INTERRUPTED (exit {self._last_exit_code}) after {elapsed/60:.1f}min")
            return 'interrupted'

        # Any other non-zero exit code is a real task failure
        self._write_task_metadata(task_id, status='failed')
        exit_code = getattr(self, '_last_exit_code', '?')
        self._qlog(f"    FAILED (exit {exit_code}) after {elapsed/60:.1f}min")
        return 'failed'

    def _setup_worker(self, i: int) -> bool:
        """Override to capture exit code for preemption detection and track per-worker results."""
        self.logger.info(f"{self.action} worker {i}: running setup...")
        for rc in self._get_setup_steps(i):
            self._last_exit_code = rc
            self.logger.info(f"{self.action} worker {i}: command returned exit code {rc}")
            if rc != 0:
                self.logger.error(f"{self.action} worker {i}: setup failed (exit code {rc}).")
                # Track per-worker result for logging
                if hasattr(self, '_worker_results'):
                    self._worker_results[i] = False
                return False
        # Track per-worker result for logging
        if hasattr(self, '_worker_results'):
            self._worker_results[i] = True
        return True

    def _get_setup_steps(self, i):
        """Execute the task command."""
        # Fewer retries for task execution - fail faster on preemption
        # timeout=0 disables timeout since tasks can run for hours
        from jobman.runner import SSH_TASK_RETRIES
        yield self._ssh(i, self.full_cmd, max_retries=SSH_TASK_RETRIES, timeout=0)


def create_env_wrapper(cfg):
    """Create a function that wraps commands with environment activation."""
    env_type = getattr(cfg.job, 'env_type', None)

    if env_type == 'venv':
        def wrapper(cmd, env_name=None):
            name = env_name or cfg.venv.name
            return f"source ~/{name}/bin/activate && {cmd}"
        return wrapper

    elif env_type == 'conda':
        def wrapper(cmd, env_name=None):
            name = env_name or cfg.conda.name
            return f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate {name} && {cmd}"
        return wrapper

    return None
