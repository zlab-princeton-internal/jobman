import os
import json
import time
import logging
import subprocess
from pathlib import Path
from omegaconf import OmegaConf

from jobman.tpu import TPU
from jobman.ssh import SSH
from jobman.gcsfuse import GCSFUSE
from jobman.envs.docker import DOCKER
from jobman.envs.conda import CONDA
from jobman.envs.venv import VENV
from jobman.command import COMMAND
from jobman.queue import Queue, QueueRunner, create_env_wrapper

from jobman.utils import setup_logger

class Job:

    # Valid job statuses
    STATUS_QUEUEING = "QUEUEING"      # Waiting for TPU allocation
    STATUS_RUNNING = "RUNNING"        # TPU ready, job executing
    STATUS_PREEMPTED = "PREEMPTED"    # TPU was preempted
    STATUS_IDLE = "IDLE"              # TPU ready but job stopped
    STATUS_PREFIX_FAIL = "PREFIX_FAIL"  # Prefix script failed
    STATUS_DEAD = "DEAD"              # TPU deleted/gone

    def __init__(self, cfg):

        self.cfg = cfg

        self.id = cfg.job.id
        self.name = cfg.job.name
        self.dir = cfg.job.dir
        self.loop = cfg.job.get('loop', False)

        self.logger = setup_logger(log_file=Path(self.dir) / 'logs' / 'job.log')
        self.tpu = TPU(cfg)
        self.ssh = SSH(cfg, self.logger) if getattr(cfg, "ssh", None) is not None else None
        self.gcsfuse = GCSFUSE(cfg, self.logger) if getattr(cfg, "gcsfuse", None) is not None else None
        self.command = COMMAND(cfg, self.logger) if getattr(cfg, "command", None) is not None else None

        self.env_type = getattr(cfg.job, "env_type", None)
        if self.env_type is None:
            self.env = None
        elif self.env_type == 'docker':
            self.env = DOCKER(cfg, self.logger) if getattr(cfg, "docker", None) is not None else None
        elif self.env_type == 'conda':
            self.env = CONDA(cfg, self.logger) if getattr(cfg, "conda", None) is not None else None
        elif self.env_type == 'venv':
            self.env = VENV(cfg, self.logger) if getattr(cfg, "venv", None) is not None else None
        else:
            raise Exception(f"Unrecognized env_type {self.env_type}")

        # Queue mode setup
        self.queue_mode = self._is_queue_mode_enabled()
        self.queue = None
        self.queue_runner = None
        if self.queue_mode:
            queue_cfg = cfg.job.queue
            queue_config_path = queue_cfg.get('config')
            prefix_cfg = queue_cfg.get('prefix', [])
            # Normalize prefix config to list format
            # Supports both old format (string) and new format (list of dicts)
            # Convert OmegaConf types to native Python first
            if OmegaConf.is_config(prefix_cfg):
                prefix_cfg = OmegaConf.to_container(prefix_cfg, resolve=True)

            if isinstance(prefix_cfg, str):
                # Old format: single string runs on worker 0
                prefix_list = [{'name': 'prefix', 'workers': 0, 'run': prefix_cfg}] if prefix_cfg.strip() else []
            elif isinstance(prefix_cfg, list):
                # New format: list of prefix scripts
                prefix_list = []
                for item in prefix_cfg:
                    if isinstance(item, str):
                        prefix_list.append({'name': 'prefix', 'workers': 0, 'run': item})
                    else:
                        # Item is already a dict after to_container conversion
                        prefix_list.append(item)
            else:
                prefix_list = []

            if queue_config_path:
                # Resolve relative paths from job directory or current directory
                if not Path(queue_config_path).is_absolute():
                    # Try relative to the original config file location first
                    base_dir = Path(cfg.job.dir).parent.parent.parent  # jobs/<user>/<job_id> -> project root
                    queue_config_path = base_dir / queue_config_path
                self.queue = Queue(queue_config_path, self.logger)
                self.queue_runner = QueueRunner(
                    cfg, self.logger, self.queue,
                    env_wrapper=create_env_wrapper(cfg),
                    prefix_list=prefix_list
                )

    def _is_queue_mode_enabled(self):
        """Check if queue mode is enabled in config."""
        queue_cfg = getattr(self.cfg.job, 'queue', None)
        if queue_cfg is None:
            return False
        return queue_cfg.get('enabled', False)

    def write_status(self, status):
        """Write job status to .job_status file for fast status checks."""
        status_file = Path(self.dir) / '.job_status'
        status_file.write_text(status)

    @staticmethod
    def read_status(job_dir):
        """Read job status from .job_status file. Returns None if not found."""
        status_file = Path(job_dir) / '.job_status'
        if status_file.exists():
            return status_file.read_text().strip()
        return None

    def request(self):
        self.logger.info("Checking TPU status...")
        self.write_status(self.STATUS_QUEUEING)
        ready, tpu_state = self.tpu.check_and_maybe_delete()

        if not ready:
            # Write more specific status if TPU was preempted/terminated
            if tpu_state in {"PREEMPTED", "TERMINATED", "STOPPED", "SUSPENDED"}:
                self.write_status(self.STATUS_PREEMPTED)
            self.logger.info("Requesting TPU...")
            success = self.tpu.request()
            if not success:
                self.logger.error("TPU allocation failed.")
                return False

            self.cfg.tpu.ips = self.tpu.get_ips()
            OmegaConf.save(self.cfg, Path(self.dir) / "config.yaml")

        self.write_status(self.STATUS_RUNNING)
        return True

    def setup(self, force=False):
        for module in [self.ssh, self.gcsfuse, self.env]:
            if module and not module.setup(force=force):
                return False
        return True
    
    def execute(self):
        self.command.full_cmd = self.env.patch_command(self.command.base_cmd) if self.env else self.command.base_cmd
        return self.command.setup()

    def _get_valid_job_ids(self):
        """Get set of job IDs that are actually running (have active tmux sessions)."""
        from jobman.jobman import JobMan
        valid_ids = set()
        try:
            jm = JobMan()
            with jm.with_meta_lock() as meta:
                for job_id, job_meta in meta.items():
                    session_name = job_meta.get('session_name', f'job_{job_id}')
                    if jm.check_tmux_session(session_name):
                        valid_ids.add(job_id)
        except Exception as e:
            self.logger.warning(f"Could not get valid job IDs: {e}")
            # Return None to skip stale filtering if we can't determine valid IDs
            return None
        return valid_ids

    def execute_queue(self, force=False):
        """Execute tasks from the queue until none remain for this accelerator."""
        if not self.queue or not self.queue_runner:
            self.logger.error("Queue mode enabled but queue not initialized")
            return False

        # Create separate queue log file
        queue_log_path = Path(self.dir) / 'logs' / 'queue.log'
        self.queue_logger = setup_logger(
            name='queue',
            log_file=queue_log_path,
            stdout=False
        )
        # Pass queue logger to runner for real-time logging
        self.queue_runner.queue_logger = self.queue_logger

        # Run prefix ONCE before processing tasks
        # If a required prefix fails, go idle (return False) instead of retrying
        try:
            # Clear prefix_done marker if forcing re-run
            if force:
                prefix_done_marker = Path(self.dir) / '.prefix_done'
                if prefix_done_marker.exists():
                    prefix_done_marker.unlink()
                self.queue_runner.prefix_done = False

            self.queue_runner.run_prefix(force=force)
            # Clear any previous prefix fail marker
            prefix_fail_marker = Path(self.dir) / '.prefix_failed'
            if prefix_fail_marker.exists():
                prefix_fail_marker.unlink()
        except RuntimeError as e:
            self.logger.error(f"Required prefix failed: {e}")
            self.logger.error("Job going idle - fix the issue and reboot with --force-prefix")
            # Write status
            self.write_status(self.STATUS_PREFIX_FAIL)
            # Clear prefix_done marker on failure
            prefix_done_marker = Path(self.dir) / '.prefix_done'
            if prefix_done_marker.exists():
                prefix_done_marker.unlink()
            return False

        accelerator = self.cfg.tpu.accelerator
        zone = self.cfg.tpu.zone
        self.logger.info(f"Queue mode: looking for tasks matching accelerator '{accelerator}' zone '{zone}'")
        self.queue_logger.info(f"{'='*60}")
        self.queue_logger.info(f"Queue mode started - accelerator: {accelerator}, zone: {zone}")
        self.queue_logger.info(f"Queue config: {self.queue.config_path}")
        self.queue_logger.info(f"{'='*60}")

        # Clean up any stale entries from previous crashed runs
        self.queue.cleanup_stale_entries(self.id)

        tasks_completed = 0
        tasks_failed = 0
        current_task = None

        try:
            while True:
                # Reload config to pick up any changes (e.g., modified commands)
                self.queue._load_config()

                # Get list of valid (actually running) job IDs to filter stale entries
                valid_job_ids = self._get_valid_job_ids()

                # Find and claim next task
                task = self.queue.find_and_claim_task(accelerator, self.id, zone, valid_job_ids)
                if task is None:
                    self.logger.info("No tasks available, waiting 60s before checking again...")
                    self.queue_logger.info("No tasks available, sleeping 60s...")
                    time.sleep(60)
                    continue

                current_task = task
                task_id = task['id']
                self.logger.info(f"Starting task '{task_id}'")

                # Log task details to queue log
                self.queue_logger.info(f"")
                self.queue_logger.info(f">>> TASK: {task_id}")
                self.queue_logger.info(f"    Repo: {task.get('repo')} (branch: {task.get('branch', 'main')})")
                self.queue_logger.info(f"    Env: {task.get('env')}")
                self.queue_logger.info(f"    Run: {task.get('run')}")

                try:
                    status = self.queue_runner.run_task(task)
                    if status == 'completed':
                        # Task completed successfully (exit code 0)
                        self.queue.release_task(task_id, self.id, status='completed')
                        tasks_completed += 1
                        self.logger.info(f"Task '{task_id}' completed successfully")
                        self.queue_logger.info(f"<<< COMPLETED: {task_id}")
                    elif status == 'interrupted':
                        # SSH failure (likely preemption) - will retry
                        self.queue.release_task(task_id, self.id, status='interrupted')
                        self.logger.warning(f"Task '{task_id}' interrupted (SSH failed, likely preemption)")
                        self.queue_logger.warning(f"<<< INTERRUPTED: {task_id} (SSH failure, will retry)")
                        raise RuntimeError("SSH connection failed - TPU likely preempted")
                    else:
                        # Task failed (non-zero exit code from actual command)
                        self.queue.release_task(task_id, self.id, status='failed')
                        tasks_failed += 1
                        self.logger.error(f"Task '{task_id}' failed (non-zero exit)")
                        self.queue_logger.error(f"<<< FAILED: {task_id} (non-zero exit code)")
                        # Continue to next task instead of stopping
                except RuntimeError:
                    # SSH failure - break out and let the loop restart
                    raise
                except Exception as e:
                    self.logger.exception(f"Task '{task_id}' failed with exception: {e}")
                    self.queue.release_task(task_id, self.id, status='interrupted')
                    self.queue_logger.warning(f"<<< INTERRUPTED: {task_id} ({e})")
                    raise  # Re-raise to trigger loop restart if enabled

                current_task = None

        except KeyboardInterrupt:
            # Clean up current task on interrupt
            if current_task:
                self.queue.release_task(current_task['id'], self.id, status='interrupted')
            raise

        except Exception:
            # Clean up current task on error (likely preemption)
            if current_task:
                self.queue.release_task(current_task['id'], self.id, status='interrupted')
            raise

        self.logger.info(f"Queue mode: completed {tasks_completed}, failed {tasks_failed}")
        self.queue_logger.info(f"")
        self.queue_logger.info(f"{'='*60}")
        self.queue_logger.info(f"Queue session ended - completed: {tasks_completed}, failed: {tasks_failed}")
        self.queue_logger.info(f"{'='*60}")
        return tasks_completed > 0 or True  # Return True even if no tasks (queue exhausted)

    def run(self, force=False, force_prefix=False):
        while True:
            try:
                # Request and setup phases (same for both modes)
                for action, step in zip(
                    ['request', 'setup'],
                    [self.request, lambda: self.setup(force=force)]
                ):
                    if not step():
                        self.logger.error(f"Job {self.id} {action} failed.")
                        break
                else:
                    # Setup succeeded, now execute
                    if self.queue_mode:
                        self.logger.info("Running in QUEUE MODE")
                        # force_prefix forces only prefix re-run, force forces everything including prefix
                        prefix_force = force or force_prefix
                        if not self.execute_queue(force=prefix_force):
                            self.logger.error(f"Job {self.id} queue execution failed.")
                    else:
                        if not self.execute():
                            self.logger.error(f"Job {self.id} execution failed.")

            except KeyboardInterrupt:
                self.logger.warning("Job interrupted by user")
                return False

            except Exception as e:
                self.logger.exception(f"Job failed with error: {e}")

            if not self.loop:
                break
            self.logger.info("Retrying job due to error...")

        self.logger.info(f"Job {self.id} finished successfully.")
