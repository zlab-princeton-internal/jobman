"""Worker loop: TPU holder + task executor."""

import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone

from .queue import Queue
from .tpu import TPU, AllocationMode, Pricing, DEFAULT_TPU_VERSION
from .utils import get_logger, jobman_dir

logger = get_logger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Worker:
    def __init__(
        self,
        tpu_name: str,
        accelerator: str,
        zone: str,
        tpu_version: str = DEFAULT_TPU_VERSION,
        pricing: Pricing = "spot",
        allocation_mode: AllocationMode = "tpu-vm",
    ):
        self.worker_id = tpu_name
        self.accelerator = accelerator
        self.zone = zone
        self.tpu = TPU(tpu_name, zone, accelerator, tpu_version, pricing, allocation_mode)
        self.queue = Queue()
        self._state_dir = jobman_dir()
        self._log_dir = os.path.join(self._state_dir, "logs", "workers", tpu_name)
        os.makedirs(self._log_dir, exist_ok=True)

        # Attach file handler to root jobman logger so all submodules
        # (tpu, queue, worker) log to the same file.
        log_file = os.path.join(self._log_dir, "worker.log")
        get_logger("jobman", log_file=log_file)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._register(status="running")
        logger.info("Worker %s started (accelerator=%s, zone=%s)",
                    self.worker_id, self.accelerator, self.zone)
        try:
            while True:
                try:
                    self._ensure_tpu_ready()
                    task = self.queue.claim(self.accelerator, self.zone, self.worker_id)
                    if task is None:
                        logger.debug("No pending tasks, sleeping 30s...")
                        time.sleep(30)
                        continue

                    success, preempted = self._run_task(task)

                    if preempted:
                        self.queue.release(task["id"], "interrupted")
                        self._handle_preemption()
                    else:
                        self.queue.release(task["id"], "done" if success else "failed")

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.exception("Unhandled error in worker loop: %s", e)
                    time.sleep(10)
        except KeyboardInterrupt:
            logger.info("Worker %s shutting down", self.worker_id)
        finally:
            self._register(status="stopped")

    # ------------------------------------------------------------------
    # TPU management
    # ------------------------------------------------------------------

    def _ensure_tpu_ready(self) -> None:
        status = self.tpu.status()
        if status == "READY":
            return
        logger.info("TPU %s status=%s, provisioning...", self.worker_id, status)
        if status not in ("NOT_FOUND", "CREATING"):
            self.tpu.delete()
        if status != "CREATING":
            self.tpu.request()
        self.tpu.wait_ready()

    def _handle_preemption(self) -> None:
        logger.info("Handling preemption for TPU %s", self.worker_id)
        try:
            self.tpu.delete()
        except Exception as e:
            logger.warning("Error deleting preempted TPU: %s", e)

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    def _run_task(self, task: dict) -> tuple[bool, bool]:
        """Run a task. Returns (success, preempted)."""
        task_id = task["id"]
        log_dir = os.path.join(self._state_dir, "logs", "tasks", task_id)
        os.makedirs(log_dir, exist_ok=True)
        run_count = task.get("run_count", 1)
        worker_suffix = self.worker_id.split("-")[-1]  # e.g. "00001" from "v4-8-us-central2-b-00001"
        log_file = os.path.join(log_dir, f"run_{run_count}_worker_{worker_suffix}.log")

        logger.info("Running task %s (%s) on TPU %s", task_id, task["name"], self.worker_id)

        num_workers = self.tpu.get_num_workers()
        script_path = task["script"]
        remote_script = f"/tmp/jobman_{task_id}.sh"

        # 1. SCP script to TPU worker 0
        scp_cmd = [
            "gcloud", "compute", "tpus", "tpu-vm", "scp",
            script_path,
            f"{self.worker_id}:{remote_script}",
            f"--zone={self.zone}",
            "--worker=0",
            "--ssh-flag=-o ConnectionAttempts=2",
            "--ssh-flag=-o ConnectTimeout=30",
            "--ssh-flag=-o StrictHostKeyChecking=no",
            "--ssh-flag=-o UserKnownHostsFile=/dev/null",
            "--ssh-flag=-o BatchMode=yes",
        ]
        scp_result = subprocess.run(scp_cmd, capture_output=True, text=True)
        if scp_result.returncode != 0:
            logger.error("SCP failed for task %s: %s", task_id, scp_result.stderr)
            # Check if preempted
            if self._is_preempted():
                return False, True
            return False, False

        # 2. SSH and run on worker 0
        env_str = (
            f"JOBMAN_TPU_NAME={self.worker_id} "
            f"JOBMAN_ZONE={self.zone} "
            f"JOBMAN_NUM_WORKERS={num_workers}"
        )
        inline = f"chmod +x {remote_script} && {env_str} bash {remote_script}"
        ssh_cmd = [
            "gcloud", "compute", "tpus", "tpu-vm", "ssh", self.worker_id,
            f"--zone={self.zone}",
            "--worker=0",
            "--ssh-flag=-o ConnectionAttempts=2",
            "--ssh-flag=-o ConnectTimeout=30",
            "--ssh-flag=-o StrictHostKeyChecking=no",
            "--ssh-flag=-o UserKnownHostsFile=/dev/null",
            "--ssh-flag=-o BatchMode=yes",
            "--command", shlex.quote(inline),
        ]

        with open(log_file, "w") as lf:
            lf.write(f"=== Task {task_id} started at {_now()} ===\n")
            lf.write(f"Script: {script_path}\n")
            lf.write(f"TPU: {self.worker_id} ({self.zone})\n\n")
            lf.flush()

            proc = subprocess.Popen(
                ssh_cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                text=True,
            )
            exit_code = proc.wait()

            lf.write(f"\n=== Task {task_id} ended at {_now()}, exit_code={exit_code} ===\n")

        if exit_code == 255:
            # Possible preemption
            if self._is_preempted():
                logger.warning("Task %s: SSH exit 255 + TPU preempted → interrupted", task_id)
                return False, True

        success = exit_code == 0
        if success:
            logger.info("Task %s completed successfully", task_id)
        else:
            logger.warning("Task %s failed with exit code %d", task_id, exit_code)
        return success, False

    def _is_preempted(self) -> bool:
        status = self.tpu.status()
        return status in ("PREEMPTED", "TERMINATED", "NOT_FOUND", "SUSPENDED")

    # ------------------------------------------------------------------
    # Worker registry
    # ------------------------------------------------------------------

    def _register(self, status: str = "running") -> None:
        registry_path = os.path.join(self._state_dir, "workers.json")
        os.makedirs(self._state_dir, exist_ok=True)

        # Read existing
        registry: dict = {}
        if os.path.exists(registry_path):
            try:
                with open(registry_path) as f:
                    registry = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        registry[self.worker_id] = {
            "worker_id": self.worker_id,
            "tpu_name": self.worker_id,
            "accelerator": self.accelerator,
            "zone": self.zone,
            "tpu_version": self.tpu.version,
            "pricing": self.tpu.pricing,
            "mode": self.tpu.mode,
            "status": status,
            "registered": _now(),
            "pid": os.getpid(),
        }

        tmp = registry_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(registry, f, indent=2)
        os.replace(tmp, registry_path)