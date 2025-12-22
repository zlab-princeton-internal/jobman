import os
import subprocess, concurrent.futures, shlex, time, random
from pathlib import Path
from textwrap import dedent
from jobman.utils import setup_logger

# SSH retry configuration for setup phase (TPU may take time to become SSH-ready)
SSH_MAX_RETRIES = 10
SSH_BASE_DELAY = 5.0    # seconds
SSH_MAX_DELAY = 600.0   # 10 minute max delay between retries

# SSH retry configuration for task execution (fail faster on preemption)
SSH_TASK_RETRIES = 3    # ~30 seconds total with base delay

# Timeout for individual SSH/SCP commands (separate from connection timeout)
# This catches commands that connect but then hang
SSH_CMD_TIMEOUT = 300   # 5 minutes for setup commands
SCP_CMD_TIMEOUT = 300   # 5 minutes for SCP operations

class MultiWorkerRunner:
    
    def __init__(self, cfg, logger, action: str):
        self.cfg = cfg
        self.logger = logger
        self.action = action
        self.workers = list(range(self.cfg.tpu.num_workers))

    def _per_worker_log(self, i): 
        return Path(self.cfg.job.dir)/"logs"/f"{self.action}_worker_{i}.log"
    
    def _get_check_steps(self, i): 
        yield -1

    def _get_setup_steps(self, i): 
        yield 0

    def _ssh(self, i:int, script:str, max_retries:int=None, timeout:int=None) -> int:
        """Execute a script on a worker via SSH.

        Args:
            i: Worker index
            script: Script to execute
            max_retries: Max retry attempts. None=use SSH_MAX_RETRIES (for setup),
                        1=no retries (for task execution where we want fast failure)
            timeout: Command timeout in seconds. None=use SSH_CMD_TIMEOUT (for setup),
                    0=no timeout (for long-running tasks)
        """
        if max_retries is None:
            max_retries = SSH_MAX_RETRIES
        if timeout is None:
            timeout = SSH_CMD_TIMEOUT
        elif timeout == 0:
            timeout = None  # No timeout

        logf = self._per_worker_log(i)

        cmd = [
            "gcloud", "alpha", "compute", "tpus", "tpu-vm", "ssh",
            f"{self.cfg.job.remote_user}@{self.cfg.tpu.name}",
            "--zone", self.cfg.tpu.zone,
            f"--worker={i}",
            "--ssh-flag=-o ConnectTimeout=30",
            "--ssh-flag=-o StrictHostKeyChecking=no",
            "--ssh-flag=-o UserKnownHostsFile=/dev/null",
            "--command", f"bash -lc {shlex.quote(script)}",
            "--quiet",
            "--no-user-output-enabled",
        ]

        private_key = getattr(self.cfg, "ssh", {}).get("private_key", None)
        if private_key:
            key_path = str(Path(private_key).expanduser())
            # Pass "-i" and the identity path as separate ssh flags so gcloud
            # forwards them as separate argv tokens to ssh/scp.
            cmd.extend(["--ssh-flag=-i", f"--ssh-flag={key_path}"])
        if os.environ.get("JOBMAN_DEBUG", "").lower() in ("1", "true", "yes", "on"):
            self.logger.debug(' '.join(cmd))

        # Retry loop with exponential backoff
        for attempt in range(max_retries):
            try:
                with open(logf, "a") as f:
                    rc = subprocess.run(cmd, stdout=f, stderr=f, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                timeout_secs = timeout if timeout else SSH_CMD_TIMEOUT
                self.logger.warning(f"{self.action} worker {i}: SSH command timed out after {timeout_secs}s (attempt {attempt + 1}/{max_retries})")
                rc = 255  # Treat timeout as connection failure

            if rc == 0:
                return 0

            # Check if this is a connection-related failure (exit code 255 for SSH)
            # For other errors (like command failed), don't retry
            if rc != 255 and attempt == 0:
                # Non-SSH error on first attempt, likely a real command failure
                return rc

            if attempt < max_retries - 1:
                # Mark that retries were attempted (for preemption detection)
                self._retry_attempted = True
                # Exponential backoff with jitter
                delay = min(SSH_BASE_DELAY * (2 ** attempt) + random.uniform(0, 2), SSH_MAX_DELAY)
                self.logger.warning(f"{self.action} worker {i}: SSH failed (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s...")
                time.sleep(delay)

        self.logger.error(f"{self.action} worker {i}: SSH failed after {max_retries} attempts")
        self._retry_attempted = True  # Mark retries were attempted
        return rc
        
    def _scp(self, i: int, local_path: str, remote_path: str, recursive: bool = False) -> int:
        logf = self._per_worker_log(i)
        cmd = [
            "gcloud", "alpha", "compute", "tpus", "tpu-vm", "scp",
            local_path,
            f"{self.cfg.job.remote_user}@{self.cfg.tpu.name}:{remote_path}",
            "--zone", self.cfg.tpu.zone,
            f"--worker={i}",
            "--quiet",
        ]
        private_key = getattr(self.cfg, "ssh", {}).get("private_key", None)
        if private_key:
            key_path = str(Path(private_key).expanduser())
            cmd.extend(["--scp-flag=-i", f"--scp-flag={key_path}"])

        if os.environ.get("JOBMAN_DEBUG", "").lower() in ("1", "true", "yes", "on"):
            self.logger.debug(' '.join(cmd))
        if recursive:
            cmd.insert(6, "--recurse")

        # Retry loop with exponential backoff
        for attempt in range(SSH_MAX_RETRIES):
            try:
                with open(logf, "a") as f:
                    rc = subprocess.run(cmd, stdout=f, stderr=f, timeout=SCP_CMD_TIMEOUT).returncode
            except subprocess.TimeoutExpired:
                self.logger.warning(f"{self.action} worker {i}: SCP command timed out after {SCP_CMD_TIMEOUT}s (attempt {attempt + 1}/{SSH_MAX_RETRIES})")
                rc = 255  # Treat timeout as connection failure

            if rc == 0:
                return 0

            if attempt < SSH_MAX_RETRIES - 1:
                # Exponential backoff with jitter
                delay = min(SSH_BASE_DELAY * (2 ** attempt) + random.uniform(0, 2), SSH_MAX_DELAY)
                self.logger.warning(f"{self.action} worker {i}: SCP failed (attempt {attempt + 1}/{SSH_MAX_RETRIES}), retrying in {delay:.1f}s...")
                time.sleep(delay)

        self.logger.error(f"{self.action} worker {i}: SCP failed after {SSH_MAX_RETRIES} attempts")
        return rc

    def _check_worker(self, i: int) -> bool:
        self.logger.info(f"{self.action} worker {i}: checking...")
        return all(rc == 0 for rc in self._get_check_steps(i))

    def _setup_worker(self, i: int) -> bool:
        self.logger.info(f"{self.action} worker {i}: running setup...")
        ok = all(rc == 0 for rc in self._get_setup_steps(i))
        if not ok:
            self.logger.error(f"{self.action} worker {i}: setup failed (some step non-zero).")
        return ok

    def _setup_one(self, i: int, force: bool = False) -> bool:
        try:
            if not force and self._check_worker(i):
                self.logger.info(f"{self.action} worker {i}: already set up, skip.")
                return True
            if force:
                self.logger.info(f"{self.action} worker {i}: force flag set, running setup...")
            return self._setup_worker(i)
        except Exception as e:
            self.logger.exception(f"{self.action} worker {i}: exception: {e}")
            return False

    def setup(self, force: bool = False, timeout: int = None) -> bool:
        """Run setup on all workers in parallel.

        Args:
            force: If True, skip checks and force re-run setup.
            timeout: Max seconds to wait for all workers. None means no timeout.
        """
        self.logger.info(f"setting up {self.action} on {len(self.workers)} workers in parallel..." + (" (force)" if force else ""))
        all_ok = True
        timed_out_workers = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.workers)) as ex:
            futures = {ex.submit(self._setup_one, i, force): i for i in self.workers}
            try:
                for fut in concurrent.futures.as_completed(futures, timeout=timeout):
                    i = futures[fut]
                    try:
                        all_ok &= fut.result()
                    except Exception as e:
                        self.logger.exception(f"{self.action} worker {i}: future exception: {e}")
                        all_ok = False
            except concurrent.futures.TimeoutError:
                # Find which workers haven't completed
                for fut, i in futures.items():
                    if not fut.done():
                        timed_out_workers.append(i)
                        fut.cancel()
                self.logger.error(f"{self.action} TIMEOUT after {timeout}s - workers stuck: {timed_out_workers}")
                all_ok = False

        if all_ok:
            self.logger.info(f"{self.action} setup completed successfully on all workers.")
        elif timed_out_workers:
            self.logger.error(f"{self.action} failed - {len(timed_out_workers)} workers timed out: {timed_out_workers}")
        else:
            self.logger.warning(f"{self.action} finished with failures on some workers.")
        return all_ok