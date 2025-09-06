import os
import subprocess, concurrent.futures, shlex
from pathlib import Path
from textwrap import dedent
from jobman.utils import setup_logger

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

    def _ssh(self, i:int, script:str) -> int:
        logf = self._per_worker_log(i)
        
        cmd = [
            "gcloud", "alpha", "compute", "tpus", "tpu-vm", "ssh", 
            f"{self.cfg.job.remote_user}@{self.cfg.tpu.name}",
            "--zone", self.cfg.tpu.zone, 
            f"--worker={i}",
            "--ssh-flag=-o ConnectTimeout=15",
            "--ssh-flag=-o StrictHostKeyChecking=no",
            "--ssh-flag=-o UserKnownHostsFile=/dev/null",
            "--command", f"bash -lc {shlex.quote(script)}",
            "--quiet",
            "--no-user-output-enabled",
        ]
        
        private_key = getattr(self.cfg, "ssh", {}).get("private_key", None)
        if private_key:
            cmd.append(f"--ssh-flag=-i {private_key}")
        if os.environ.get("JOBMAN_DEBUG", "").lower() in ("1", "true", "yes", "on"):
            self.logger.debug(' '.join(cmd))
        with open(logf, "a") as f:
            return subprocess.run(cmd, stdout=f, stderr=f).returncode
        
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
            cmd.append(f"--scp-flag=-i {private_key}")
        
        if os.environ.get("JOBMAN_DEBUG", "").lower() in ("1", "true", "yes", "on"):
            self.logger.debug(' '.join(cmd))
        if recursive:
            cmd.insert(6, "--recurse")
        with open(logf, "a") as f:
            return subprocess.run(cmd, stdout=f, stderr=f).returncode

    def _check_worker(self, i: int) -> bool:
        self.logger.info(f"{self.action} worker {i}: checking...")
        return all(rc == 0 for rc in self._get_check_steps(i))

    def _setup_worker(self, i: int) -> bool:
        self.logger.info(f"{self.action} worker {i}: running setup...")
        ok = all(rc == 0 for rc in self._get_setup_steps(i))
        if not ok:
            self.logger.error(f"{self.action} worker {i}: setup failed (some step non-zero).")
        return ok

    def _setup_one(self, i: int) -> bool:
        try:
            if self._check_worker(i):
                self.logger.info(f"{self.action} worker {i}: already set up, skip.")
                return True
            return self._setup_worker(i)
        except Exception as e:
            self.logger.exception(f"{self.action} worker {i}: exception: {e}")
            return False

    def setup(self) -> bool:
        self.logger.info(f"setting up {self.action} on {len(self.workers)} workers in parallel...")
        all_ok = True
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.workers)) as ex:
            futures = {ex.submit(self._setup_one, i): i for i in self.workers}
            for fut in concurrent.futures.as_completed(futures):
                i = futures[fut]
                try:
                    all_ok &= fut.result()
                except Exception as e:
                    self.logger.exception(f"{self.action} worker {i}: future exception: {e}")
                    all_ok = False
        if all_ok:
            self.logger.info(f"{self.action} setup completed successfully on all workers.")
        else:
            self.logger.warning(f"{self.action} finished with failures on some workers.")
        return all_ok