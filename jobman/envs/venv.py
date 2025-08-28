import re
import subprocess
import concurrent.futures
from pathlib import Path

from jobman.envs.base import ENV
from jobman.utils import setup_logger

class VENV(ENV):
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.env_name = cfg.venv.name
        self.requirements_file = cfg.venv.requirements_file
        self.python = cfg.venv.get('python', 'python3.10')
        
        self.logger = setup_logger(log_file=Path(cfg.job.dir) / "logs" / "job.log")
        self.normalize_python()

    def normalize_python(self):
        val = str(self.python).strip()

        if re.fullmatch(r"3\.\d+", val):
            normalized = f"python{val}"
        elif re.fullmatch(r"python3\.\d+", val):
            normalized = val
        else:
            raise ValueError(
                f"Invalid python version string '{val}'. "
                "Must be like '3.10' or 'python3.10'."
            )

        self.logger.debug(f"Normalized python executable: {normalized}")
        self.python = normalized

    def setup(self):
        self.logger.info(f"Setting up Venv environment on TPU workers...")

        any_failed = False
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.cfg.tpu.num_workers) as executor:
            futures = [executor.submit(self.setup_worker, i) for i in range(self.cfg.tpu.num_workers)]
            for future in concurrent.futures.as_completed(futures):
                if exc := future.exception():
                    self.logger.error(f"Worker thread failed: {exc}")
                    any_failed = True

        if any_failed:
            self.logger.warning("Venv setup completed with at least one worker failed.")
        else:
            self.logger.info("Venv setup completed successfully on all workers.")
        return not any_failed
    
    def setup_worker(self, i):
        if self._check_worker(i):
            self.logger.info(f"Worker {i}: VENV already set up.")
            return
        
        self.logger.info(f"Worker {i}: Setting up VENV...")
        log_file = Path(self.cfg.job.dir) / "logs" / f"venv_worker_{i}.log"
        remote_venv_dir = f"~/{self.env_name}"
        remote_req_file = f"~/requirements_{self.env_name}.txt"
        local_req_file = self.requirements_file

        with open(log_file, "w") as f:
            try:
                # Step 1: Copy requirements.txt to remote
                scp_cmd = [
                    "gcloud", "alpha", "compute", "tpus", "tpu-vm", "scp",
                    str(local_req_file),
                    f"{self.cfg.tpu.name}:{remote_req_file}",
                    "--zone", self.cfg.tpu.zone,
                    f"--worker={i}",
                    "--quiet",
                ]
                subprocess.run(scp_cmd, check=True, stdout=f, stderr=f)

                # Step 2: Create virtualenv and install requirements
                remote_cmd = f"""
                    sudo apt install {self.python}-venv -y
                    {self.python} -m venv {remote_venv_dir} || true && \
                    source {remote_venv_dir}/bin/activate && \
                    pip install --upgrade pip && \
                    pip install -r {remote_req_file}
                """
                ssh_cmd = [
                    "gcloud", "alpha", "compute", "tpus", "tpu-vm", "ssh", self.cfg.tpu.name,
                    "--zone", self.cfg.tpu.zone,
                    f"--worker={i}",
                    "--command", remote_cmd,
                    "--quiet",
                ]
                subprocess.run(ssh_cmd, check=True, stdout=f, stderr=f)

            except Exception as e:
                self.logger.error(f"Worker {i} venv setup failed: {e}")
                raise
      
    def check(self):
        pass      
        
    def _check_worker(self, i: int) -> bool:
        """Check if the VENV is already set up on worker i."""
        self.logger.info(f"Worker {i}: Checking VENV setup...")

        remote_cmd = f"""
            if [ -f ~/{self.env_name}/bin/activate ]; then
                echo "VENV_EXISTS"
            else
                echo "NO_VENV"
            fi
        """

        ssh_cmd = [
            "gcloud", "alpha", "compute", "tpus", "tpu-vm", "ssh", self.cfg.tpu.name,
            "--zone", self.cfg.tpu.zone,
            f"--worker={i}",
            "--ssh-flag=-o ConnectTimeout=10",
            "--ssh-flag=-o StrictHostKeyChecking=no",
            "--ssh-flag=-o UserKnownHostsFile=/dev/null",
            "--command", f"bash -lc \"{remote_cmd}\"",
            "--quiet",
        ]

        try:
            result = subprocess.run(
                ssh_cmd, capture_output=True, text=True, timeout=30
            )
            if "VENV_EXISTS" in result.stdout:
                self.logger.info(f"Worker {i}: VENV already present.")
                return True
            else:
                return False
        except Exception as e:
            self.logger.warning(f"Worker {i}: Failed to check VENV: {e}")
            return False

    def patch_command(self, cmd):
        return f'source ~/{self.env_name}/bin/activate && {cmd}'
    