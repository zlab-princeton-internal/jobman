import argparse
import subprocess
import concurrent.futures
from pathlib import Path
from omegaconf import OmegaConf

from jobman.envs.base import ENV
from jobman.utils import setup_logger

class CONDA(ENV):
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.env_name = cfg.conda.name
        self.config_file = Path(cfg.conda.config_file)

        self.logger = setup_logger(log_file=Path(cfg.job.dir) / "logs" / "job.log")

    def setup(self):
        self.logger.info(f"Setting up Conda environment on TPU workers...")

        any_failed = False
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.cfg.tpu.num_workers) as executor:
            futures = [executor.submit(self.setup_worker, i) for i in range(self.cfg.tpu.num_workers)]
            for future in concurrent.futures.as_completed(futures):
                if exc := future.exception():
                    self.logger.error(f"Worker thread failed: {exc}")
                    any_failed = True

        if any_failed:
            self.logger.warning("Conda setup completed with at least one worker failed.")
        else:
            self.logger.info("Conda setup completed successfully on all workers.")
        return not any_failed

    def setup_worker(self, i):
        # if self._check_worker(i):
        #     self.logger.info(f"Worker {i}: Conda already set up.")
        #     return
        
        self.logger.info(f"Worker {i}: Setting up Conda...")
        log_file = Path(self.cfg.job.dir) / "logs" / f"conda_worker_{i}.log"

        remote_env_file = f"~/{self.config_file.name}"

        with open(log_file, "w") as f:
            try:
                # Step 1: scp env file to remote worker
                scp_cmd = [
                    "gcloud", "alpha", "compute", "tpus", "tpu-vm", "scp",
                    str(self.config_file),  # local path
                    f"{self.cfg.tpu.name}:{remote_env_file}",  # remote path
                    "--zone", self.cfg.tpu.zone,
                    f"--worker={i}",
                    "--quiet",
                ]
                subprocess.run(scp_cmd, check=True, stdout=f, stderr=f)

                # Step 2: install Miniconda + create env
                remote_cmd = f"""        
                    if [ ! -d ~/miniconda ]; then
                        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
                        bash miniconda.sh -b -p ~/miniconda
                    fi && \
                    source ~/miniconda/etc/profile.d/conda.sh
                    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 
                    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
                    conda env create -n {self.env_name} -f {remote_env_file} --yes
                """

                ssh_cmd = [
                    "gcloud", "alpha", "compute", "tpus", "tpu-vm", "ssh", self.cfg.tpu.name,
                    "--zone", self.cfg.tpu.zone,
                    f"--worker={i}",
                    "--command", remote_cmd,
                    "--ssh-flag=-o ConnectTimeout=15",
                    "--ssh-flag=-o StrictHostKeyChecking=no",
                    "--ssh-flag=-o UserKnownHostsFile=/dev/null",
                    "--quiet",
                ]
                subprocess.run(ssh_cmd, check=True, stdout=f, stderr=f)

            except Exception as e:
                self.logger.error(f"Worker {i} Conda setup failed: {e}")
                raise
    
    def _check_worker(self, i):
        self.logger.info(f"Worker {i}: Checking Conda setup...")

        remote_cmd = f"""
            if [ -x ~/miniconda/bin/conda ]; then
                source ~/miniconda/etc/profile.d/conda.sh
                conda env list | grep -q '^{self.env_name}\\s'
            fi
        """
        try:
            subprocess.run(
                [
                    "gcloud", "alpha", "compute", "tpus", "tpu-vm", "ssh", self.cfg.tpu.name,
                    "--zone", self.cfg.tpu.zone,
                    f"--worker={i}",
                    "--command", remote_cmd,
                    "--quiet",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except subprocess.CalledProcessError:
            return Fals

    def patch_command(self, cmd):
        return f'source ~/miniconda/etc/profile.d/conda.sh && conda run -n {self.env_name} bash -c \'{cmd}\''