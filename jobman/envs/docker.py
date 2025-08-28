import subprocess
import concurrent.futures
from pathlib import Path

from jobman.envs.base import ENV
from jobman.utils import setup_logger

class DOCKER(ENV):
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.image = cfg.docker.image
        self.env_vars = cfg.docker.get('env_vars', [])
        self.mount_dirs = cfg.docker.get('mount_dirs', [])
        self.workdir = cfg.docker.get('work_dir', None)
        self.flags = cfg.docker.get('flags', None)
        
        self.logger = setup_logger(log_file=Path(cfg.job.dir) / "logs" / "job.log")
        
    def setup(self):
        self.logger.info(f"Setting up Docker on TPU workers...")
        
        any_failed = False
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.cfg.tpu.num_workers) as executor:
            futures = [executor.submit(self._setup_worker, i) for i in range(self.cfg.tpu.num_workers)]
            for future in concurrent.futures.as_completed(futures):
                if exc := future.exception():
                    self.logger.error(f"Worker thread failed: {exc}")
                    any_failed = True

        if any_failed:
            self.logger.warning("GCSFuse setup completed with at least one worker failed.")
        else:
            self.logger.info("GCSFuse setup completed successfully on all workers.")
        return not any_failed
    
    def _setup_worker(self, i):
        if self._check_worker(i):
            self.logger.info(f"Worker {i}: Docker image {self.image} already exists.")
            return
        
        self.logger.info(f"Worker {i}: Setting up Docker...")
        log_file = Path(self.cfg.job.dir) / "logs"  / f"docker_worker_{i}.log"

        with open(log_file, "w") as f:
            try:
                cmd1 = [
                    "gcloud", "alpha", "compute", "tpus", "tpu-vm", "ssh", self.cfg.tpu.name,
                    "--zone", self.cfg.tpu.zone,
                    f"--worker={i}",
                    "--command", "sudo usermod -aG docker $USER && sudo systemctl restart docker",
                    f"--ssh-key-file={self.cfg.ssh.private_key}",
                    "--ssh-flag=-o ConnectTimeout=15",
                    "--ssh-flag=-o StrictHostKeyChecking=no",
                    "--ssh-flag=-o UserKnownHostsFile=/dev/null",
                    "--quiet",
                ]
                subprocess.run(cmd1, check=True, stdout=f, stderr=f)

                cmd2 = [
                    "gcloud", "alpha", "compute", "tpus", "tpu-vm", "ssh", self.cfg.tpu.name,
                    "--zone", self.cfg.tpu.zone,
                    f"--worker={i}",
                    "--command", f"docker pull {self.image}",
                    f"--ssh-key-file={self.cfg.ssh.private_key}",
                    "--ssh-flag=-o ConnectTimeout=15",
                    "--ssh-flag=-o StrictHostKeyChecking=no",
                    "--ssh-flag=-o UserKnownHostsFile=/dev/null",
                    "--quiet",
                ]
                subprocess.run(cmd2, check=True, stdout=f, stderr=f)
            except Exception as e:
                self.logger.error(f"Worker {i} setup failed: {e}")
                raise        
            
    def _check_worker(self, i):
        self.logger.info(f"Worker {i}: Checking Docker image...")
        log_file = Path(self.cfg.job.dir) / "logs" / f"docker_worker_{i}.log"
        
        with open(log_file, "w") as f:
            try:
                check_cmd = [
                    "gcloud", "alpha", "compute", "tpus", "tpu-vm", "ssh", self.cfg.tpu.name,
                    "--zone", self.cfg.tpu.zone,
                    f"--worker={i}",
                    "--command", f"docker image inspect {self.image}",
                    f"--ssh-key-file={self.cfg.ssh.private_key}",
                    "--ssh-flag=-o ConnectTimeout=15",
                    "--ssh-flag=-o StrictHostKeyChecking=no",
                    "--ssh-flag=-o UserKnownHostsFile=/dev/null",
                    "--quiet",
                ]
                if subprocess.run(check_cmd, check=True, stdout=f, stderr=f).returncode == 0:
                    return True
                else:   
                    self.logger.warning(f"Worker {i}: Docker image {self.image} not found")
                    return False
            except subprocess.CalledProcessError as e:
                # self.logger.error(f"Worker {i}: Error checking Docker image: {e}")
                return False
        
    def patch_command(self, cmd):

        var_flags = []
        volume_flags = []
        for e in self.env_vars:
            assert "=" in e, "expecting format <var_name>=<var_value>"
            var_flags.append(f"-e {e}")
        var_flags_str = " ".join(var_flags)
        
        for d in self.mount_dirs:
            d = str(Path(d).expanduser())
            if ":" in d:
                host_path, container_path = d.split(":", 1)
            else:
                host_path = container_path = d
            volume_flags.append(f"-v {host_path}:{container_path}")
        
        volume_flags_str = " ".join(volume_flags)
        workdir_flag = f"-w {self.workdir}" if self.workdir else ""
        flags_str = " ".join(self.flags or [])

        docker_cmd = f"sudo docker run {flags_str} {var_flags_str} {volume_flags_str} {workdir_flag} {self.image} bash -c \"{cmd}\""

        return docker_cmd
        
