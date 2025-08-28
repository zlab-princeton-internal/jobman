import os
import subprocess
import concurrent.futures
from pathlib import Path
from textwrap import dedent

from jobman.utils import setup_logger

class SSH:
    
    def __init__(self, cfg):
        
        self.cfg = cfg
        self.private_key = Path(self.cfg.ssh.private_key).expanduser()
        self.identities = self.cfg.ssh.get("identities", None)
        
        self.logger = setup_logger(log_file=Path(cfg.job.dir) / 'logs' / 'job.log')
        
    def setup(self):
        if self.identities is None:
            self.logger.info(f"No SSH identities specified. Skipping...")
            return True
        self.logger.info(f"Copying SSH keys to TPU workers...")

        any_failed = False
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.cfg.tpu.num_workers) as executor:
            futures = [executor.submit(self._setup_worker, i) for i in range(self.cfg.tpu.num_workers)]
            for future in concurrent.futures.as_completed(futures):
                if exc := future.exception():
                    self.logger.error(f"Worker SSH thread failed: {exc}")
                    any_failed = True

        if any_failed:
            self.logger.warning("SSH setup completed with at least one worker failed.")
        else:
            self.logger.info("SSH setup completed successfully on all workers.")
        return not any_failed
        
    def _setup_worker(self, i):
        self.logger.info(f"Worker {i}: Setting up SSH")
        log_file = Path(self.cfg.job.dir) / "logs" / f"ssh_worker_{i}.log"
        ssh_setup_cmds = [
            "mkdir -p ~/.ssh",
            "chmod 700 ~/.ssh"
        ]
        combined_config = ""
        
        with open(log_file, "w") as f:
            for entry in self.identities:
                priv = Path(entry.private_key).expanduser()
                pub = Path(entry.public_key).expanduser()
                config_entry = dedent(entry.config_entry).strip()

                if not priv.exists() or not pub.exists():
                    self.logger.error(f"SSH key not found: {priv} or {pub}")
                    continue

                self._copy_key_to_worker(i, priv, f)
                self._copy_key_to_worker(i, pub, f)

                ssh_setup_cmds += [
                    f"chmod 600 ~/.ssh/{priv.name}",
                    f"chmod 644 ~/.ssh/{pub.name}",
                ]
                
                combined_config += config_entry + "\n\n"
            
            if combined_config:
                escaped_config = combined_config.strip().replace('"', '\\"').replace('\n', '\\n')
                ssh_setup_cmds += [
                    f'sudo printf "{escaped_config}\\n" > ~/.ssh/config',
                    'chmod 600 ~/.ssh/config'
                ]

            cmd = " && ".join(ssh_setup_cmds)
            self.logger.debug(f"Worker {i}: Running SSH config cmd: {cmd}")
            self._configure_remote_ssh(i, cmd, f)
        
        return True
        
    def _copy_key_to_worker(self, i, key_file, f):
        target_path = f"{self.cfg.tpu.name}:~/.ssh/{key_file.name}"
        scp_cmd = [
            "gcloud", "compute", "tpus", "tpu-vm", "scp", str(key_file), target_path,
            "--worker", str(i), "--zone", self.cfg.tpu.zone,
            "--ssh-key-file", str(self.private_key),
            "--scp-flag=-o ConnectTimeout=15",
            "--scp-flag=-o StrictHostKeyChecking=no",
            "--scp-flag=-o UserKnownHostsFile=/dev/null",
            "--quiet",
        ]
        self.logger.debug("Using scp command:")
        self.logger.debug(" ".join(scp_cmd))
        try:
            subprocess.run(scp_cmd, check=True, stdout=f, stderr=f)
            self.logger.debug(f"Worker {i}: Copied {key_file.name}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Worker {i}: Failed to copy {key_file.name}: {e}")
            
    def _configure_remote_ssh(self, i, cmd, f):
        ssh_cmd = [
            "gcloud", "alpha", "compute", "tpus", "tpu-vm", "ssh", self.cfg.tpu.name,
            "--worker", str(i), "--zone", self.cfg.tpu.zone,
            "--command", cmd,
            "--ssh-key-file", str(self.private_key),
            "--ssh-flag=-o ConnectTimeout=15",
            "--ssh-flag=-o StrictHostKeyChecking=no",
            "--ssh-flag=-o UserKnownHostsFile=/dev/null",
            "--quiet",
        ]
        try:
            subprocess.run(ssh_cmd, check=True, stdout=f, stderr=f)
            self.logger.debug(f"Worker {i}: Remote SSH configured")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Worker {i}: Remote SSH config failed: {e}")
            
    def _check_worker(self, i):
        raise NotImplementedError


