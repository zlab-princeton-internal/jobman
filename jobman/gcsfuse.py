import subprocess
import concurrent.futures
from pathlib import Path
from textwrap import dedent
from jobman.utils import setup_logger

class GCSFUSE:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bucket = cfg.gcsfuse.bucket_name
        self.mount_path = cfg.gcsfuse.mount_path

        self.logger = setup_logger(log_file=Path(cfg.job.dir) / "logs" / "job.log")
        
    def setup(self):
        self.logger.info(f"Setting up GCSFuse and mounting bucket to TPU workers...")

        if not self.bucket or not self.mount_path:
            self.logger.error("GCSFuse config missing `bucket_name` or `mount_path`.")
            return False
        
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
            self.logger.info(f"Worker {i}: GCSFuse already set up and bucket mounted.")
            return
        
        self.logger.info(f"Worker {i}: Setting up GCSFuse...")
        log_file = Path(self.cfg.job.dir) / "logs" / f"gcsfuse_worker_{i}.log"

        gcsfuse_script = dedent(f"""
            set -e
            GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
            echo '[INFO] Adding gcsfuse repo...'
            echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt ${{GCSFUSE_REPO}} main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list

            echo '[INFO] Downloading GPG key...'
            sudo curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc >/dev/null

            if sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; then
                LOCK_PID=$(sudo lsof -t /var/lib/dpkg/lock-frontend || true)
                if [ -n "$LOCK_PID" ]; then
                    echo "[WARN] Killing process $LOCK_PID holding dpkg lock"
                    sudo kill -9 $LOCK_PID
                    sleep 2
                fi
            fi
            # This is needed because sometimes the lock is occupied and installation fails

            echo '[INFO] Updating packages and installing gcsfuse...'
            sudo apt-get update -y && sudo apt-get install -y gcsfuse

            if ! command -v gcsfuse &> /dev/null; then
                echo '[ERROR] gcsfuse install failed!'
                exit 1
            fi

            echo '[INFO] Creating mount path...'
            sudo mkdir -p {self.mount_path}

            echo '[INFO] Mounting bucket...'
            mountpoint -q {self.mount_path} || sudo gcsfuse --implicit-dirs --dir-mode=777 --file-mode=777 --o allow_other {self.bucket} {self.mount_path}

            echo '[INFO] Listing contents...'
            sudo ls -la {self.mount_path}
        """)

        ssh_cmd = [
            "gcloud", "compute", "tpus", "tpu-vm", "ssh", self.cfg.tpu.name,
            f"--worker={i}",
            f"--zone={self.cfg.tpu.zone}",
            f"--ssh-key-file={self.cfg.ssh.private_key}",
            "--ssh-flag=-o ConnectTimeout=15",
            "--ssh-flag=-o StrictHostKeyChecking=no",
            "--ssh-flag=-o UserKnownHostsFile=/dev/null",
            "--command", gcsfuse_script,
            "--quiet",
        ]

        with open(log_file, "w") as f:
            try:
                subprocess.run(ssh_cmd, check=True, stdout=f, stderr=f)
                self.logger.info(f"Worker {i}: GCSFuse setup complete.")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Worker {i}: GCSFuse setup failed: {e}")

    def _check_worker(self, i):
        self.logger.info(f"Worker {i}: Checking GCSFuse...")
        log_file = Path(self.cfg.job.dir) / "logs" / f"gcsfuse_worker_{i}.log"
        cmd = f"which gcsfuse && mount | grep {self.mount_path} && test -n \"$(sudo ls -A {self.mount_path} 2>/dev/null)\""
        with open(log_file, "w") as f:
            try:
                check_cmd = [
                    "gcloud", "alpha", "compute", "tpus", "tpu-vm", "ssh", self.cfg.tpu.name,
                    "--zone", self.cfg.tpu.zone,
                    f"--worker={i}",
                    "--command", cmd,
                    f"--ssh-key-file={self.cfg.ssh.private_key}",
                    "--ssh-flag=-o ConnectTimeout=15",
                    "--ssh-flag=-o StrictHostKeyChecking=no",
                    "--ssh-flag=-o UserKnownHostsFile=/dev/null",
                    "--quiet",
                ]
                if subprocess.run(check_cmd, check=True, stdout=f, stderr=f).returncode == 0:
                    return True
                else:   
                    return False
            except subprocess.CalledProcessError as e:
                # self.logger.error(f"Worker {i}: Error checking GCSFuse: {e}")
                return False
    
