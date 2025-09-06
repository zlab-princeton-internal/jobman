from pathlib import Path
from jobman.runner import MultiWorkerRunner

install_script = """
set -e
echo "[INFO] Adding gcsfuse apt repo ..."
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt gcsfuse-$(lsb_release -c -s) main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list

echo "[INFO] Installing repo key ..."
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc >/dev/null

# best-effort unlock dpkg to avoid stuck installs
if sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; then
    LOCK_PID=$(sudo lsof -t /var/lib/dpkg/lock-frontend || true)
    if [ -n "$LOCK_PID" ]; then
        echo "[WARN] Killing process $LOCK_PID holding dpkg lock"
        sudo kill -9 $LOCK_PID || true
        sleep 2
    fi
fi

echo "[INFO] apt update & install gcsfuse ..."
sudo apt-get update -y && sudo apt-get install -y gcsfuse

if ! command -v gcsfuse >/dev/null 2>&1; then
    echo "[ERROR] gcsfuse not found after install"
    exit 1
fi
"""

mount_script = """
set -e
echo "[INFO] Ensuring mount path exists: {mnt}"
{sudo}mkdir -p {mnt}

echo "[INFO] Mounting bucket if not already mounted ..."
mountpoint -q {mnt} || {sudo}gcsfuse --implicit-dirs --dir-mode=777 --file-mode=777 {allow_other_opt} {bucket} {mnt}

echo "[INFO] Listing mount ..."
{sudo}ls -la {mnt} || true
"""

class GCSFUSE(MultiWorkerRunner):
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger, action='gcsfuse')
        
        self.bucket = cfg.gcsfuse.bucket_name
        self.mount_path = cfg.gcsfuse.mount_path
        # self.sudo = 'sudo ' if cfg.job.env_type == 'docker' else '' 
        self.sudo = ''
        self.allow_other_opt = '-o allow_other' if self.sudo else ''
        
    def _get_check_steps(self, i):
        yield self._ssh(i, "command -v gcsfuse >/dev/null 2>&1")
        yield self._ssh(i, f"mountpoint -q {self.mount_path}")

    def _get_setup_steps(self, i):
        yield self._ssh(i, install_script)
        yield self._ssh(i, mount_script.format(
            sudo=self.sudo,
            mnt=self.mount_path,
            bucket=self.bucket,
            allow_other_opt=self.allow_other_opt
        ))

    