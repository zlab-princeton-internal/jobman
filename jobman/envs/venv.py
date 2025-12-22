import re
from pathlib import Path
from jobman.runner import MultiWorkerRunner

install_cmd = """
# best-effort unlock dpkg to avoid stuck installs
if sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; then
    LOCK_PID=$(sudo lsof -t /var/lib/dpkg/lock-frontend || true)
    if [ -n "$LOCK_PID" ]; then
        echo "[WARN] Killing process $LOCK_PID holding dpkg lock"
        sudo kill -9 $LOCK_PID || true
        sleep 2
    fi
fi

sudo apt install {python}-venv -y
{python} -m venv {remote_venv_dir} || true && \
source {remote_venv_dir}/bin/activate && \
pip install --upgrade pip && \
pip install -r {remote_req_file}
"""

check_cmd = """
if [ -f {remote_venv_dir}/bin/activate ]; then
    exit 0
else
    exit 1
fi
"""

class VENV(MultiWorkerRunner):
    
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger, action='venv')
        
        self.venv_name = cfg.venv.name
        self.req_file = cfg.venv.requirements_file
        self.remote_req_file = f"~/{Path(self.req_file).name}"
        self.remote_venv_dir = f"~/{self.venv_name}"
        self.python = cfg.venv.get('python', 'python3.10')
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
        
    def _get_check_steps(self, i):
        yield self._ssh(i, check_cmd.format(
            remote_venv_dir=self.remote_venv_dir
        ))
        
    def _get_setup_steps(self, i):
        
        yield self._scp(i, str(self.req_file), self.remote_req_file)
        yield self._ssh(i, install_cmd.format(
            python=self.python,
            remote_venv_dir=self.remote_venv_dir, 
            remote_req_file=self.remote_req_file
        ))

    def patch_command(self, cmd):
        return f"source {self.remote_venv_dir}/bin/activate && {cmd}"
    