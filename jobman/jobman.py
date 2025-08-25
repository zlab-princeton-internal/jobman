import re
import os
import math
import json
import fcntl
import shutil
import logging
import subprocess
from pathlib import Path
from datetime import datetime 
from tabulate import tabulate
from omegaconf import OmegaConf
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed

from jobman.job import Job
from jobman.utils import setup_logger

jobs_dir = Path("jobs") 
jobman_dir = jobs_dir / ".jobman"

def infer_num_workers(accelerator: str) -> int:
    """
    Infer number of workers based on accelerator type.
    Examples:
        v4-256 -> 32 workers (256 // 8)
        v5e-32 -> 8 workers (32 // 4)
    """
    match = re.search(r"v(\d+)[a-z]*-(\d+)", accelerator.lower())
    if not match:
        raise ValueError(f"Invalid accelerator format: {accelerator}")
    
    version, chips = int(match.group(1)), int(match.group(2))
    if version in [2, 3, 4]:
        return math.ceil(chips / 8)
    elif version in [5, 6]:
        return math.ceil(chips / 4)
    else:
        raise ValueError(f"Unknown TPU version in accelerator: {accelerator}")

def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)

class JobMan:

    def __init__(self):
        jobman_dir.mkdir(parents=True, exist_ok=True)
        self.meta_file = jobman_dir / "meta.json"
        self.lock_file = jobman_dir / "lock"
        self.cntr_file = jobman_dir / "next_job_id.txt"
        self.logger = setup_logger(stdout=True)
        
        if not self.lock_file.exists():
            self.lock_file.touch()
        if not self.meta_file.exists():
            _atomic_write_text(self.meta_file, "{}\n")
        if not self.cntr_file.exists():
            _atomic_write_text(self.cntr_file, "0\n")
        
    @contextmanager
    def with_meta_lock(self):
        with open(self.lock_file, "r+") as lock_fp:
            fcntl.flock(lock_fp, fcntl.LOCK_EX)
            try:
                if self.meta_file.exists():
                    meta = json.loads(self.meta_file.read_text())
                else:
                    meta = {}
                yield meta
                # Write back updated meta
                self.meta_file.write_text(json.dumps(meta, indent=2))
            finally:
                fcntl.flock(lock_fp, fcntl.LOCK_UN)
            
    def create_job(self, config_path):
        job_id = self.get_next_job_id()
        user = os.environ['USER']
        job_dir = Path(f"jobs/{user}/{job_id}")
        job_dir.mkdir(parents=True, exist_ok=True)
        
        with self.with_meta_lock() as meta:
            meta[f"job_{job_id}"] = {
                "job_id": job_id,
                "user": user,
                "job_dir": str(job_dir),
                "created_at": datetime.now().isoformat()
            }
        
        cfg = OmegaConf.load(config_path)
        cfg.job.id = job_id
        cfg.job.user = user
        cfg.job.dir = str(job_dir)
        cfg.tpu.num_workers = infer_num_workers(cfg.tpu.accelerator)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.job.name = f"{cfg.job.name}_{ts}"
        cfg.tpu.name = f"{cfg.tpu.name}_{ts}"
        OmegaConf.save(cfg, job_dir / "config.yaml")
        
        self.logger.info(f"Created job {job_id}. See info at {job_dir}")
        
        return job_id
    
    def get_next_job_id(self):

        with open(self.lock_file, "r+") as lock_fp:
            fcntl.flock(lock_fp, fcntl.LOCK_EX)
            current = int(self.cntr_file.read_text())
            next_id = current + 1
            self.cntr_file.write_text(str(next_id))
            fcntl.flock(lock_fp, fcntl.LOCK_UN)
            return f"{next_id:06d}"
    
    def start_job(self, job_id):
        
        with self.with_meta_lock() as meta:
            meta_data = meta.get(f"job_{job_id}")
        job_dir = Path(meta_data.get("job_dir"))
        session_name = f"job_{job_id}"
        
        logs_dir = job_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "job.log"
        
        config_path = job_dir / "config.yaml"
        run_cmd = f"jobman run {job_id}"

        tmux_cmd = f'tmux new-session -d -s {session_name} "{run_cmd} | tee -a {log_file}"'
        subprocess.run(tmux_cmd, shell=True, check=True)

        self.update_job_meta(
            job_id,
            status="RUNNING",
            backend="tmux",
            session_name=session_name,
            started_at=datetime.now().isoformat(),
        )

        self.logger.info(f"Job {job_id} started. See logs at {logs_dir}/job.log.")
    
    def check_tmux_session(self, session_name: str) -> bool:
        return subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        ).returncode == 0
    
    def cancel_job(self, job_id):
        key = f"job_{job_id}"
        
        with self.with_meta_lock() as meta:
            job_meta = meta.get(key)

        if not job_meta:
            self.logger.warning(f"No metadata found for job {job_id}")
            return False

        session_name = job_meta.get("session_name")
        if not session_name:
            self.logger.error(f"No tmux session_name found for job {job_id}")
            return False
        
        # First check if session exists
        session_exists = self.check_tmux_session(session_name)

        if not session_exists:
            self.logger.warning(f"Session '{session_name}' does not exist. Nothing to cancel.")
            return False

        # Try to kill the tmux session
        try:
            subprocess.run(["tmux", "kill-session", "-t", session_name], check=True)
            self.update_job_meta(
                job_id,
                status="FAILED",
                ended_at=datetime.now().isoformat()
            )
            self.logger.info(f"Cancelled job {job_id} by killing tmux session {session_name}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to kill tmux session {session_name}: {e}")
            return False
            
    def delete_job(self, job_id):
        self.logger.info(f"Deleting job {job_id}...")
    
        try:
            cancelled = self.cancel_job(job_id)
            self.logger.debug(f"cancel_job returned {cancelled}")
        except Exception as e:
            self.logger.warning(f"Failed to cancel job {job_id} before deletion: {e}")

        with self.with_meta_lock() as meta:
            meta_data = meta.get(f"job_{job_id}")
        job_dir = Path(meta_data.get("job_dir"))
        config_path = job_dir / "config.yaml"
        if config_path.exists():
            try:
                cfg = OmegaConf.load(config_path)
                job = Job(cfg)
                job.delete()
            except Exception as e:
                self.logger.exception(f"Failed to delete job {job_id}: {e}")
        else:
            self.logger.error(f"Job {job_id} config not found at {config_path}")
        
        try:
            shutil.rmtree(job_dir)
            self.logger.info(f"Deleted job directory {job_dir}")
        except Exception as e:
            self.logger.error(f"Failed to delete job directory {job_dir}: {e}")

        self.remove_job_meta(job_id)
        self.logger.info(f"Deleted job {job_id} successfully")
        return True
    
    def list_jobs(self):
        rows = []

        with self.with_meta_lock() as meta:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {
                    executor.submit(self.fetch_job_info, metadata): job_key
                    for job_key, metadata in meta.items()
                }

                for future in as_completed(futures):
                    rows.append(future.result())

        rows.sort(key=lambda x: x[0])
        headers = ["Job ID", "User", "Name", "Accelerator", "Zone", "Host0 IP", "Status"]
        print(tabulate(rows, headers=headers, tablefmt="github"))
            
    def fetch_job_info(self, meta):
        try:
            job_id = meta.get("job_id")
            user = meta.get("user")
            
            # started = meta.get("started_at", meta.get("created_at", "N/A"))
            session_name = meta.get("session_name", f"job_{job_id}")

            config_path = Path(meta.get("job_dir")) / "config.yaml"
            if config_path.exists():
                cfg = OmegaConf.load(config_path)
                job_name = cfg.job.name
                accelerator = cfg.tpu.accelerator
                zone = cfg.tpu.zone
                try:
                    host0_ip = next(ip.get("external_ip", "N/A") for ip in cfg.tpu.ips if ip.worker == 0)
                except Exception:
                    host0_ip = "N/A"
            else:
                job_name = accelerator = zone = host0_ip = "N/A"
                cfg = None
            
            try:
                gcloud_state = subprocess.run(
                    [
                        "gcloud", "alpha", "compute", "tpus", "tpu-vm", "describe",
                        cfg.tpu.name, "--zone", cfg.tpu.zone, "--format=value(state)"
                    ],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                gcloud_state = gcloud_state.stdout.strip()
                if self.check_tmux_session(session_name):
                    status = "RUNNING" if gcloud_state in {"READY", "ACTIVE"} else "QUEUEING"
                elif cfg:
                    status = "IDLE" if gcloud_state in {"READY", "ACTIVE"} else "DEAD"
                else:
                    status = "UNKNOWN"
            except:
                status = "UNKNOWN"

            return [job_id, user, job_name, accelerator, zone, host0_ip, status]
        except Exception as e:
            return [job_id, "ERROR", "ERROR", "ERROR", "ERROR", "ERROR", f"ERROR: {e}"]
        
    def get_job_meta(self, job_id):
        with self.with_meta_lock() as meta:
            return meta.get(f"job_{job_id}", None)

    def update_job_meta(self, job_id, **kwargs):
        with self.with_meta_lock() as meta:
            key = f"job_{job_id}"
            if key not in meta:
                meta[key] = {"job_id": job_id}
            meta[key].update(kwargs)
            meta[key]["last_seen"] = datetime.now().isoformat()

    def remove_job_meta(self, job_id):
        with self.with_meta_lock() as meta:
            key = f"job_{job_id}"
            if key in meta:
                del meta[key]