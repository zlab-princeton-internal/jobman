import re
import os
import math
import json
import time
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
from jobman.tpu import TPU
from jobman.utils import setup_logger

BASE_DIR = Path(__file__).resolve().parent 
jobs_dir = (BASE_DIR / ".." / "jobs").resolve()
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
                meta = json.loads(self.meta_file.read_text())
                yield meta
                self.meta_file.write_text(json.dumps(meta, indent=2))
            finally:
                fcntl.flock(lock_fp, fcntl.LOCK_UN)
            
    def validate_cfg(self, cfg):
        if cfg.gcsfuse:
            self.logger.info("checking TPU-bucket region consistency...")
            zone = cfg.tpu.zone
            try:
                region = subprocess.run(
                    ["gcloud", "storage", "buckets", "describe", f"gs://{cfg.gcsfuse.bucket_name}", "--format=value(location)"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
                ).stdout.strip().lower()
            except Exception as e:
                raise Exception(f"Checking bucket gs://{cfg.gcsfuse.bucket_name} region failed: {e}")
                
            if not zone.startswith(region):
                raise ValueError(f"Bucket region {region} does not match TPU VM zone {cfg.tpu.zone}. It's strongly suggested to confirm these configurations match to minimize cost.")
        
        return True
            
    def create_job(self, config_path):
        cfg = OmegaConf.load(config_path)
        self.validate_cfg(cfg)
        
        job_id = self.get_next_job_id()
        user = os.environ['USER']
        job_dir = jobs_dir / user / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        cfg.job.id = job_id
        cfg.job.user = user
        cfg.job.dir = str(job_dir)
        cfg.tpu.num_workers = infer_num_workers(cfg.tpu.accelerator)
        cfg.job.name = f"{cfg.job.name}_{job_id}"
        cfg.tpu.name = f"{cfg.tpu.name}_{job_id}"
        OmegaConf.save(cfg, job_dir / "config.yaml")
        
        with self.with_meta_lock() as meta:
            meta[job_id] = {
                "name": cfg.job.name,
                "user": user,
                "job_dir": str(job_dir),
            }
        
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
    
    def start_job(self, job_id, name=None):
        with self.with_meta_lock() as meta:
            job_meta = meta.get(job_id)
        job_dir = Path(job_meta.get("job_dir"))
        session_name = f"job_{job_id}"
        
        logs_dir = job_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "job.log"
        
        config_path = job_dir / "config.yaml"
        run_cmd = f"jobman run {job_id}"
        if name:
            run_cmd += f" --name {name}"

        subprocess.run(f'tmux new-session -d -s {session_name} "{run_cmd} | tee -a {log_file}"', shell=True, check=True)
        
        with self.with_meta_lock() as meta:
            meta[job_id].update(
                session_name=session_name,
                config_path=str(config_path),
            )

        self.logger.info(f"Job {job_id} started. See logs at {str(log_file)}.")
        self.logger.info(f"Config snapshot saved at {str(config_path)}. Modify the snapshot if you need to resume the job.")
    
    def check_tmux_session(self, session_name: str) -> bool:
        return subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        ).returncode == 0
    
    def stop_job(self, job_id):
        with self.with_meta_lock() as meta:
            if job_id not in meta:
                raise ValueError(f"No metadata found for job {job_id}")
            job_meta = meta[job_id]

        session_name = job_meta.get("session_name", None)
        if not session_name:
            self.logger.error(f"No tmux session found for job {job_id}")
            return False
        
        if not self.check_tmux_session(session_name):
            self.logger.warning(f"Session '{session_name}' does not exist. Nothing to cancel.")
            return False

        try:
            subprocess.run(["tmux", "kill-session", "-t", session_name], check=True)
            self.logger.info(f"Cancelled job {job_id} by killing tmux session {session_name}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to kill tmux session {session_name}: {e}")
            return False
            
    def delete_job(self, job_id):
        self.stop_job(job_id)
            
        with self.with_meta_lock() as meta:
            job_meta = meta[job_id]
        config_path = Path(job_meta.get("config_path"))
        if config_path.exists():
            try:
                cfg = OmegaConf.load(config_path)
                tpu = TPU(cfg, self.logger)
                tpu.delete()
            except Exception as e:
                self.logger.exception(f"Failed to delete job {job_id}: {e}")
        else:
            self.logger.error(f"Job {job_id} config not found at {config_path}")
            return False
       
        return True
    
    def clean_job(self, job_id):
        if not self.delete_job(job_id):
            self.logger(f"Job {job_id} deletion failed")
            return False
        
        with self.with_meta_lock() as meta:
            job_meta = meta[job_id]
        
        job_dir = Path(job_meta.get("job_dir"))
        try:
            shutil.rmtree(job_dir, ignore_errors=True)
            self.logger.info(f"Deleted job directory {job_dir}")
        except Exception as e:
            self.logger.error(f"Failed to delete job directory {job_dir}: {e}")

        with self.with_meta_lock() as meta:
            del meta[job_id]
        self.logger.info(f"Cleaned logs of {job_id} successfully")
        return True
    
    def list_jobs(self):
        rows = []

        with self.with_meta_lock() as meta:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {
                    executor.submit(self.fetch_job_info, job_id, metadata): job_id
                    for job_id, metadata in meta.items()
                }
                for future in as_completed(futures):
                    rows.append(future.result())

        rows.sort(key=lambda x: x[0])
        headers = ["Job ID", "User", "Name", "Accelerator", "Zone", "Host0 IP", "Status"]
        print(tabulate(rows, headers=headers, tablefmt="github"))
            
    def fetch_job_info(self, job_id, job_meta):
        try:
            user = job_meta.get("user")
            session_name = job_meta.get("session_name", f"job_{job_id}")
            config_path = Path(job_meta.get("job_dir")) / "config.yaml"
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
                ).stdout.strip()
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