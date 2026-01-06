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
        v6e-8  -> 1 worker (single host, 8 chips)
        v6e-32 -> 8 workers (32 // 4)
    """
    match = re.search(r"v(\d+)[a-z]*-(\d+)", accelerator.lower())
    if not match:
        raise ValueError(f"Invalid accelerator format: {accelerator}")

    version, chips = int(match.group(1)), int(match.group(2))
    # v4: 2 cores/chip, 4 chips/host → chips/8
    # v5e/v5p: 4 chips/host → chips/4
    # v6e: 8 chips/host for single-host (v6e-8), 4 chips/host for multi-host
    if version in [2, 3, 4]:
        return math.ceil(chips / 8)
    elif version == 5:
        return math.ceil(chips / 4)
    elif version == 6:
        # v6e-8 is single host (8 chips), larger configs are 4 chips/host
        if chips <= 8:
            return 1
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

            # Collect all buckets to check: primary + extra_mounts
            buckets_to_check = [cfg.gcsfuse.bucket_name]

            extra_mounts = getattr(cfg.gcsfuse, 'extra_mounts', None)
            if extra_mounts:
                from omegaconf import OmegaConf
                if OmegaConf.is_config(extra_mounts):
                    extra_mounts = OmegaConf.to_container(extra_mounts, resolve=True)
                for mount in extra_mounts:
                    extra_bucket = mount.get('bucket_name')
                    if extra_bucket and extra_bucket not in buckets_to_check:
                        buckets_to_check.append(extra_bucket)

            # Check each bucket's region
            for bucket_name in buckets_to_check:
                try:
                    region = subprocess.run(
                        ["gcloud", "storage", "buckets", "describe", f"gs://{bucket_name}", "--format=value(location)"],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
                    ).stdout.strip().lower()
                except Exception as e:
                    raise Exception(f"Checking bucket gs://{bucket_name} region failed: {e}")

                if not zone.startswith(region):
                    raise ValueError(f"Bucket '{bucket_name}' region '{region}' does not match TPU VM zone '{cfg.tpu.zone}'. It's strongly suggested to confirm these configurations match to minimize cost.")

                self.logger.info(f"  bucket '{bucket_name}' region '{region}' matches zone '{zone}'")

        return True
            
    def _find_next_worker_num(self, base_name):
        """Find the next available worker number for a given base TPU name."""
        import re
        used_nums = set()

        # Check all existing jobs for matching TPU names
        with self.with_meta_lock() as meta:
            for job_id, job_meta in meta.items():
                job_dir = job_meta.get('job_dir')
                if not job_dir:
                    continue
                config_path = Path(job_dir) / 'config.yaml'
                if not config_path.exists():
                    continue
                try:
                    job_cfg = OmegaConf.load(config_path)
                    tpu_name = job_cfg.tpu.name
                    # Check if it matches pattern: base_name_N
                    if tpu_name.startswith(base_name + '_'):
                        suffix = tpu_name[len(base_name) + 1:]
                        if suffix.isdigit():
                            used_nums.add(int(suffix))
                except Exception:
                    continue

        # Find smallest available number starting from 1
        num = 1
        while num in used_nums:
            num += 1
        return num

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

        # Use worker_num if specified, auto-detect if "auto", otherwise use job_id
        worker_num = cfg.job.get('worker_num', None)
        if worker_num == 'auto' or (worker_num is None and cfg.job.get('queue', {}).get('enabled', False)):
            # Auto-detect next available worker number for queue workers
            # Resolve any variable interpolation first
            base_name = OmegaConf.to_container(cfg, resolve=True)['tpu']['name']
            worker_num = self._find_next_worker_num(base_name)
            self.logger.info(f"Auto-assigned worker_num: {worker_num}")
            suffix = str(worker_num)
        elif worker_num is not None:
            suffix = str(worker_num)
        else:
            suffix = job_id

        cfg.job.worker_num = worker_num if worker_num is not None else None
        cfg.job.name = f"{cfg.job.name}_{suffix}"
        cfg.tpu.name = f"{cfg.tpu.name}_{suffix}"
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
    
    def start_job(self, job_id, force=False, force_prefix=False):
        with self.with_meta_lock() as meta:
            job_meta = meta.get(job_id)
        job_dir = Path(job_meta.get("job_dir"))
        session_name = f"job_{job_id}"

        logs_dir = job_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "job.log"

        config_path = job_dir / "config.yaml"
        run_cmd = f"jobman run {job_id}"
        if force:
            run_cmd += " --force"
        if force_prefix:
            run_cmd += " --force-prefix"

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

        # Release any queue tasks claimed by this job (before killing tmux)
        self._release_queue_tasks(job_id, job_meta)

        session_name = job_meta.get("session_name", None)
        if not session_name:
            self.logger.warning(f"No tmux session name found for job {job_id}")
            self._write_job_status(job_meta.get('job_dir'), "IDLE")
            return True  # Nothing to stop, but not an error

        if not self.check_tmux_session(session_name):
            self.logger.info(f"Session '{session_name}' does not exist. Nothing to stop.")
            self._write_job_status(job_meta.get('job_dir'), "IDLE")
            return True  # Already stopped, not an error

        try:
            subprocess.run(["tmux", "kill-session", "-t", session_name], check=True)
            self._write_job_status(job_meta.get('job_dir'), "IDLE")
            self.logger.info(f"Stopped job {job_id} by killing tmux session {session_name}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to kill tmux session {session_name}: {e}")
            return False

    def _write_job_status(self, job_dir, status):
        """Write job status to .job_status file."""
        if job_dir:
            status_file = Path(job_dir) / '.job_status'
            try:
                status_file.write_text(status)
            except Exception:
                pass  # Best effort

    def _release_queue_tasks(self, job_id, job_meta):
        """Release any queue tasks claimed by this job."""
        job_dir = job_meta.get('job_dir')
        if not job_dir:
            return
        config_path = Path(job_dir) / 'config.yaml'
        if not config_path.exists():
            return
        try:
            cfg = OmegaConf.load(config_path)
            queue_cfg = cfg.job.get('queue', {})
            if not queue_cfg.get('enabled', False):
                return
            queue_config_path = queue_cfg.get('config')
            if not queue_config_path:
                return
            # Resolve relative to job config directory
            queue_config_path = Path(queue_config_path)
            if not queue_config_path.is_absolute():
                queue_config_path = (Path(job_dir) / queue_config_path).resolve()
            if not queue_config_path.exists():
                return
            from jobman.queue import Queue
            q = Queue(queue_config_path, self.logger)
            released = q.release_all_for_job(job_id)
            if released:
                self.logger.info(f"Released queue tasks for job {job_id}: {released}")
        except Exception as e:
            self.logger.warning(f"Failed to release queue tasks for job {job_id}: {e}")
            
    def delete_job(self, job_id):
        """Delete a job and its TPU."""
        self.stop_job(job_id)

        with self.with_meta_lock() as meta:
            job_meta = meta.get(job_id)
            if not job_meta:
                self.logger.error(f"Job {job_id} not found in metadata")
                return False

        job_dir = job_meta.get("job_dir")
        config_path = Path(job_meta.get("config_path"))
        if config_path.exists():
            try:
                cfg = OmegaConf.load(config_path)
                tpu = TPU(cfg, self.logger)
                tpu.delete()
            except Exception as e:
                self.logger.exception(f"Failed to delete TPU for job {job_id}: {e}")
        else:
            self.logger.warning(f"Job {job_id} config not found at {config_path}, skipping TPU deletion")

        # Write DEAD status before removing from metadata
        self._write_job_status(job_dir, "DEAD")

        # Remove from metadata
        with self.with_meta_lock() as meta:
            if job_id in meta:
                del meta[job_id]
                self.logger.info(f"Removed job {job_id} from metadata")

        return True
    
    def clean_job(self, job_id):
        # Get job_dir before delete_job removes metadata
        with self.with_meta_lock() as meta:
            job_meta = meta.get(job_id)
            if not job_meta:
                self.logger.error(f"Job {job_id} not found in metadata")
                return False
            job_dir = Path(job_meta.get("job_dir"))

        if not self.delete_job(job_id):
            self.logger.error(f"Job {job_id} deletion failed")
            return False

        # Delete job directory
        try:
            shutil.rmtree(job_dir, ignore_errors=True)
            self.logger.info(f"Deleted job directory {job_dir}")
        except Exception as e:
            self.logger.error(f"Failed to delete job directory {job_dir}: {e}")

        self.logger.info(f"Cleaned job {job_id} successfully")
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

        # Summary: total cores by zone (only RUNNING/IDLE jobs with active TPUs)
        cores_by_zone = {}  # zone -> {chip_type -> total_cores}
        for row in rows:
            # row = [job_id, user, job_name, accelerator, zone, host0_ip, status]
            accelerator, zone, status = row[3], row[4], row[6]
            if status not in {"RUNNING", "IDLE", "QUEUEING"}:
                continue
            if accelerator == "N/A" or zone == "N/A":
                continue
            # Parse accelerator: v6e-64 -> chip_type=v6e, cores=64
            try:
                parts = accelerator.rsplit('-', 1)
                if len(parts) == 2:
                    chip_type = parts[0]  # e.g., v6e, v4, v5e
                    cores = int(parts[1])
                    if zone not in cores_by_zone:
                        cores_by_zone[zone] = {}
                    if chip_type not in cores_by_zone[zone]:
                        cores_by_zone[zone][chip_type] = 0
                    cores_by_zone[zone][chip_type] += cores
            except (ValueError, IndexError):
                pass

        if cores_by_zone:
            print("\nTotal cores:")
            for zone in sorted(cores_by_zone.keys()):
                chips = cores_by_zone[zone]
                chip_strs = [f"{chip}-{cores}" for chip, cores in sorted(chips.items())]
                print(f"  {zone}: {', '.join(chip_strs)}")
            
    def fetch_job_info(self, job_id, job_meta):
        try:
            user = job_meta.get("user")
            session_name = job_meta.get("session_name", f"job_{job_id}")
            job_dir = Path(job_meta.get("job_dir"))
            config_path = job_dir / "config.yaml"
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

            has_tmux = self.check_tmux_session(session_name)
            status_file = job_dir / '.job_status'

            # Always use gcloud for accurate TPU state
            try:
                gcloud_state = subprocess.run(
                    [
                        "gcloud", "alpha", "compute", "tpus", "tpu-vm", "describe",
                        cfg.tpu.name, "--zone", cfg.tpu.zone, "--format=value(state)"
                    ],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                ).stdout.strip()
            except:
                gcloud_state = ""

            if has_tmux:
                if gcloud_state in {"READY", "ACTIVE"}:
                    status = "RUNNING"
                else:
                    # TPU not ready (preempted/suspended/gone) - job will detect and requeue
                    status = "QUEUEING"
            else:
                # No tmux session - check status file first, then gcloud
                if status_file.exists():
                    file_status = status_file.read_text().strip()
                    if file_status in {"IDLE", "DEAD", "PREFIX_FAIL"}:
                        status = file_status
                    elif gcloud_state in {"READY", "ACTIVE"}:
                        status = "IDLE"
                    else:
                        status = "DEAD"
                elif cfg:
                    if gcloud_state in {"READY", "ACTIVE"}:
                        status = "IDLE"
                    else:
                        status = "DEAD"
                else:
                    status = "UNKNOWN"

            return [job_id, user, job_name, accelerator, zone, host0_ip, status]
        except Exception as e:
            return [job_id, "ERROR", "ERROR", "ERROR", "ERROR", "ERROR", f"ERROR: {e}"]
