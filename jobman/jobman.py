import re
import os
import pdb
import glob
import math
import json
import time
import fcntl
import shlex
import shutil
import logging
import textwrap
import subprocess
from pathlib import Path
from datetime import datetime 
from tabulate import tabulate
from omegaconf import OmegaConf
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed

from jobman.tpu import TPU
from jobman.utils import setup_logger, add_log_file, send_notification

BASE_DIR = Path(__file__).resolve().parent 
log_dir = (BASE_DIR / ".." / "logs").resolve()
jobman_dir = log_dir / ".meta"

ZONES = [
    "asia-northeast1-b",
    "us-central2-b",
    "us-east1-d",
    "europe-west4-a",
    "europe-west4-b",
]

PROJECT_ID = "vision-mix"
USER_NAME = "yufeng"

class JobMan:

    def __init__(self):
        jobman_dir.mkdir(parents=True, exist_ok=True)
        self.lock_file = jobman_dir / "lock"
        self.tpu_id_file = jobman_dir / "next_tpu_id.txt"
        self.task_id_file = jobman_dir / "next_task_id.txt"
        if not self.lock_file.exists():
            self.lock_file.touch()
        if not self.tpu_id_file.exists():
            self._atomic_write_text(self.tpu_id_file, "0\n")
        if not self.task_id_file.exists():
            self._atomic_write_text(self.task_id_file, "0\n")
        self.logger = setup_logger(stdout=True)
        
    def _run_cmd(self, cmd):
        try:
            self.logger.debug(f"Running cmd {cmd}")
            result = subprocess.run(
                cmd, 
                shell=True, check=True,  capture_output=True, text=True,
            )
            return result.stdout.strip()
        except Exception as e:
            self.logger.error(f"Error running command '{cmd}': {e}")
            return "ERROR"
        
    def _atomic_write_text(self, path: str, text: str) -> None:
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, path)
            
    def get_next_task_id(self):
        with open(self.lock_file, "r+") as lock_fp:
            fcntl.flock(lock_fp, fcntl.LOCK_EX)
            current = int(self.task_id_file.read_text())
            next_id = current + 1
            self.task_id_file.write_text(str(next_id))
            fcntl.flock(lock_fp, fcntl.LOCK_UN)
            return f"{next_id:06d}"
        
    def get_next_tpu_id(self):
        with open(self.lock_file, "r+") as lock_fp:
            fcntl.flock(lock_fp, fcntl.LOCK_EX)
            current = int(self.tpu_id_file.read_text())
            next_id = current + 1
            self.tpu_id_file.write_text(str(next_id))
            fcntl.flock(lock_fp, fcntl.LOCK_UN)
            return next_id
    
    def create_tpu(self, cfg_path):
        cfg_path = Path(cfg_path).resolve()
        tpus_dir = (log_dir / "tpus").resolve()
        # save config
        if tpus_dir not in cfg_path.parents:
            tpu_id = f"{self.get_next_tpu_id():04d}"
            cfg = OmegaConf.load(cfg_path)
            cfg.tpu.id = tpu_id
            cfg.tpu.name = f"{USER_NAME}-{cfg.tpu.acceleratorType}-{cfg.tpu.id}"
            cfg.tpu.project_id = PROJECT_ID
            
            tpu_dir = log_dir / "tpus" / f"{cfg.tpu.id}-{cfg.tpu.acceleratorType}"
            cfg.tpu.dir = str(tpu_dir)
            tpu_dir.mkdir(parents=True, exist_ok=True)
            cfg_path = tpu_dir / "config.yaml"
            OmegaConf.save(cfg, cfg_path)
        else:
            tpu_id = cfg_path.parent.name.split("-")[0]
            tpu_dir = cfg_path.parent
            
        cmd = f"python -c 'from jobman.tpu import TPU; tpu = TPU(\\\"{cfg_path}\\\"); tpu.request()'"
        tmux_cmd = f'tmux new-session -d -s tpu_{tpu_id} "{cmd} 2>&1 | tee -a {shlex.quote(str(tpu_dir / "tpu.log"))}"'
        result = self._run_cmd(tmux_cmd)
        self.logger.info(f"See TPU request logs at {tpu_dir / 'tpu.log'}")
        
    def create_task(self, cfg_path):
        pass
    
    def delete_tpu(self, tpu_id):
        tpus_dir = (log_dir / "tpus").resolve()
        self.logger.info("Killing TPU request session if exists...")
        kill_tmux_cmd = f"tmux kill-session -t tpu_{tpu_id}"
        result = self._run_cmd(kill_tmux_cmd)

        for d in tpus_dir.iterdir():
            if d.is_dir() and d.name.split('-')[0] == tpu_id:
                target_tpu_dir = d
                config_path = target_tpu_dir / "config.yaml"
                tpu = TPU(config_path)
                tpu.delete()
                return

        self.logger.error(f"TPU with id {tpu_id} not found.")
    
    def cancel(self, task_id):
        cfg_path = glob.glob(str(log_dir / f"{job_id}_*" / "config.yaml"))
        if not cfg_path:
            self.logger.error(f"Job {job_id} not found.")
            return

        session_names = [s.split(":")[0] for s in subprocess.run("tmux ls", shell=True, capture_output=True, text=True).stdout.splitlines() if s.startswith(f"{job_id}_")]
        if not session_names:
            self.logger.error(f"Job {job_id} already completed or dead.")
            return
        subprocess.run(f"tmux kill-session -t {session_names[0]}", shell=True)
        self.logger.info(f"Killed tmux session: {session_names[0]}")
        
        cfg = OmegaConf.load(cfg_path[0])
        tpu_name = cfg.tpu.name
        zone = cfg.tpu.zone
        return tpu_name, zone
    
    def find_idle_tpu(self, acceleratorType, zone, job_id):
        tpus = self.list_queues(zone)
        tasks = self.list_tasks()

        used_tpus = set()
        for t in tasks:
            used_tpus.add((t["tpu_name"], t["tpu_zone"], t["tpu_accelerator"]))
        
        for tpu in tpus:
            if (
                tpu["acceleratorType"] == acceleratorType and
                (tpu["name"], zone, acceleratorType) not in used_tpus and
                tpu["state"] == "READY"
            ):
                return tpu
        
        return False
        
    def list_queues(self, zone=None):
        queues = []
        def parse_results(zone, results):
            results = json.loads(results)
            for r in results:
                t = {
                    "acceleratorType": r["tpu"]["nodeSpec"][0]["node"]["acceleratorType"],
                    "name": r["name"].split("/")[-1],
                    "zone": zone,
                    "state": r["state"]["state"],
                }
                if t["name"].startswith("yufeng"):
                    queues.append(t)
        
        zones_to_search = [zone] if zone else ZONES
        for z in zones_to_search:
            cmd = f"gcloud compute tpus queued-resources list --zone {z} --project {PROJECT_ID} --format=json"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            parse_results(z, result.stdout)
            
        return queues
    
    def list_tasks(self):
        job_id_names = [d.name for d in log_dir.iterdir() if d.is_dir() and re.match(r"^\d{6}_", d.name)]
        session_names = [s.split(":")[0] for s in subprocess.run("tmux ls", shell=True, capture_output=True, text=True).stdout.splitlines()]
        jobs = []
        for job_id_name in job_id_names:
            if job_id_name not in session_names:
                continue
            # status = "ACTIVE" if os.path.exists(str(log_dir / f"{job_id_name}" / "config.yaml")) else "PENDING"
            cfg = OmegaConf.load(str(log_dir / f"{job_id_name}" / "config.yaml"))
            res = subprocess.run(f"gcloud compute tpus queued-resources describe {cfg.tpu.name} --zone {cfg.tpu.zone} --project {PROJECT_ID} --format=json", shell=True, capture_output=True, text=True).stdout
            try:
                state = json.loads(res)["state"]["state"]
            except:
                state = "UNKNOWN"
            j = {
                "job_id": job_id_name.split("_")[0],
                "name": "_".join(job_id_name.split("_")[1:]),
                "tpu_name": cfg.tpu.name,
                "tpu_zone": cfg.tpu.zone,
                "tpu_accelerator": cfg.tpu.acceleratorType,
                "state": state
            }
            jobs.append(j)
        print(tabulate(jobs, headers="keys", tablefmt="github"))
        return jobs