import argparse
import subprocess
import concurrent.futures
from pathlib import Path
from textwrap import dedent
from omegaconf import OmegaConf
from collections.abc import Iterable
from jobman.utils import setup_logger

class COMMAND:
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.base_cmd = cfg.command.cmd
        self.full_cmd = None
        self.workers = self.infer_workers() 
        
        self.logger = setup_logger(log_file=Path(cfg.job.dir) / "logs" / "job.log")
        
    def infer_workers(self):
        accelerator = self.cfg.tpu.accelerator
        num_workers = self.cfg.tpu.num_workers
        worker_spec = self.cfg.command.get("workers", "all")

        if worker_spec == "all":
            return list(range(num_workers))

        elif isinstance(worker_spec, int):
            if not (0 <= worker_spec < num_workers):
                print(f"Invalid worker index: {worker_spec}. Only {num_workers} workers available.", "ERROR")
                return []
            return [worker_spec]

        elif isinstance(worker_spec, Iterable):
            workers = []
            seen = set()
            for w in worker_spec:
                if not isinstance(w, int) or not (0 <= w < num_workers):
                    log(f"Invalid worker index in list: {w}. Only {num_workers} workers available.", "ERROR")
                    return []
                if w in seen:
                    log(f"Duplicate worker index specified: {w}.", "ERROR")
                    return []
                seen.add(w)
                workers.append(w)
            return workers

        else:
            log(f"Invalid type for 'worker': {type(worker_spec)}. Must be 'all', int, or list of int.", "ERROR")
            return []

    def run(self):
        self.logger.info(f"Launching command on workers: {self.workers}")
        if self.full_cmd is None:
            self.full_cmd = self.base_cmd
        self.logger.debug("Executing command:")
        self.logger.debug(self.full_cmd) 
        all_success = True

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            futures = {executor.submit(self.run_worker, i): i for i in self.workers}
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                if not future.result():
                    all_success = False

        if all_success:
            self.logger.info("Command ran successfully on all workers.")
        else:
            self.logger.warning("Command failed on one or more workers.")
        return all_success
    
    def run_worker(self, i):
        self.logger.info(f"Worker {i}: Launching command")  

        ssh_cmd = [
            "gcloud", "alpha", "compute", "tpus", "tpu-vm", "ssh", self.cfg.tpu.name,
            "--zone", self.cfg.tpu.zone,
            f"--worker={i}",
            "--ssh-flag=-o ConnectTimeout=15",
            "--ssh-flag=-o StrictHostKeyChecking=no",
            "--ssh-flag=-o UserKnownHostsFile=/dev/null",
            "--command", f"stdbuf -oL -eL bash -lc \"{self.full_cmd}\"",
            "--quiet",
        ]

        log_file = Path(self.cfg.job.dir) / "logs" / f"main_command_worker_{i}.log"
        with open(log_file, "a") as f:
            result = subprocess.run(ssh_cmd, stdout=f, stderr=f)
            if result.returncode != 0:
                self.logger.error(f"Worker {i}: command failed.")
                return False
        return True
    