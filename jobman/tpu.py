import os
import time
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime

from jobman.utils import setup_logger

class TPU:
    
    def __init__(self, cfg, logger=None):
        self.name = cfg.tpu.name
        self.zone = cfg.tpu.zone
        self.accelerator = cfg.tpu.accelerator
        self.version = cfg.tpu.version
        self.pricing = cfg.tpu.pricing
        self.flags = cfg.tpu.get("flags", [])
        
        self.mode = cfg.tpu.allocation_mode
        self.logger = logger if logger is not None \
            else setup_logger(name='tpu', log_file=Path(cfg.job.dir) / "logs" / "tpu.log")
    
    def _run_command(self, cmd, redirect=None, check=True):
        if os.environ.get("JOBMAN_DEBUG", "").lower() in ("1", "true", "yes", "on"):
            self.logger.debug(f"Running command: {' '.join(cmd)}")
        return subprocess.run(
            cmd,
            stdout=redirect if redirect else subprocess.PIPE,
            stderr=redirect if redirect else subprocess.PIPE,
            text=True,
            check=check
        )
    
    def _check_tpu_vm_status(self):
        try:
            result = self._run_command([
                "gcloud", "alpha", "compute", "tpus", "tpu-vm", "describe",
                self.name, "--zone", self.zone, "--format=value(state)"
            ]).stdout
            if not result:
                return "NOT FOUND"
            return result.strip()
        except Exception as e:
            self.logger.error(f"Error checking TPU VM status: {e}")
            return "NOT FOUND"
    
    def _check_queued_resource_status(self):
        try:
            result = self._run_command([
                "gcloud", "compute", "tpus", "queued-resources", "describe",
                self.name, "--zone", self.zone, "--format=value(state)"
            ]).stdout
            if not result:
                return "NOT FOUND"
            return result.replace("state=", "")
        except Exception as e:
            self.logger.error(f"Error checking queued TPU status: {e}")
            return "ERROR"
        
    def _check_and_maybe_delete(self):
        assert self.mode in {"tpu-vm", "queued-resources"}
        if self.wait_tpu_vm_until_ready():
            self.logger.info("TPU is ready, no need to delete.")
            return True
        self.logger.info("TPU is not ready, deleting and retrying...")
        self.delete()
        return False
        
    def request(self):
        cmd = [
            "gcloud", "alpha" if self.mode == "tpu-vm" else "", "compute", "tpus",
            "tpu-vm" if self.mode == "tpu-vm" else "queued-resources",
            "create", self.name,
            "--zone", self.zone,
            "--accelerator-type", self.accelerator,
            "--version" if self.mode == "tpu-vm" else "--runtime-version", self.version
        ]

        if self.mode == "queued-resources":
            cmd += ["--node-id", self.name]

        if self.pricing == "preemptible":
            cmd += ["--preemptible"]
        elif self.pricing == "spot":
            cmd += ["--spot"]
            
        cmd += self.flags
        
        cmd = [x for x in cmd if x]

        if self.mode == "tpu-vm":
            return self._request_tpu_vm(cmd)
        else:
            return self._request_queued_resources(cmd)
        
    def _request_tpu_vm(self, cmd):
        for attempt in range(1, 1000):
            self.logger.info(f"Attempt {attempt}: Creating TPU VM...")
            result = self._run_command(cmd)
            if result.returncode == 0:
                self.logger.info("TPU VM created successfully.")
                return True
            self.logger.error("Failed. Retrying immediately...")

    def _request_queued_resources(self, cmd):
        if self._run_command(cmd).returncode != 0:
            self.logger.error("Failed to submit queued resource.")
            return False
        self.logger.info("Queued resource submitted. Polling until READY...")
        return self.wait_tpu_vm_until_ready()
    
    def wait_tpu_vm_until_ready(self, poll_interval=30, max_wait=9000):
        for i in range(max_wait // poll_interval):
            vm_status = self._check_tpu_vm_status()
            self.logger.info(f"VM status: {vm_status}")
            if vm_status in {"READY", "ACTIVE"}:
                self.logger.info("TPU is READY!")
                return True
            elif vm_status in {"FAILED", "DELETING", "UNSPECIFIED", "PREEMPTED"}:
                self.logger.error(f"TPU failed or disappeared: {vm_status}")
                return False
            elif vm_status in {"NOT FOUND"}:
                queue_status = self._check_queued_resource_status() 
                self.logger.info(f"Queue status: {queue_status}")
                if any(status in queue_status for status in ["FAILED", "SUSPENDED", "SUSPENDING", "ERROR", "NOT FOUND"]):
                    self.logger.error(f"Queued Resources failed: {queue_status}")
                    return False
            time.sleep(poll_interval)
        return False
    
    def get_ips(self):
        """Get internal and external IPs of all TPU workers."""
        try:
            result = self._run_command([
                "gcloud", "alpha", "compute", "tpus", "tpu-vm", "describe",
                self.name, "--zone", self.zone, "--format=json"
            ])
            data = json.loads(result.stdout)
            workers = data.get("networkEndpoints", [])

            ip_info = []
            for i, w in enumerate(workers):
                external_ip = w.get("accessConfig", {}).get("externalIp", "-")
                internal_ip = w.get("ipAddress", "-")
                ip_info.append({
                    "worker": i,
                    "internal_ip": internal_ip,
                    "external_ip": external_ip
                })

            return ip_info

        except subprocess.CalledProcessError as e:
            print(f"Failed to describe TPU: {e.stderr}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []
    
    def delete(self):
        self.logger.info(f"Deleting VM {self.name} in zone {self.zone}...")
        cmd = [
            "gcloud", "alpha", "compute", "tpus", "tpu-vm", "delete",
            self.name, "--zone", self.zone, "--quiet"
        ]
        try:
            self.logger.debug(f"Running command: {' '.join(cmd)}")
            self._run_command(cmd)
            self.logger.info("TPU VM deleted successfully.")
        except:
            self.logger.info("No TPU VM to delete or deletion failed (possibly already gone).")
            
        if self.mode == 'tpu-vm':
            return
            
        self.logger.info(f"Checking status of Queued Resources {self.name} in zone {self.zone}...")
        cmd = [
            "gcloud", "compute", "tpus", "queued-resources", "delete",
            self.name, "--zone", self.zone, "--quiet"
        ]
        try: 
            self.logger.debug(f"Running command: {' '.join(cmd)}")
            self._run_command(cmd)
            self.logger.info("Queued resources deleted.")
        except:
            self.logger.warning("No Queued resources to delete or deletion failed (possibly already gone).")

