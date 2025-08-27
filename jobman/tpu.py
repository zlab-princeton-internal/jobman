import time
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime

from jobman.utils import setup_logger

class TPU:
    
    def __init__(self, cfg):
        self.name = cfg.tpu.name
        self.zone = cfg.tpu.zone
        self.accelerator = cfg.tpu.accelerator
        self.version = cfg.tpu.version
        self.pricing = cfg.tpu.pricing
        self.tags = cfg.tpu.tags
        self.metadata = cfg.tpu.metadata
        self.startup_script = cfg.tpu.get("startup_script", None)
        
        self.mode = cfg.tpu.allocation_mode
        self.log_file = Path(cfg.job.dir) / "logs" / "tpu.log"
        self.logger = setup_logger(log_file=self.log_file)
        
    def check_tpu_status(self):
        """Check current status of the TPU."""
        if self.mode == "tpu-vm":
            return self._check_tpu_vm_status()
        else:
            return self._check_queued_resource_status()
    
    def _check_tpu_vm_status(self):
        try:
            result = subprocess.run(
                [
                    "gcloud", "alpha", "compute", "tpus", "tpu-vm", "describe",
                    self.name, "--zone", self.zone, "--format=value(state)"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return result.stdout.strip() or "NOT FOUND"
        except Exception as e:
            self.logger.error(f"Error checking TPU VM status: {e}")
            return "UNKNOW"
    
    def _check_queued_resource_status(self):
        try:
            result = subprocess.run(
                [
                    "gcloud", "compute", "tpus", "queued-resources", "describe",
                    self.name, "--zone", self.zone, "--format=value(state)"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            result = result.stdout.strip().replace("state=", "")
            return result if result else "NOT FOUND"
        except Exception as e:
            self.logger.error(f"Error checking queued TPU status: {e}")
            return "UNKNOWN"
        
    def check_and_maybe_delete(self):
        assert self.mode in {"tpu-vm", "queued-resources"}
        status = self._check_tpu_vm_status()
        if status in {"READY", "ACTIVE"}:
            return True
        elif status in {"PREEMPTED", "TERMINATED", "STOPPED", "SUSPENDED"}:
            self.logger.warning(f"TPU is in unrecoverable state: {status}. Deleting...")
            self.delete()
        elif status in {"CREATING", "PROVISIONING"}:
            self.logger.info("TPU is currently provisioning. Waiting until it becomes ready...")
            if self.wait_tpu_vm_until_ready():
                return True
            else:
                self.logger.warning("TPU failed to become ready. Deleting...")
                self.delete()
        elif status in {"NOT FOUND"}:
            self.delete()
        else:
            self.logger.error(f"Unexpected TPU status: {status}. Deleting as precaution.")
            self.delete()
        
        return False
        
    def request(self):
        
        base_cmd = [
            "gcloud", "alpha" if self.mode == "tpu-vm" else "", "compute", "tpus",
            "tpu-vm" if self.mode == "tpu-vm" else "queued-resources",
            "create", self.name,
            "--zone", self.zone,
            "--accelerator-type", self.accelerator,
            "--version" if self.mode == "tpu-vm" else "--runtime-version", self.version
        ]

        if self.mode == "queued-resources":
            base_cmd += ["--node-id", self.name]

        if self.pricing == "preemptible":
            base_cmd += ["--preemptible"]
        elif self.pricing == "spot":
            base_cmd += ["--spot"]

        if self.startup_script:
            base_cmd += ["--metadata", f"startup-script={self.startup_script}"]
        elif self.metadata:
            meta_str = ",".join(f"{k}={v}" for k, v in self.metadata.items())
            base_cmd += ["--metadata", meta_str]

        if self.tags:
            base_cmd += ["--tags", ",".join(self.tags)]

        # Clean empty strings from gcloud command
        cmd = [x for x in base_cmd if x]

        self.logger.debug("TPU creation command:")
        self.logger.debug(" ".join(cmd))

        if self.mode == "tpu-vm":
            return self._request_tpu_vm(cmd)
        else:
            return self._request_queued_resources(cmd)
        
    def _request_tpu_vm(self, cmd):
        attempt = 1
        while True:
            self.logger.info(f"Attempt {attempt}: Creating TPU VM...")
            with open(self.log_file, "a") as f:
                result = subprocess.run(cmd, stdout=f, stderr=f)
            if result.returncode == 0:
                self.logger.info("TPU VM created successfully.")
                return True
            self.logger.error("Failed. Retrying immediately...")
            attempt += 1
    
    def _request_queued_resources(self, cmd):
        with open(self.log_file, "a") as f:
            result = subprocess.run(cmd, stdout=f, stderr=f)

        if result.returncode != 0:
            self.logger.error("Failed to submit queued resource.")
            return False

        self.logger.info("Queued resource submitted. Polling until READY...")

        return self.wait_tpu_vm_until_ready()
    
    def wait_tpu_vm_until_ready(self, poll_interval=30):
        while True:
            status = self._check_tpu_vm_status()
            self.logger.info(f"Current status: {status}")
            if status in {"READY", "ACTIVE"}:
                self.logger.info("TPU is READY!")
                return True
            elif status in {"FAILED", "DELETING", "UNSPECIFIED"}:
                self.logger.error(f"TPU failed or disappeared: {status}")
                return False
            elif status in {"NOT FOUND"}:
                queue_status =  self._check_queued_resource_status() 
                if queue_status in {"FAILED", "SUSPENDED"}:
                    self.logger.error(f"Queued Resources failed: {queue_status}")
                    return False
            time.sleep(poll_interval)
    
    def get_ips(self):
        """Get internal and external IPs of all TPU workers."""
        try:
            result = subprocess.run(
                [
                    "gcloud", "alpha", "compute", "tpus", "tpu-vm", "describe",
                    self.name, "--zone", self.zone, "--format=json"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
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
        self.logger = setup_logger(stdout=True)
        
        self.logger.info(f"Deleting TPU {self.name} in zone {self.zone}...")
        
        vm_status = self._check_tpu_vm_status()
        if vm_status != "NOT FOUND":
            cmd = [
                "gcloud", "alpha", "compute", "tpus", "tpu-vm", "delete",
                self.name, "--zone", self.zone, "--quiet"
            ]
            try:
                self.logger.debug(f"Running command: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
                self.logger.info("TPU VM deleted successfully.")
            except:
                self.logger.info("No TPU VM to delete or deletion failed (possibly already gone).")
        else:
            self.logger.info("TPU VM not found. Skipping deletion.")
            
        if self.mode == "queued-resources":
            queue_status = self._check_queued_resource_status()
            if queue_status != "NOT FOUND":
                cmd = [
                    "gcloud", "compute", "tpus", "queued-resources", "delete",
                    self.name, "--zone", self.zone, "--quiet"
                ]
                try:
                    self.logger.debug(f"Running command: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)
                    self.logger.info("Queued resources deleted.")
                except:
                    self.logger.warning("No Queued resources to delete or deletion failed (possibly already gone).")
            else:
                self.logger.info("Queued resources not found. Skipping deletion.")
            

