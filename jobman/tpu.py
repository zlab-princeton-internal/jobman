import os
import time
import json
import shlex
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf

from jobman.utils import setup_logger, send_notification

class TPU:
    
    def __init__(self, config_path):
        cfg = OmegaConf.load(config_path)
        self.name = cfg.tpu.name
        self.zone = cfg.tpu.zone
        self.accelerator = cfg.tpu.acceleratorType
        self.version = cfg.tpu.version
        self.pricing = cfg.tpu.pricing
        self.flags = cfg.tpu.get("flags", [])
        self.project_id = cfg.tpu.project_id
        self.loop = cfg.loop
        
        self.logger = setup_logger(name='tpu', stdout=True)
    
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
    
    def _check_tpu_vm_status(self):
        tpu_vm_cmd = (
            "gcloud compute tpus tpu-vm"
            f" describe {self.name}"
            f" --zone {self.zone} --format='value(state)'"
        )
        return self._run_cmd(tpu_vm_cmd)
        
    def _check_queued_resource_status(self):
        queued_res_cmd = (
            "gcloud compute tpus queued-resources"
            f" describe {self.name}"
            f" --zone {self.zone} --format='value(state.state)'"
        )
        return self._run_cmd(queued_res_cmd)
    
    def _check_and_maybe_delete(self):
        if self._poll():
            self.logger.info("TPU is ready, no need to delete.")
            return True
        self.logger.info("TPU is not ready, deleting and retrying...")
        self.delete()
        return False
        
    def request(self):
        while 1:
            if self._check_and_maybe_delete():
                time.sleep(60)
                continue
            try:
                tpu_cmd = (
                    f"gcloud compute tpus queued-resources create {shlex.quote(self.name)} "
                    f"--zone {shlex.quote(self.zone)} "
                    f"--project {shlex.quote(self.project_id)} "
                    f"--accelerator-type={shlex.quote(self.accelerator)} "
                    f"--node-id={shlex.quote(self.name)} "
                    f"--runtime-version={shlex.quote(self.version)}"
                )
                if self.pricing in ["preemptible", "spot"]:
                    tpu_cmd += f" --{self.pricing}"
                
                for f in self.flags:
                    tpu_cmd += f" {f}"
                # request with tmux to avoid blocking the main thread
                
                subprocess.run(tpu_cmd, shell=True, check=True)
                result = self._poll()
                if result:
                    send_notification(f"TPU {self.name} is READY!", title="TPU Ready")
            except Exception as e:
                self.logger.error(f"Error requesting TPU: {e}")
            if not self.loop:
                break
            time.sleep(60)
    
    def _poll(self, poll_interval=60, max_wait=3000):
        for i in range(max_wait // poll_interval):
            vm_status = self._check_tpu_vm_status()
            self.logger.info(f"VM status: {vm_status}")
            if vm_status in {"READY", "ACTIVE"}:
                self.logger.info("TPU is READY!")
                return True
            elif vm_status in {"FAILED", "DELETING", "UNSPECIFIED", "PREEMPTED"}:
                self.logger.error(f"TPU failed or disappeared: {vm_status}")
                return False
            elif vm_status in {"ERROR"}:
                queue_status = self._check_queued_resource_status()
                self.logger.info(f"Queue status: {queue_status}")
                if any(status in queue_status for status in ["FAILED", "SUSPENDED", "SUSPENDING", "ERROR", "NOT FOUND"]):
                    self.logger.error(f"Queued Resources failed: {queue_status}")
                    return False
            time.sleep(poll_interval)
        return False
    
    def delete(self):
        delete_vm_cmd = (
            "gcloud alpha compute tpus tpu-vm delete"
            f" {self.name} --zone {self.zone} --quiet"
        )
        result = self._run_cmd(delete_vm_cmd)
        msg = "TPU VM deleted successfully." if result != "ERROR" \
            else "No TPU VM to delete or deletion failed (possibly already gone)."
        self.logger.info(msg)
            
        delete_queued_cmd = (
            "gcloud compute tpus queued-resources delete"
            f" {self.name} --zone {self.zone} --project {self.project_id} --quiet"
        )
        result = self._run_cmd(delete_queued_cmd)
        msg = "Queued resources deleted successfully." if result != "ERROR" \
            else "No Queued resources to delete or deletion failed (possibly already gone)."
        self.logger.info(msg)
    
    def get_ips(self):
        """Get internal and external IPs of all TPU workers."""
        cmd = (
            "gcloud alpha compute tpus tpu-vm describe"
            f" {self.name} --zone {self.zone} --format=json"
        )
        result = self.run_cmd(cmd)
        if result == "ERROR":
            return []
        data = json.loads(result)
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
        