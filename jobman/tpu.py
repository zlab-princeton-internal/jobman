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
            ]).stdout.strip()
            return result or "NOT FOUND"
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").strip()
            if "NOT_FOUND" in stderr or "was not found" in stderr.lower():
                return "NOT FOUND"
            self.logger.error(f"Error checking TPU VM status: {stderr or e}")
            return "UNKNOWN"
        except Exception as e:
            self.logger.error(f"Error checking TPU VM status: {e}")
            return "UNKNOWN"
    
    def _check_queued_resource_status(self):
        try:
            result = self._run_command([
                "gcloud", "compute", "tpus", "queued-resources", "describe",
                self.name, "--zone", self.zone, "--format=value(state)"
            ]).stdout.strip()
            if not result:
                return "NOT FOUND"
            for segment in result.split(";"):
                segment = segment.strip()
                if segment.startswith("state="):
                    state = segment.split("=", 1)[1].strip()
                    return state if state else "UNKNOWN"
            return result
        except Exception as e:
            self.logger.error(f"Error checking queued TPU status: {e}")
            return "UNKNOWN"

    def _get_queued_resource_error(self):
        """Get detailed error information for a failed queued resource."""
        try:
            result = self._run_command([
                "gcloud", "compute", "tpus", "queued-resources", "describe",
                self.name, "--zone", self.zone, "--format=json"
            ])
            data = json.loads(result.stdout)

            # Try to extract error information from various fields
            error_info = []

            # Check state message
            if "stateData" in data:
                state_data = data["stateData"]
                if "state" in state_data:
                    error_info.append(f"State: {state_data['state']}")
                if "stateInitiator" in state_data:
                    error_info.append(f"Initiator: {state_data['stateInitiator']}")

            # Check for error details
            if "queuedResourceState" in data:
                qr_state = data["queuedResourceState"]
                if "failedData" in qr_state:
                    failed_data = qr_state["failedData"]
                    if "error" in failed_data:
                        error_info.append(f"Error: {failed_data['error']}")

            # Check for any error or failure messages in the response
            if "error" in data:
                error_info.append(f"Error: {data['error']}")

            # If we found specific errors, return them
            if error_info:
                return " | ".join(error_info)

            # Otherwise return the full JSON for debugging
            return f"Full status: {json.dumps(data, indent=2)}"

        except Exception as e:
            return f"Could not retrieve error details: {e}"

    def _get_tpu_vm_error(self):
        """Get detailed error information for a failed TPU VM."""
        try:
            result = self._run_command([
                "gcloud", "alpha", "compute", "tpus", "tpu-vm", "describe",
                self.name, "--zone", self.zone, "--format=json"
            ])
            data = json.loads(result.stdout)

            # Try to extract error information
            error_info = []

            # Check health and symptoms
            if "health" in data:
                error_info.append(f"Health: {data['health']}")

            if "symptoms" in data:
                symptoms = data["symptoms"]
                if symptoms:
                    error_info.append(f"Symptoms: {symptoms}")

            # Check for error messages
            if "error" in data:
                error_info.append(f"Error: {data['error']}")

            # Check state details
            if "state" in data:
                error_info.append(f"State: {data['state']}")

            # If we found specific errors, return them
            if error_info:
                return " | ".join(error_info)

            # Otherwise return the full JSON for debugging
            return f"Full status: {json.dumps(data, indent=2)}"

        except Exception as e:
            return f"Could not retrieve error details: {e}"
        
    def check_and_maybe_delete(self):
        """Check TPU status and delete if in bad state.

        Returns:
            tuple: (ready: bool, state: str) where state is the TPU state
                   that caused the action (e.g., "READY", "PREEMPTED", "QUEUEING")
        """
        assert self.mode in {"tpu-vm", "queued-resources"}
        status = self._check_tpu_vm_status()
        if self.mode == "queued-resources" and status in {"NOT FOUND", "UNKNOWN"}:
            queue_status = self._check_queued_resource_status()
            if queue_status in {"FAILED", "SUSPENDED", "NOT FOUND", "UNKNOWN"}:
                if queue_status == "FAILED":
                    error_details = self._get_queued_resource_error()
                    self.logger.warning(
                        f"Queued resource is in unrecoverable state: {queue_status}. Deleting..."
                    )
                    self.logger.error(f"Failure details: {error_details}")
                else:
                    self.logger.warning(
                        f"Queued resource is in unrecoverable state: {queue_status}. Deleting..."
                    )
                # Delete the queued resource and requeue
                self.delete()
                return False, queue_status
            else:
                self.logger.info(
                    f"Queued resource status: {queue_status}. TPU VM not ready yet; leaving queued resource."
                )
                return False, "QUEUEING"
        if status in {"READY", "ACTIVE"}:
            return True, status
        elif status in {"PREEMPTED", "TERMINATED", "STOPPED", "SUSPENDED"}:
            self.logger.warning(f"TPU is in unrecoverable state: {status}. Deleting...")
            self.delete()
            return False, status
        elif status in {"CREATING", "PROVISIONING"}:
            self.logger.info("TPU is currently provisioning. Waiting until it becomes ready...")
            if self.wait_tpu_vm_until_ready():
                return True, "READY"
            else:
                self.logger.warning("TPU failed to become ready. Deleting...")
                self.delete()
                return False, "FAILED"
        elif status in {"NOT FOUND"}:
            self.delete()
            return False, status
        else:
            self.logger.error(f"Unexpected TPU status: {status}. Deleting as precaution.")
            self.delete()
            return False, status
        
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
            result = self._run_command(cmd, check=False)
            if result.returncode == 0:
                self.logger.info("TPU VM created successfully.")
                return True
            self.logger.error(f"Failed: {result.stderr.strip()}. Retrying immediately...")
        self.logger.error("Exhausted all retry attempts for TPU VM creation.")
        return False

    def _request_queued_resources(self, cmd):
        result = self._run_command(cmd, check=False)
        if result.returncode != 0:
            self.logger.error(f"Failed to submit queued resource: {result.stderr.strip()}")
            return False
        self.logger.info("Queued resource submitted. Polling until READY...")
        return self.wait_tpu_vm_until_ready()
    
    def wait_tpu_vm_until_ready(self, poll_interval=30, max_wait=168 * 60 * 60):
        for _ in range(max_wait // poll_interval):
            status = self._check_tpu_vm_status()
            if status in {"READY", "ACTIVE"}:
                self.logger.info("TPU is READY!")
                return True
            if status in {"FAILED", "DELETING", "UNSPECIFIED"}:
                self.logger.error(f"TPU failed or disappeared: {status}")
                if status == "FAILED":
                    error_details = self._get_tpu_vm_error()
                    self.logger.error(f"TPU VM failure details: {error_details}")
                return False

            if self.mode == "queued-resources" and status in {"NOT FOUND", "UNKNOWN"}:
                queue_status = self._check_queued_resource_status()
                self.logger.info(f"Queued resource status: {queue_status}")
                if queue_status == "FAILED":
                    self.logger.error(f"Queued Resources failed: {queue_status}")
                    error_details = self._get_queued_resource_error()
                    self.logger.error(f"Failure details: {error_details}")
                    return False
                if queue_status in {"NOT FOUND", "UNKNOWN"}:
                    self.logger.error(f"Queued Resources failed: {queue_status}")
                    return False
                if queue_status == "SUSPENDED":
                    self.logger.warning(f"Queued resource is SUSPENDED. Deleting and requeuing...")
                    self.delete()
                    return False
            else:
                self.logger.info(f"Current status: {status}")

            time.sleep(poll_interval)

        self.logger.error("TPU did not become ready within the allotted wait time.")
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
        """Delete TPU VM and queued resources."""
        self.logger.info(f"Checking status of TPU VM {self.name} in zone {self.zone}...")
        try:
            vm_status = self._check_tpu_vm_status()
        except:
            vm_status = "NOT FOUND"
        if vm_status != "NOT FOUND":
            self.logger.info(f"Deleting TPU VM {self.name} in zone {self.zone}...")
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
        else:
            self.logger.info("TPU VM not found. Skipping deletion.")

        if self.mode == 'tpu-vm':
            return

        self.logger.info(f"Checking status of Queued Resources {self.name} in zone {self.zone}...")
        try:
            queue_status = self._check_queued_resource_status()
        except:
            queue_status = "NOT FOUND"
        if queue_status != "NOT FOUND":
            self.logger.info(f"Deleting QUEUE {self.name} in zone {self.zone}...")
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
        else:
            self.logger.info("Queued resources not found. Skipping deletion.")
            
