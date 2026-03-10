"""TPU lifecycle management for jobman-lite."""

import json
import subprocess
import time
from dataclasses import dataclass, field
from typing import Literal

from .utils import get_logger

logger = get_logger(__name__)

TPUStatus = Literal["READY", "CREATING", "PREEMPTED", "TERMINATED", "SUSPENDED",
                    "NOT_FOUND", "UNKNOWN"]
AllocationMode = Literal["tpu-vm", "queued-resources"]
Pricing = Literal["spot", "preemptible", "standard"]

# Default runtime version for TPU VMs
DEFAULT_TPU_VERSION = "tpu-ubuntu2204-base"


def _run(cmd: list[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    logger.debug("Running: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=capture, text=True, check=check)


@dataclass
class TPU:
    name: str
    zone: str
    accelerator: str          # e.g. "v4-8", "v5e-16"
    version: str = DEFAULT_TPU_VERSION
    pricing: Pricing = "spot"
    mode: AllocationMode = "tpu-vm"

    # Derived fields
    _queued_resource_id: str = field(default="", init=False, repr=False)

    def __post_init__(self):
        self._queued_resource_id = f"qr-{self.name}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request(self) -> None:
        """Create the TPU VM (or queued resource)."""
        if self.mode == "tpu-vm":
            self._create_tpu_vm()
        else:
            self._create_queued_resource()

    def status(self) -> TPUStatus:
        """Return the current TPU status."""
        if self.mode == "tpu-vm":
            return self._tpu_vm_status()
        else:
            return self._queued_resource_status()

    def delete(self) -> None:
        """Delete the TPU VM (and queued resource if applicable)."""
        if self.mode == "queued-resources":
            self._delete_queued_resource()
        self._delete_tpu_vm()

    def wait_ready(self, timeout: int = 86400, poll_interval: int = 30) -> None:
        """Poll until READY or timeout. Uses exponential backoff up to poll_interval."""
        logger.info("Waiting for TPU %s to be READY (timeout=%ds)...", self.name, timeout)
        start = time.time()
        interval = 10
        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                raise TimeoutError(f"TPU {self.name} not ready after {timeout}s")
            st = self.status()
            if st == "READY":
                logger.info("TPU %s is READY (%.0fs elapsed)", self.name, elapsed)
                return
            if st in ("PREEMPTED", "TERMINATED"):
                raise RuntimeError(f"TPU {self.name} entered terminal state: {st}")
            logger.info("TPU %s status=%s, waiting %ds...", self.name, st, interval)
            time.sleep(interval)
            interval = min(interval * 2, poll_interval)

    def get_num_workers(self) -> int:
        """Infer number of TPU workers from accelerator type."""
        return _num_workers(self.accelerator)

    # ------------------------------------------------------------------
    # TPU-VM mode helpers
    # ------------------------------------------------------------------

    def _create_tpu_vm(self) -> None:
        cmd = [
            "gcloud", "alpha", "compute", "tpus", "tpu-vm", "create", self.name,
            f"--zone={self.zone}",
            f"--accelerator-type={self.accelerator}",
            f"--version={self.version}",
        ]
        if self.pricing == "spot":
            cmd.append("--spot")
        elif self.pricing == "preemptible":
            cmd.append("--preemptible")
        logger.info("Creating TPU VM %s (%s) in %s [%s]...", self.name, self.accelerator,
                    self.zone, self.pricing)
        _run(cmd, check=True, capture=False)

    def _tpu_vm_status(self) -> TPUStatus:
        result = _run([
            "gcloud", "compute", "tpus", "tpu-vm", "describe", self.name,
            f"--zone={self.zone}", "--format=json"
        ], check=False)
        if result.returncode != 0:
            if "NOT_FOUND" in result.stderr or "not found" in result.stderr.lower():
                return "NOT_FOUND"
            return "UNKNOWN"
        try:
            info = json.loads(result.stdout)
            state = info.get("state", "UNKNOWN").upper()
            return _normalize_status(state)
        except (json.JSONDecodeError, KeyError):
            return "UNKNOWN"

    def _delete_tpu_vm(self) -> None:
        logger.info("Deleting TPU VM %s...", self.name)
        result = _run([
            "gcloud", "compute", "tpus", "tpu-vm", "delete", self.name,
            f"--zone={self.zone}", "--quiet"
        ], check=False)
        if result.returncode != 0 and "NOT_FOUND" not in result.stderr:
            logger.warning("Failed to delete TPU VM %s: %s", self.name, result.stderr)

    # ------------------------------------------------------------------
    # Queued-resources mode helpers
    # ------------------------------------------------------------------

    def _create_queued_resource(self) -> None:
        cmd = [
            "gcloud", "alpha", "compute", "tpus", "queued-resources", "create",
            self._queued_resource_id,
            f"--node-id={self.name}",
            f"--zone={self.zone}",
            f"--accelerator-type={self.accelerator}",
            f"--runtime-version={self.version}",
        ]
        if self.pricing == "spot":
            cmd.append("--spot")
        logger.info("Creating queued resource %s for TPU %s...", self._queued_resource_id, self.name)
        _run(cmd, check=True, capture=False)

    def _queued_resource_status(self) -> TPUStatus:
        result = _run([
            "gcloud", "alpha", "compute", "tpus", "queued-resources", "describe",
            self._queued_resource_id,
            f"--zone={self.zone}", "--format=json"
        ], check=False)
        if result.returncode != 0:
            if "NOT_FOUND" in result.stderr or "not found" in result.stderr.lower():
                return "NOT_FOUND"
            return "UNKNOWN"
        try:
            info = json.loads(result.stdout)
            state = info.get("state", {})
            if isinstance(state, dict):
                state = state.get("state", "UNKNOWN")
            qr_state = str(state).upper()
        except (json.JSONDecodeError, KeyError):
            return "UNKNOWN"

        # When the QR is ACTIVE, the TPU VM exists — query it directly for the
        # precise VM state (e.g. READY vs PREEMPTED vs SUSPENDED).
        if qr_state == "ACTIVE":
            return self._tpu_vm_status()

        return _normalize_status(qr_state)

    def _delete_queued_resource(self) -> None:
        logger.info("Deleting queued resource %s...", self._queued_resource_id)
        result = _run([
            "gcloud", "alpha", "compute", "tpus", "queued-resources", "delete",
            self._queued_resource_id,
            f"--zone={self.zone}", "--quiet", "--force"
        ], check=False)
        if result.returncode != 0 and "NOT_FOUND" not in result.stderr:
            logger.warning("Failed to delete queued resource %s: %s",
                           self._queued_resource_id, result.stderr)


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _normalize_status(state: str) -> TPUStatus:
    mapping = {
        "READY": "READY",
        "PROVISIONING": "CREATING",
        "CREATING": "CREATING",
        "WAITING_FOR_RESOURCES": "CREATING",
        "PREEMPTED": "PREEMPTED",
        "TERMINATED": "TERMINATED",
        "SUSPENDING": "SUSPENDED",
        "SUSPENDED": "SUSPENDED",
        "NOT_FOUND": "NOT_FOUND",
    }
    return mapping.get(state, "UNKNOWN")  # type: ignore[return-value]


def _num_workers(accelerator: str) -> int:
    """Infer worker count from accelerator string like v4-8, v5e-16, v6e-32."""
    try:
        parts = accelerator.lower().split("-")
        chips = int(parts[-1])
        gen = parts[0]  # e.g. "v4", "v5e", "v6e"
        if gen == "v4":
            return max(1, chips // 8)
        elif gen in ("v5e", "v6e"):
            return max(1, chips // 4)
        else:
            # Default: assume 8 chips per host
            return max(1, chips // 8)
    except (IndexError, ValueError):
        return 1
