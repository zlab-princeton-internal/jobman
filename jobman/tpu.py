"""TPU lifecycle management for jobman-lite."""

import json
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from typing import Literal

from .utils import get_logger

logger = get_logger(__name__)

TPUStatus = Literal[
    "READY",
    "WAITING_FOR_RESOURCES",
    "PROVISIONING",
    "CREATING",
    "UNHEALTHY",
    "PREEMPTED",
    "TERMINATED",
    "SUSPENDED",
    "FAILED",
    "NOT_FOUND",
    "UNKNOWN",
]
AllocationMode = Literal["tpu-vm", "queued-resources"]
Pricing = Literal["spot", "preemptible", "standard"]

# Default runtime version for TPU VMs
DEFAULT_TPU_VERSION = "tpu-ubuntu2204-base"

_TPU_VERSION_MAP = {
    "v4": "tpu-ubuntu2204-base",
    "v5e": "v2-alpha-tpuv5-lite",
    "v5p": "v2-alpha-tpuv5",
    "v6e": "v2-alpha-tpuv6e",
}


def resolve_tpu_version(accelerator: str) -> str:
    """Return the appropriate TPU runtime version for a given accelerator type."""
    family = accelerator.split("-")[0]  # e.g. "v4" from "v4-8"
    return _TPU_VERSION_MAP.get(family, DEFAULT_TPU_VERSION)


def _format_cmd(cmd: list[str]) -> str:
    return shlex.join(cmd)


def _command_output(result: subprocess.CompletedProcess) -> str:
    parts = []
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if stderr:
        parts.append(f"stderr: {stderr}")
    if stdout:
        parts.append(f"stdout: {stdout}")
    return "\n".join(parts)


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    cmd_str = _format_cmd(cmd)
    logger.debug("Running: %s", cmd_str)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = _command_output(result) or "no stdout/stderr"
        logger.warning("gcloud command failed (%d): %s\n%s", result.returncode, cmd_str, detail)
        if check:
            raise RuntimeError(
                f"Command failed ({result.returncode}): {cmd_str}\n{detail}"
            )
    return result


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
        self._queued_resource_id = f"{self.name}"

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

    def vm_status(self) -> TPUStatus:
        """Return the TPU VM status directly."""
        return self._tpu_vm_status()

    def describe(self) -> dict | None:
        """Return the raw TPU describe payload for the active resource, if available."""
        if self.mode == "queued-resources":
            return self._describe_queued_resource()
        return self._describe_tpu_vm()

    def health_status(self) -> str:
        """Return HEALTHY, UNHEALTHY, or UNKNOWN for the active TPU resource."""
        if self.mode == "queued-resources":
            qr_info = self._describe_queued_resource()
            if not isinstance(qr_info, dict):
                return "UNKNOWN"
            state = qr_info.get("state", {})
            if isinstance(state, dict):
                state = state.get("state", "UNKNOWN")
            if str(state).upper() != "ACTIVE":
                return "UNKNOWN"
            info = self._describe_tpu_vm()
        else:
            info = self._describe_tpu_vm()
        if not isinstance(info, dict):
            return "UNKNOWN"
        return _normalize_health_status(info)

    def health_description(self) -> str:
        """Return a human-readable health summary, if present."""
        if self.mode == "queued-resources":
            qr_info = self._describe_queued_resource()
            if not isinstance(qr_info, dict):
                return ""
            state = qr_info.get("state", {})
            if isinstance(state, dict):
                state = state.get("state", "UNKNOWN")
            if str(state).upper() != "ACTIVE":
                return ""
            info = self._describe_tpu_vm()
        else:
            info = self._describe_tpu_vm()
        if not isinstance(info, dict):
            return ""
        description = info.get("healthDescription")
        if isinstance(description, str) and description.strip():
            return description.strip()
        symptoms = info.get("symptoms")
        if isinstance(symptoms, list):
            details: list[str] = []
            for symptom in symptoms:
                if not isinstance(symptom, dict):
                    continue
                symptom_type = str(symptom.get("symptomType", "")).strip()
                symptom_details = str(symptom.get("details", "")).strip()
                text = ", ".join(part for part in (symptom_type, symptom_details) if part)
                if text:
                    details.append(text)
            return "; ".join(details)
        return ""

    def queued_resource_status(self) -> str:
        """Return the queued resource state directly (e.g. ACTIVE, FAILED, NOT_FOUND)."""
        if self.mode != "queued-resources":
            return "-"
        result = _run([
            "gcloud", "compute", "tpus", "queued-resources", "describe",
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
            return str(state).upper()
        except (json.JSONDecodeError, KeyError):
            return "UNKNOWN"

    def delete(self) -> None:
        """Delete the TPU VM (and queued resource if applicable)."""
        if self.mode == "queued-resources":
            self._delete_queued_resource()
        self._delete_tpu_vm()

    def wait_ready(
        self,
        timeout: int = 86400,
        poll_interval: int = 30,
        status_callback=None,
    ) -> None:
        """Poll until READY or timeout. Uses exponential backoff up to poll_interval."""
        logger.info("Waiting for TPU %s to be READY (timeout=%ds)...", self.name, timeout)
        start = time.time()
        interval = 10
        last_reported: TPUStatus | None = None
        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                raise TimeoutError(f"TPU {self.name} not ready after {timeout}s")
            st = self.status()
            if st == "READY":
                logger.info("TPU %s is READY (%.0fs elapsed)", self.name, elapsed)
                return
            if st == "UNHEALTHY":
                raise RuntimeError(
                    f"TPU {self.name} entered unhealthy state: "
                    f"{self.health_description() or 'unhealthy'}"
                )
            if st in ("PREEMPTED", "TERMINATED", "FAILED", "SUSPENDED"):
                raise RuntimeError(f"TPU {self.name} entered terminal state: {st}")
            if status_callback is not None and st != last_reported:
                status_callback(st)
                last_reported = st
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
        _run(cmd, check=True)

    def _tpu_vm_status(self) -> TPUStatus:
        info = self._describe_tpu_vm()
        if info == "NOT_FOUND":
            return "NOT_FOUND"
        if not isinstance(info, dict):
            return "UNKNOWN"
        state = str(info.get("state", "UNKNOWN")).upper()
        normalized = _normalize_status(state)
        if _normalize_health_status(info) == "UNHEALTHY":
            return "UNHEALTHY"
        return normalized

    def _delete_tpu_vm(self) -> None:
        logger.info("Deleting TPU VM %s...", self.name)
        result = _run([
            "gcloud", "compute", "tpus", "tpu-vm", "delete", self.name,
            f"--zone={self.zone}", "--quiet"
        ], check=False)
        stderr = result.stderr or ""
        if result.returncode != 0 and "NOT_FOUND" not in stderr and "not found" not in stderr.lower():
            logger.warning("Failed to delete TPU VM %s", self.name)

    # ------------------------------------------------------------------
    # Queued-resources mode helpers
    # ------------------------------------------------------------------

    def _create_queued_resource(self) -> None:
        cmd = [
            "gcloud", "compute", "tpus", "queued-resources", "create",
            self._queued_resource_id,
            f"--node-id={self.name}",
            f"--zone={self.zone}",
            f"--accelerator-type={self.accelerator}",
            f"--runtime-version={self.version}",
        ]
        if self.pricing == "spot":
            cmd.append("--spot")
        logger.info("Creating queued resource %s for TPU %s...", self._queued_resource_id, self.name)
        _run(cmd, check=True)

    def _queued_resource_status(self) -> TPUStatus:
        info = self._describe_queued_resource()
        if info == "NOT_FOUND":
            return "NOT_FOUND"
        if not isinstance(info, dict):
            return "UNKNOWN"
        try:
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

    def _delete_queued_resource(self, max_retries: int = 3) -> None:
        logger.info("Deleting queued resource %s...", self._queued_resource_id)
        for attempt in range(1, max_retries + 1):
            result = _run([
                "gcloud", "alpha", "compute", "tpus", "queued-resources", "delete",
                self._queued_resource_id,
                f"--zone={self.zone}", "--quiet", "--force"
            ], check=False)
            stderr = result.stderr or ""
            if result.returncode == 0:
                return
            if "NOT_FOUND" in stderr or "not found" in stderr.lower():
                return
            logger.warning(
                "Failed to delete queued resource %s (attempt %d/%d)",
                self._queued_resource_id, attempt, max_retries,
            )
            if attempt < max_retries:
                time.sleep(5 * attempt)

    def _describe_tpu_vm(self) -> dict | None:
        result = _run([
            "gcloud", "compute", "tpus", "tpu-vm", "describe", self.name,
            f"--zone={self.zone}", "--format=json"
        ], check=False)
        if result.returncode != 0:
            if "NOT_FOUND" in result.stderr or "not found" in result.stderr.lower():
                return "NOT_FOUND"
            return None
        try:
            payload = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _describe_queued_resource(self) -> dict | None:
        result = _run([
            "gcloud", "compute", "tpus", "queued-resources", "describe",
            self._queued_resource_id,
            f"--zone={self.zone}", "--format=json"
        ], check=False)
        if result.returncode != 0:
            if "NOT_FOUND" in result.stderr or "not found" in result.stderr.lower():
                return "NOT_FOUND"
            return None
        try:
            payload = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            return None
        return payload if isinstance(payload, dict) else None


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _normalize_status(state: str) -> TPUStatus:
    mapping = {
        "READY": "READY",
        "PROVISIONING": "PROVISIONING",
        "CREATING": "CREATING",
        "UNHEALTHY": "UNHEALTHY",
        "WAITING_FOR_RESOURCES": "WAITING_FOR_RESOURCES",
        "PREEMPTED": "PREEMPTED",
        "TERMINATED": "TERMINATED",
        "SUSPENDING": "SUSPENDED",
        "SUSPENDED": "SUSPENDED",
        "FAILED": "FAILED",
        "NOT_FOUND": "NOT_FOUND",
    }
    return mapping.get(state, "UNKNOWN")  # type: ignore[return-value]


def _normalize_health_status(info: dict) -> str:
    health = str(info.get("health", "") or "").upper()
    description = str(info.get("healthDescription", "") or "").lower()
    symptoms = info.get("symptoms", [])

    if health.startswith("UNHEALTHY"):
        return "UNHEALTHY"
    if health == "HEALTHY":
        return "HEALTHY"

    if "maintenance event" in description or "out of memory" in description or "oom" in description:
        return "UNHEALTHY"

    if isinstance(symptoms, list):
        for symptom in symptoms:
            if not isinstance(symptom, dict):
                continue
            symptom_type = str(symptom.get("symptomType", "") or "").upper()
            symptom_details = str(symptom.get("details", "") or "").lower()
            if symptom_type.startswith("OUT_OF_MEMORY"):
                return "UNHEALTHY"
            if "maintenance event" in symptom_details or "out of memory" in symptom_details or "oom" in symptom_details:
                return "UNHEALTHY"

    if health:
        return "UNHEALTHY"
    return "UNKNOWN"


def _num_workers(accelerator: str) -> int:
    """Infer worker count from accelerator string like v4-8, v5e-16, v6e-32."""
    try:
        parts = accelerator.lower().split("-")
        chips = int(parts[-1])
        gen = parts[0]  # e.g. "v4", "v5e", "v6e"
        if gen in ("v4", "v5p"):
            return max(1, chips // 8)
        elif gen in ("v5e", "v6e"):
            return max(1, chips // 4)
        else:
            # Default: assume 8 chips per host
            return max(1, chips // 8)
    except (IndexError, ValueError):
        return 1
