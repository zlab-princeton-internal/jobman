#!/usr/bin/env python3
"""Count queued TPU resources by status in one or more zones.

Also sums running TPU cores from queued resources.
Example: two queues with accelerator type v4-128 -> running_cores = 256.
"""

import pdb
import argparse
import json
import subprocess
from collections import Counter
import re


WAITING_STATES = {
    "WAITING_FOR_RESOURCES",
    "ACCEPTED",
}

PROVISIONING_STATES = {
    "CREATING",
    "PROVISIONING",
    "STATE_UNSPECIFIED",
}

RUNNING_STATES = {
    "READY", 
    "ACTIVE"
}


def _run_json(cmd):
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr.strip()}")
    stdout = result.stdout.strip()
    if not stdout:
        return []
    return json.loads(stdout)


def _queue_state(item):
    # list output commonly exposes state as either:
    # - "state": "ACTIVE"
    # - "state": {"state": "ACTIVE", ...}
    state = item.get("state")
    if isinstance(state, dict):
        state = state.get("state")
    if not state:
        state = item.get("queuedResourceState", {}).get("state")
    return str(state or "").strip().upper()


def _extract_accelerator_type(item):
    """Best-effort extraction of accelerator type from queued resource JSON."""
    # Common shape from gcloud list output
    if item.get("acceleratorType"):
        return str(item["acceleratorType"])

    # Fallbacks seen in nested representations
    tpu = item.get("tpu") or {}
    node_spec = tpu.get("nodeSpec")
    if isinstance(node_spec, dict):
        if node_spec.get("acceleratorType"):
            return str(node_spec["acceleratorType"])
        if isinstance(node_spec.get("node"), dict) and node_spec["node"].get("acceleratorType"):
            return str(node_spec["node"]["acceleratorType"])

    if isinstance(node_spec, list) and node_spec:
        first = node_spec[0] if isinstance(node_spec[0], dict) else {}
        if isinstance(first.get("node"), dict) and first["node"].get("acceleratorType"):
            return str(first["node"]["acceleratorType"])
        if first.get("acceleratorType"):
            return str(first["acceleratorType"])

    node_specs = tpu.get("nodeSpecs") or []
    if node_specs and isinstance(node_specs, list):
        first = node_specs[0] if isinstance(node_specs[0], dict) else {}
        if first.get("node") and isinstance(first["node"], dict):
            node = first["node"]
            if node.get("acceleratorType"):
                return str(node["acceleratorType"])
        if first.get("acceleratorType"):
            return str(first["acceleratorType"])

    return ""


def _accelerator_to_cores(accelerator_type: str) -> int:
    """Parse cores from accelerator string like v4-128, v5e-16, v6e-4."""
    m = re.search(r"-(\d+)$", accelerator_type)
    return int(m.group(1)) if m else 0


def count_zone(zone: str):
    counts = Counter(waiting_for_resources=0, provisioning=0, running=0, running_cores=0)

    # Queued resources
    queues = _run_json([
        "gcloud",
        "compute",
        "tpus",
        "queued-resources",
        "list",
        "--zone",
        zone,
        "--format=json",
    ])
    for qr in queues:
        # pdb.set_trace()
        if 'yufeng' not in qr['name']:
            continue
        state = _queue_state(qr)
        acc = _extract_accelerator_type(qr)
        cores =_accelerator_to_cores(acc)
        if state in WAITING_STATES:
            counts["waiting_for_resources"] += cores
        elif state in PROVISIONING_STATES:
            counts["provisioning"] += cores
        elif state in RUNNING_STATES:
            counts["running"] += cores
        elif state == "SUSPENDED":
            counts["suspended"] += cores
        elif state == "FAILED":
            counts["failed"] += cores
        else:
            continue
        
    counts["total_cores"] = counts["waiting_for_resources"] + counts["provisioning"] + counts["running"]

    return counts


def main():
    total = Counter(waiting_for_resources=0, provisioning=0, running=0, running_cores=0)
    for zone in ["us-central2-b", "us-central1-b", "us-east5-b"]:
        counts = count_zone(zone)
        total.update(counts)
        print(f"zone: {zone}")
        print(f"waiting_for_resources: {counts['waiting_for_resources']}")
        print(f"provisioning: {counts['provisioning']}")
        print(f"running: {counts['running']}")
        print(f"suspended: {counts['suspended']}")
        print(f"failed: {counts['failed']}")
        print(f"total_cores: {counts['total_cores']}")
        print()

if __name__ == "__main__":
    main()
