import json
import subprocess
from tabulate import tabulate
from googleapiclient import discovery
from google.auth import default

credentials, _ = default()
compute = discovery.build("compute", "v1", credentials=credentials)

PROJECT_ID = "vision-mix"

ZONAL_QUOTA = {
    "us-east1-d": {
        "v6e-preemptible": 64
    },
    "us-central1-a": {
        "v5litepod-preemptible": 64
    },
    "us-central2-b": {
        "v4-preemptible": 512,
        "v4-ondemand": 64
    },
    "europe-west4-a": {
        "v6e-preemptible": 64
    },
    "europe-west4-b": {
        "v5litepod-preemptible": 64
    },
}

REGIONS = sorted(set(zone.rsplit("-", 1)[0] for zone in ZONAL_QUOTA))

def get_tpu_usage_by_type(zone):
    """Return a dict of {(tpu_type, schedule): total_cores} used in a given zone."""
    usage = {}
    try:
        result = subprocess.run(
            [
                "gcloud", "alpha", "compute", "tpus", "tpu-vm", "list",
                f"--project={PROJECT_ID}",
                f"--zone={zone}",
                "--format=json"
            ],
            capture_output=True,
            check=True,
        )
        tpus = json.loads(result.stdout)
        for tpu in tpus:
            acc_type = tpu.get("acceleratorType", "")  # e.g., "v4-256"
            scheduling = tpu.get("schedulingConfig", {})
            is_preemptible = scheduling.get("preemptible", False) or scheduling.get("spot", False)

            try:
                tpu_type, chips_str = acc_type.split("-")
                chips = int(chips_str)
                schedule = "preemptible" if is_preemptible else "ondemand"
                key = (tpu_type, schedule)
                usage[key] = usage.get(key, 0) + chips
            except Exception:
                continue
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Querying zone {zone}: {e.stderr.decode().strip()}")
    return usage

def get_quota_usage(metric: str):
    results = {}
    for region in REGIONS:
        try:
            res = compute.regions().get(project=PROJECT_ID, region=region).execute()
            for q in res.get("quotas", []):
                if q["metric"] == metric:
                    results[region] = {"used": int(q["usage"]), "limit": int(q["limit"])}
        except Exception as e:
            print(f"[ERROR] Failed to get quota {metric} for {region}: {e}")
    return results

def main():
    tpu_rows = []
    for zone, quota_dict in ZONAL_QUOTA.items():
        usage = get_tpu_usage_by_type(zone)
        for tpu_sched_key, quota in quota_dict.items():
            try:
                tpu_type, sched = tpu_sched_key.split("-")
                used = usage.get((tpu_type, sched), 0)
                tpu_rows.append([tpu_type, sched, used, quota, zone])
            except:
                continue

    print("====== TPU Core Quota Usage ======\n")
    print(tabulate(
        tpu_rows,
        headers=["TPU Type", "Schedule", "Used", "Quota", "Zone"],
        tablefmt="github"
    ))

    print("\n====== Regional IP Quota Usage ======\n")
    print(tabulate(
        [
            [region, v["used"], v["limit"]]
            for region, v in sorted(get_quota_usage('IN_USE_ADDRESSES').items())
        ],
        headers=["Region", "Used", "Quota"],
        tablefmt="github"
    ))
    
    print("\n====== Regional VM Instances Usage ======\n")
    print(tabulate(
        [
            [region, v["used"], v["limit"]]
            for region, v in sorted(get_quota_usage('INSTANCES').items())
        ],
        headers=["Region", "Used", "Quota"],
        tablefmt="github"
    ))
    
    print("\n====== Regional Persistent Disk Usage ======\n")
    print(tabulate(
        [
            [region, v["used"], v["limit"]]
            for region, v in sorted(get_quota_usage('DISKS_TOTAL_GB').items())
        ],
        headers=["Region", "Used", "Quota"],
        tablefmt="github"
    ))

if __name__ == "__main__":
    main()