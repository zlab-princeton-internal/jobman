import subprocess
from tqdm import tqdm
from yaspin import yaspin
from tabulate import tabulate

# List of GCS buckets to profile
def list_buckets():
    try:
        output = subprocess.check_output(
            ["gcloud", "storage", "buckets", "list", "--format=value(name)"]
        ).decode("utf-8")
        return output.strip().splitlines()
    except subprocess.CalledProcessError as e:
        print("Failed to list buckets:", e)
        return []

def get_bucket_size(bucket_name):
    try:
        output = subprocess.check_output(
            ["gsutil", "du", "-s", f"gs://{bucket_name}"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8")
        size_bytes = int(output.strip().split()[0])
        return size_bytes
    except subprocess.CalledProcessError:
        print(f"Error reading bucket: {bucket_name}")
        return 0

def format_size(bytes_val):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} PB"

def main():
    buckets = list_buckets()
    # buckets = [b for b in buckets if b.startswith("llm_pruning")]
    total_size = 0
    rows = []

    sizes = []

    with tqdm(buckets) as pbar:
        for bucket in pbar:
            pbar.set_description(f"processing bucket: {bucket}")
            size = get_bucket_size(bucket)
            total_size += size
            rows.append([bucket, format_size(size)])

    rows.append([None, None])
    rows.append(["TOTAL", format_size(total_size)])

    print(tabulate(rows, headers=["Bucket Name", "Size"], tablefmt="github"))    

if __name__ == "__main__":
    main()