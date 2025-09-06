import re
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent 

def parse_loss_from_log(log_path):
    """
    Parses step and loss from a MaxText log.
    Returns deduplicated step-loss pairs.
    """
    losses = {}
    lrs = {}

    with open(log_path, "r") as f:
        for line in f:
            if line.startswith('completed step:'):
                raw = line.split(', ')
                metrics = {each.split(': ')[0]: float(each.split(': ')[1]) for each in raw}
                step = metrics['completed step']
                loss = metrics['loss']
                lr = metrics['lr']
                losses[step] = loss
                lrs[step] = lr

    steps = sorted(losses.keys())
    losses = [losses[step] for step in steps]
    lrs = [lrs[step] for step in steps]
    return steps, losses, lrs

def plot_losses(filtered_dict, prefix):
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    for label, path in filtered_dict.items():
        steps, losses, lrs = parse_loss_from_log(path)
        if steps:
            axes[0].plot(steps, losses, label=label)
            axes[1].plot(steps, lrs, label=label)
        else:
            print(f"[WARN] No loss found for {label}")

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Loss Curves (prefix: {prefix})")
    axes[0].grid(True)
    axes[0].legend()
    
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("LR")
    axes[1].set_title(f"LR Curves (prefix: {prefix})")
    axes[1].grid(True)
    axes[1].legend()
    
    plt.tight_layout()
    
    save_path = str(BASE_DIR / f"loss_{prefix}.png")
    plt.savefig(save_path)
    print(f"[INFO] Saved loss plot to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="", help="Prefix to filter labels")
    args = parser.parse_args()

    label_to_path = {
        "llama3.1-4b-width-S250": "/n/fs/vision-mix/yx1168/jobman-exp/jobs/yx1168/000024/logs/command_worker_0.log",
        "llama3.1-4b-depth-S250": "/n/fs/vision-mix/yx1168/jobman-exp/jobs/yx1168/000025/logs/command_worker_0.log",
        "llama3.1-8b-L200": "/n/fs/vision-mix/yx1168/jobman-exp/jobs/yx1168/000026/logs/command_worker_0.log",
    }

    filtered = {k: v for k, v in label_to_path.items() if k.startswith(args.prefix)}
    if not filtered:
        print(f"[ERROR] No entries matched prefix '{args.prefix}'")
        return

    plot_losses(filtered, args.prefix)

if __name__ == "__main__":
    main()