# Jobman-lite

Lightweight TPU job orchestration. Workers hold TPUs persistently and poll a shared queue for tasks.

## Install

```bash
conda create -n jobman-lite python=3.12 -y
conda activate jobman-lite
pip install -e .
```

## Script Header Format

```bash
#!/bin/bash
#JOBMAN --accelerator=v4-8
#JOBMAN --zone=us-central2-b
#JOBMAN --name=my-task          # optional
#JOBMAN --tpu-version=tpu-ubuntu2204-base  # optional
#JOBMAN --max-retries=3         # optional

# Injected env vars: JOBMAN_TPU_NAME, JOBMAN_ZONE, JOBMAN_NUM_WORKERS

# Multi-host setup (user handles this):
gcloud compute tpus tpu-vm ssh $JOBMAN_TPU_NAME --zone=$JOBMAN_ZONE \
  --worker=all -- 'conda activate myenv && pip install -r requirements.txt'

# Main training (runs on worker 0, coordinates multi-host via JAX):
python train.py
```

## Usage

```bash
# Start a worker (holds the TPU, polls for tasks)
jobman worker start --accelerator=v4-8 --zone=us-central2-b
jobman worker list
jobman worker stop my-tpu

# Submit tasks
jobman submit train.sh
jobman submit train.sh --name=run-1 --accelerator=v4-8 --zone=us-central2-b

# Monitor
jobman status
jobman logs <task-id>
jobman logs <task-id> --follow

# Manage tasks
jobman cancel <task-id>
jobman reset <task-id>   # re-queue a failed/cancelled task
```

## State Directory

State is stored in `~/.jobman/` (override with `$JOBMAN_DIR`):

```
~/.jobman/
├── workers.json          # worker registry
├── queue.json            # task queue
└── logs/
    ├── workers/<tpu>/worker.log
    └── tasks/<task-id>/task.log
```

## Design

- **No modules** — users handle all env setup inside their script
- **Worker ID = TPU name** — unique, human-readable
- **Script runs on worker 0** — user uses `gcloud ... --worker=all` for multi-host setup
- **fcntl.flock()** for queue atomicity — no database needed
- **Fault-tolerant** — SSH exit 255 + TPU status check detects preemption; task re-queued automatically
