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
jobman worker stop my-tpu

# Submit tasks
jobman task submit train.sh
jobman task submit train.sh --name=run-1 --accelerator=v4-8 --zone=us-central2-b

# Monitor
jobman status
jobman task logs <task-id>
jobman task logs <task-id> --follow
jobman worker show my-tpu

# Manage tasks
jobman task pause <task-id>
jobman task requeue <task-id>   # put a failed/paused task back to pending
jobman task delete <task-id>
```

`jobman worker start --startup-script ...` now runs that script as worker bootstrap on all TPU hosts after the TPU becomes ready and before the worker claims any tasks. If bootstrap fails, no task is claimed; the worker retries bootstrap after the TPU is healthy again.

`jobman task pause` and `jobman task delete` now affect running tasks as well: the worker notices the queue-state change, terminates the in-flight SSH command, and then either leaves the task paused or removes it entirely.

## State Directory

State is stored in `/scratch/yx3038/pruning/jobman-lite` by default (override with `$JOBMAN_DIR`).
Logs are stored in `/scratch/yx3038/pruning/jobman-lite/logs` by default (override with `$JOBMAN_LOG_DIR`):

```
JOBMAN_DIR/
├── workers.json          # worker registry
├── queue.json            # task queue
```

```
JOBMAN_LOG_DIR/
├── workers/<tpu>/
│   ├── worker.log
│   ├── bootstrap.log
│   ├── <startup-script>.sh
│   └── timeline.jsonl
└── tasks/<task-id>/
    ├── <submitted-script>.sh
    └── run_*.log
```

Submitted task scripts are copied into `tasks/<task-id>/` under the log directory at submit time.
Workers execute that frozen copy rather than the original source path.
Worker bootstrap output is written under `workers/<tpu>/bootstrap.log`.
The worker's startup script is copied into `workers/<tpu>/` at worker start time.
Worker lifecycle events are appended to `workers/<tpu>/timeline.jsonl`.

## Design

- **Worker bootstrap is worker-scoped** — `--startup-script` runs before task claiming, not per task
- **Worker ID = TPU name** — unique, human-readable
- **Script runs on worker 0** — user uses `gcloud ... --worker=all` for multi-host setup
- **fcntl.flock()** for queue atomicity — no database needed
- **Fault-tolerant** — SSH exit 255 + TPU status check detects preemption; task re-queued automatically
