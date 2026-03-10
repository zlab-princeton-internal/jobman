# jobman-lite

Lightweight TPU job orchestration. Workers hold TPUs persistently and poll a shared file-based queue for tasks.

## Install

```bash
conda create -n jobman-lite python=3.12 -y
conda activate jobman-lite
pip install -e .
```

## Project Structure

```
jobman/
├── cli.py       # Click CLI entry point (jobman command)
├── worker.py    # Worker loop: TPU holder + task executor
├── tpu.py       # TPU lifecycle via gcloud (tpu-vm and queued-resources modes)
├── queue.py     # File-based task queue with fcntl locking
├── utils.py     # Logger setup, jobman_dir()
└── __main__.py  # python -m jobman entry point
```

## State Directory

`/scratch/yx3038/pruning/jobman-lite/` (override with `$JOBMAN_DIR`):
- `workers.json` — worker registry keyed by TPU name
- `queue.json` — task queue keyed by task ID
- `worker_counter` — monotonically increasing int for auto-naming workers
- `logs/workers/<tpu_name>/worker.log`
- `logs/tasks/<task_id>/task.log`

## Key Design Decisions

- **Worker ID = TPU name** — no separate ID counter; unique and human-readable
- **Auto-generated TPU names**: `{accelerator}-{zone}-{count:05d}`, e.g. `v4-8-us-central2-b-00001`. Counter is global (shared across all accelerator/zone combos) and stored in `worker_counter`.
- **fcntl.flock()** for queue atomicity — no database needed
- **Script runs on worker 0** — user uses `gcloud ... --worker=all` inside their script for multi-host setup; JAX coordinates multi-host internally
- **Preemption detection**: SSH exit code 255 + `tpu.status()` check → task re-queued as `interrupted`
- **Retry logic**: `fail_count` tracks failures; task permanently fails after `max_retries`; preemption (`interrupted`) does not increment `fail_count`

## TPU Allocation Modes

- `tpu-vm` (default): `gcloud alpha compute tpus tpu-vm create`
- `queued-resources`: `gcloud alpha compute tpus queued-resources create`; queued resource ID = `qr-{tpu_name}`

## Worker Count Inference (`tpu.py:_num_workers`)

- v4: chips / 8
- v5e, v6e: chips / 4
- others: chips / 8

## Task States

`pending` → `running` → `done` / `failed`
`running` → `interrupted` → `pending` (preemption, no fail_count increment)
`failed` (if fail_count < max_retries) → `pending` (auto-retry)
`pending`/`running` → `cancelled` (manual)
any → `pending` (manual reset via `jobman reset`)

## CLI Commands

```bash
jobman worker start --accelerator=v4-8 --zone=us-central2-b   # --tpu-name optional
jobman worker stop <tpu-name>
jobman worker list

jobman submit train.sh [--name=...] [--accelerator=...] [--zone=...] [--max-retries=N]
jobman cancel <task-id>
jobman reset <task-id>
jobman status
jobman logs <task-id> [-f] [-n 50]
```

## Script Header Format

```bash
#!/bin/bash
#JOBMAN --accelerator=v4-8
#JOBMAN --zone=us-central2-b
#JOBMAN --name=my-task        # optional
#JOBMAN --tpu-version=tpu-ubuntu2204-base  # optional
#JOBMAN --max-retries=3       # optional

# Injected env vars: JOBMAN_TPU_NAME, JOBMAN_ZONE, JOBMAN_NUM_WORKERS
```