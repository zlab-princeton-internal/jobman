# Queue Mode

Queue mode allows multiple TPU jobs to pick tasks from a shared queue.

## Quick Start

```bash
# 1. Edit tasks (just add id + run under the right accelerator)
vim queue_machines/tasks.yaml

# 2. Create workers
jobman create queue_machines/worker_v6e128_us_central1b.yaml

# 3. Monitor
jobman queue queue_machines/tasks.yaml
```

## Files

| File | Description |
|------|-------------|
| `tasks.yaml` | Task queue - edit to add/remove tasks |
| `tasks_state.yaml` | **Auto-generated** - tracks done/running/history |
| `worker_*.yaml` | Worker configs (one per accelerator+zone combo) |

## Task Format (tasks.yaml)

Tasks are grouped by accelerator. You only need to add `id` + `run`:

```yaml
repos:
  maxtext: https://github.com/TaiMingLu/maxtext-distillation

defaults:
  env: maxtext_env
  repo: maxtext
  branch: main
  workdir: maxtext
  max_jobs: 1

accelerators:
  v6e-128:
    tasks:
      - id: pretrain-8b-s42
        run: bash train/pretrain_v6/llama8b-finewebedu-vanilla-s42.sh

      - id: pretrain-8b-s43
        run: bash train/pretrain_v6/llama8b-finewebedu-vanilla-s43.sh

  v6e-64:
    tasks:
      - id: pretrain-3b-s42
        run: bash train/pretrain_v6/llama3b-finewebedu-vanilla-s42.sh
```

## Worker Config (worker_*.yaml)

The `prefix` runs before each task (env vars, gcloud config, ssh keys):

```yaml
job:
  queue:
    enabled: true
    config: queue_machines/tasks.yaml
    prefix: |
      export BUCKET_NAME=taiming_us_central1
      gcloud config set project vision-mix
      rm -f ~/.ssh/google_compute_engine ~/.ssh/google_compute_engine.pub
      ssh-keygen -t rsa -f ~/.ssh/google_compute_engine -C "$USER" -N "" -q
      gcloud compute os-login ssh-keys add --key-file ~/.ssh/google_compute_engine.pub --quiet
```

## State File (tasks_state.yaml)

Auto-generated. Tracks for each task:
- `done`: true/false
- `running`: list of {job_id, started} currently running
- `run_count`: total attempts
- `history`: list of {job_id, started, ended, status}

View with: `jobman queue queue_machines/tasks.yaml`

## Commands

```bash
# View status
jobman queue queue_machines/tasks.yaml

# Reset one task
jobman queue queue_machines/tasks.yaml --reset pretrain-8b-s42

# Reset all tasks
jobman queue queue_machines/tasks.yaml --reset-all
```

## Switching Modes

To run a specific script instead of queue:
1. Set `queue.enabled: false` in worker yaml
2. Add a `command:` section
3. `jobman reboot <job_id>`
