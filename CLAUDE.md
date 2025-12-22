# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Jobman is a modular TPU job orchestration system for managing multi-host training jobs on Google Cloud TPUs. Jobs live as tmux sessions managed by a central `jobman` process, with state stored in `jobs/.jobman/`.

## Installation

```bash
pip install -e .
```

Prerequisites: `gcloud` CLI with alpha/beta components, tmux, authenticated via `gcloud auth login`.

## Common Commands

### Job Management
```bash
jobman create <config.yaml>  # Create and start job from config
jobman list                  # List all jobs with status
jobman run <job_id>          # Run job in foreground (debugging)
jobman run <job_id> --force  # Force re-run setup steps
jobman ssh <job_id>          # SSH into job's TPU VM (worker 0)
jobman resume <job_id>       # Resume stopped job
jobman stop <job_id>         # Kill tmux session
jobman reboot <job_id>       # Stop and restart job
jobman delete <job_id>       # Stop job and delete TPU
jobman clean <job_id>        # Delete job, TPU, and logs
```

### Profiling
```bash
jobman quota    # View TPU quota
jobman storage  # View GCS storage usage
```

### Running Tests
```bash
pytest tests/
```

## Architecture

### Core Components

**jobman/jobman.py** - Central orchestrator (`JobMan` class):
- Manages job metadata in `jobs/.jobman/meta.json` with file locking
- Creates/starts/stops tmux sessions for each job
- Validates bucket region matches TPU zone
- Infers worker count from accelerator type (e.g., v4-256 -> 32 workers)

**jobman/job.py** - Job execution (`Job` class):
- Lifecycle: `request()` -> `setup()` -> `execute()`
- Coordinates TPU, SSH, GCSFuse, and environment modules
- Supports `loop: true` for auto-restart on preemption

**jobman/runner.py** - Multi-worker orchestration (`MultiWorkerRunner`):
- Base class for parallel operations across TPU workers
- Provides `_ssh()` and `_scp()` helpers for gcloud compute SSH/SCP
- Check-then-setup pattern: skips setup if checks pass (unless `--force`)

**jobman/tpu.py** - TPU lifecycle management (`TPU` class):
- Supports `tpu-vm` and `queued-resources` allocation modes
- Handles states: READY, ACTIVE, PREEMPTED, TERMINATED, etc.
- Auto-deletes unhealthy TPUs and requeues

**jobman/cli.py** - Click-based CLI entry point

### Module Inheritance Pattern

SSH, GCSFuse, CONDA, VENV, DOCKER, and COMMAND all extend `MultiWorkerRunner`:
- Override `_get_check_steps(worker_id)` for idempotency checks
- Override `_get_setup_steps(worker_id)` for installation/config
- `setup(force=False)` runs checks first, skips if all pass

## Configuration Structure (YAML)

```yaml
job:
  name: my-job
  env_type: docker|conda|venv  # Environment type
  loop: true|false             # Auto-restart on exit
  remote_user: username

tpu:
  allocation_mode: queued-resources|tpu-vm
  accelerator: v4-32           # Determines worker count
  name: tpu-name
  zone: us-central2-b
  version: tpu-ubuntu2204-base
  pricing: spot|ondemand|preemptible

gcsfuse:
  bucket_name: my-bucket       # Must be in same region as TPU zone
  mount_path: /home/user/gcs

ssh:
  private_key: ~/.ssh/key
  identities:                  # Keys to copy to TPU VMs
    - private_key: ~/.ssh/key
      public_key: ~/.ssh/key.pub
      config_entry: |          # SSH config to install
        Host 10.*
          IdentityFile ~/.ssh/key

docker|conda|venv:             # Environment-specific config
  # See GET_STARTED.md for details

command:
  cmd: |                       # Shell script to execute
    echo "hello"
  workers: [0]|"all"|[0,1,2]   # Which workers run command
```

## Job States

- **QUEUEING**: tmux session running, TPU not yet ready
- **RUNNING**: tmux session running, TPU ready
- **IDLE**: no tmux session, TPU still exists
- **DEAD**: no tmux session, no TPU

## Debugging

1. If `jobman create` appears to hang, run `jobman run <job_id>` to see errors in foreground
2. Logs are at `jobs/<user>/<job_id>/logs/job.log`
3. Per-worker logs: `jobs/<user>/<job_id>/logs/<action>_worker_<N>.log`
4. Enable debug logging: `export JOBMAN_DEBUG=1`
5. Check TPU status on Google Cloud Console to verify job state consistency

## Queue Mode

Queue mode allows multiple jobs to pick tasks from a shared queue instead of running fixed scripts. Useful for managing many experiments across preemptible TPUs.

### Queue Commands
```bash
jobman queue queue_machines/tasks.yaml              # View queue status
jobman queue queue_machines/tasks.yaml --reset <id> # Reset a specific task
jobman queue queue_machines/tasks.yaml --reset-all  # Reset all tasks
```

### Queue Config (`queue_machines/tasks.yaml`)
```yaml
repos:
  maxtext: https://github.com/user/repo  # Short aliases for repos

defaults:
  branch: main
  workdir: maxtext  # Clone destination (~/<workdir>)

tasks:
  - id: pretrain-8b-s42
    accelerator: v6e-128       # Must match job's TPU type
    env: maxtext_env           # venv/conda name
    repo: maxtext              # Key from repos above
    run: bash train/run.sh     # Command to run (from workdir)
    max_jobs: 1                # Max concurrent executions
```

### Job Config for Queue Mode (`queue_machines/worker_*.yaml`)
```yaml
job:
  queue:
    enabled: true
    config: queue_machines/tasks.yaml
  # ... rest of job config (tpu, ssh, venv, etc.)
```

### How It Works
1. Job starts, requests TPU, runs setup (SSH, gcsfuse, env) as normal
2. Looks for first unclaimed task matching its accelerator
3. Deletes old repo, clones fresh, runs task command
4. On completion: marks done, picks next task
5. On preemption: releases task (stays not-done), restarts if `loop: true`

### Switching Modes
To switch a job from queue mode to fixed script:
1. Edit job config: set `queue.enabled: false` or remove `queue` section
2. Add/modify `command` section with your script
3. Run `jobman reboot <job_id>`

## Key Implementation Details

- Worker count inference: v4 chips/8, v5e chips/4, v6e-8 is 1 worker, larger v6e chips/4
- Bucket region validation prevents cross-region egress costs
- SSH identity files are SCP'd to each worker's `~/.ssh/`
- Commands run via `bash -lc` through gcloud SSH
- Queue state stored in `<queue_config>_state.yaml` with file locking
