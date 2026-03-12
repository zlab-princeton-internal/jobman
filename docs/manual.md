# Jobman-lite Manual

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [GCP Auth](#gcp-auth)
- [Mental Model](#mental-model)
- [Script Headers](#script-headers)
- [Injected Environment Variables](#injected-environment-variables)
- [TPU Allocation Modes](#tpu-allocation-modes)
- [Common Commands](#common-commands)
  - [Worker Management](#worker-management)
  - [Task Management](#task-management)
- [Status](#status)
- [Process States](#process-states)
- [Task States](#task-states)
- [Bootstrap Scripts](#bootstrap-scripts)
- [Logging](#logging)
- [Email Notifications](#email-notifications)
- [Failure Handling](#failure-handling)
- [State Layout](#state-layout)
- [Typical Workflow](#typical-workflow)
- [Troubleshooting](#troubleshooting)

## Overview

`jobman-lite` is a lightweight TPU job orchestration tool.

It has two core ideas:
- workers keep TPU capacity alive and poll a shared file-based queue
- tasks are frozen shell scripts with `#JOBMAN` headers that tell workers what TPU shape they need

The control plane is local and file-based:
- worker registry: `workers.json`
- task queue: `queue.json`
- logs: `logs/workers/...` and `logs/tasks/...`

## Prerequisites

Required on the machine where you run `jobman`:
- `tmux`
- Google Cloud SDK (`gcloud`)
- Python 3.12

Required in GCP:
- TPU APIs enabled
- quota in the target zone
- permissions to create, describe, and delete TPU VMs or queued resources

Install:

```bash
conda create -n jobman-lite python=3.12 -y
conda activate jobman-lite
pip install -e .
```

Basic checks:

```bash
tmux -V
gcloud --version
gcloud auth list
gcloud config get-value project
```

## GCP Auth

Authenticate and select the correct project before starting workers:

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project <your-gcp-project>
gcloud config set compute/zone <default-zone>
```

Useful verification:

```bash
gcloud auth list
gcloud config list
gcloud services list --enabled | grep tpu
```

If TPU creation fails, check:
- active account
- selected project
- TPU quota in the target zone
- IAM permissions

## Mental Model

### Workers

A worker is a persistent local process, normally run inside a tmux session.

The worker:
- ensures its TPU exists and is healthy
- optionally runs a worker bootstrap script on all TPU hosts
- claims matching tasks from the queue
- runs each task on worker 0
- requeues or finalizes tasks based on outcome

Worker ID equals TPU name.

### Tasks

A task is a submitted shell script plus metadata extracted from `#JOBMAN` headers and CLI overrides.

At submit time:
- the script is copied into `logs/tasks/<task-id>/`
- the queue entry points to that frozen copy

Workers execute the frozen snapshot, not the original source file.

## Script Headers

Task scripts are shell scripts with `#JOBMAN` headers near the top.

Example:

```bash
#!/usr/bin/env bash
#JOBMAN --accelerator=v4-8
#JOBMAN --zone=us-central2-b
#JOBMAN --name=my-task
#JOBMAN --tpu-version=tpu-ubuntu2204-base
#JOBMAN --max-retries=3
#JOBMAN --mail-user=you@example.com
#JOBMAN --mail-type=BEGIN,END,FAIL

set -euo pipefail
```

Supported task headers:
- `--accelerator`
- `--zone`
- `--name`
- `--tpu-version`
- `--max-retries`
- `--pricing`
- `--mail-user`
- `--mail-type`

`jobman task submit` can still override TPU-related fields like accelerator and zone, but mail settings are header-driven.

## Injected Environment Variables

When a task runs, jobman injects:
- `JOBMAN_TPU_NAME`
- `JOBMAN_ZONE`
- `JOBMAN_NUM_WORKERS`

Use these inside the task script to fan out setup across hosts with `gcloud ... --worker=all`.

## TPU Allocation Modes

Worker allocation supports:
- `tpu-vm`
- `queued-resources`

Default worker start mode is defined by the CLI. Internally, worker status surfaces both:
- VM status
- queued-resource status

## Common Commands

### Worker Management

Start a worker:

```bash
jobman worker start --accelerator=v4-8 --zone=us-central2-b
```

Start with bootstrap:

```bash
jobman worker start \
  --accelerator=v6e-128 \
  --zone=us-east5-b \
  --startup-script=scripts/setup/maxtext_bootstrap.sh
```

Stop:

```bash
jobman worker stop my-worker
jobman worker stop --all
jobman worker stop -a v4-16 -z us-central2-b
```

Resume:

```bash
jobman worker resume my-worker
jobman worker resume -a v4-16
```

Reboot:

```bash
jobman worker reboot my-worker
jobman worker reboot --all
```

Delete:

```bash
jobman worker delete my-worker
jobman worker delete --all
```

Delete releases that worker's running tasks and deletes TPUs in parallel.

Inspect:

```bash
jobman worker show my-worker
jobman worker logs my-worker -f
jobman worker ssh my-worker
```

### Task Management

Submit:

```bash
jobman task submit train.sh
jobman task submit train.sh --name=run-1 --accelerator=v4-8 --zone=us-central2-b
```

Show:

```bash
jobman task show <task-id>
```

Pause:

```bash
jobman task pause <task-id>
jobman task pause --all
jobman task pause -p 'llama3*s50*3e-4*'
```

Requeue:

```bash
jobman task requeue <task-id>
jobman task requeue -a v4-128 -z us-central2-b
```

Delete:

```bash
jobman task delete <task-id>
jobman task delete --all
```

### Status

```bash
jobman status
jobman status --workers-only
jobman status --task-only
jobman status --live-only
```

`status` shows:
- worker process state
- TPU VM state
- queued-resource state
- task queue state

## Process States

Worker process states:
- `running`: tmux session exists
- `stopped`: registry says intentionally stopped
- `dead`: registry says running, but tmux session is gone

Important:
- `dead` means the local worker process is gone
- it does not necessarily mean the TPU is gone

## Task States

Typical task lifecycle:

```text
pending -> running -> done
pending -> running -> failed -> pending   (retry)
pending -> running -> failed              (after max retries)
running -> interrupted -> pending         (preemption / TPU loss)
pending|running -> paused
pending|running -> deleted
```

## Bootstrap Scripts

`--startup-script` is worker-scoped, not task-scoped.

Behavior:
- runs after TPU becomes healthy
- runs on all TPU hosts
- must succeed before task claiming starts
- if it fails, the worker retries later

Bootstrap logs live at:
- `logs/workers/<worker>/bootstrap.log`

Each worker host completion/failure is logged as it happens.

## Logging

Worker logs:
- `logs/workers/<worker>/worker.log`
- `logs/workers/<worker>/bootstrap.log`
- `logs/workers/<worker>/timeline.jsonl`

Task logs:
- `logs/tasks/<task-id>/run_<run>_worker_<suffix>.log`

Use:

```bash
jobman worker logs <worker> -f
jobman task show <task-id>
```

## Email Notifications

Mail settings live in task headers:

```bash
#JOBMAN --mail-user=you@example.com
#JOBMAN --mail-type=BEGIN,END,FAIL
```

On first submit of a task with mail headers, jobman prompts for Brevo setup.

Local config file:
- `.jobman_brevo.json`

That file can represent:
- configured sender/API key
- explicitly disabled local sending

If local Brevo sending is disabled, submit prints a warning and tasks continue without email delivery.

Brevo docs:
- https://developers.brevo.com/docs/getting-started

## Failure Handling

### TPU Failures

When TPU status becomes unavailable or terminal, workers try to:
- release the current task
- delete the TPU
- request a new TPU

The tmux session is not intentionally killed as part of TPU recovery.

### Worker Exceptions

Workers now recover in two layers:
- in-loop exception handling tries to recover/release the current task
- top-level worker supervision restarts the worker process if a fatal error escapes

### SSH/SCP Failures

Worker SSH/SCP paths now log:
- full `gcloud` command
- exit code
- stderr
- stdout when useful

### TPU Lifecycle Failures

TPU create/delete/describe failures in `jobman/tpu.py` now also surface raw `gcloud` stderr/stdout more clearly.

## State Layout

Default state root:

```text
/scratch/yx3038/pruning/jobman-lite
```

Override with:
- `JOBMAN_DIR`
- `JOBMAN_LOG_DIR`

Layout:

```text
JOBMAN_DIR/
├── workers.json
├── queue.json
├── queue.json.lock
└── worker_counter

JOBMAN_LOG_DIR/
├── workers/<worker>/
│   ├── worker.log
│   ├── bootstrap.log
│   ├── timeline.jsonl
│   └── <startup-script>.sh
└── tasks/<task-id>/
    ├── <submitted-script>.sh
    └── run_*.log
```

## Typical Workflow

1. Authenticate `gcloud` and select the project.
2. Start one or more workers.
3. Submit task scripts with `#JOBMAN` headers.
4. Watch `jobman status`.
5. Inspect `worker.log`, `bootstrap.log`, and task run logs when failures happen.
6. Pause, requeue, or delete tasks as needed.

## Troubleshooting

### `dead` worker

The tmux session is gone.

Check:
- `logs/workers/<worker>/worker.log`
- login node stability
- tmux availability

Then resume or reboot the worker.

### TPU create failure

Check:
- project
- auth
- quota
- zone
- runtime version
- surfaced `gcloud` stderr in worker logs

### Task stuck in `running`

Possible causes:
- worker died unexpectedly while task was assigned
- remote SSH process hung

Use:
- `jobman worker show <worker>`
- `jobman task show <task-id>`
- `jobman worker delete <worker>` if you need to force task release

### Mail requested but no emails arrive

Check:
- `.jobman_brevo.json`
- whether local Brevo sending was disabled
- sender identity is verified in Brevo
- worker logs for email warnings
