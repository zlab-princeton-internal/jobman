<h1 align="center">Jobman v3</h1>

<p align="center">
  <img src="figs/jobman_logo.png" alt="Jobman Logo" width="512"/>
</p>

Jobman v3 is a modular and extensible job management system for TPU VMs, featuring a powerful **queue mode** for managing large-scale experiment runs across preemptible TPUs.

## News

- **2025-01-05**: v3 release with comprehensive queue mode for managing experiment queues across multiple TPUs.
- **2025-08-28**: Added support for conda and venv as well as [unit tests](tests).
- **2025-08-22**: Added quota and storage [viewer](https://github.com/Zephyr271828/jobman/blob/jobman-v2/GET_STARTED.md#profiling-commands).

## Installation

In order to use Jobman, you need to make sure `gcloud` is available on your machine in the first place. You may refer to [the official doc](https://cloud.google.com/sdk/docs/install) to do so.
Afterwards, also install `alpha` and `beta`:
```bash
gcloud components install alpha beta
```

Login with your gcloud account:
```bash
gcloud auth login
gcloud auth application-default login
```

Also make sure tmux has been installed:
```bash
tmux -V
```
If not, follow [tmux wiki](https://github.com/tmux/tmux/wiki/Installing) to install tmux.

Lastly, build the jobman package from source:
```bash
python -m pip install --upgrade pip
pip install -e .
```

### Quick Start
Try the following command to submit a minimal job:
```bash
jobman create configs/quick_start.yaml
```
Then check its status:
```bash
jobman list
```

## Get Started
Before you start using Jobman (properly), be sure to go through [GET_STARTED.md](GET_STARTED.md). This is vital for you to proceed to run your own jobs.

## Overall Structure
This section differs from the Get Started section as it explains briefly how Jobman works. Basically, each job is viewed as a data structure or a class by Jobman, with
- life cycle, including queueing, running, idle, and dead managed by a centralized data structure `jobman`. Specifically, `jobman` creates and kills tmux sessions to manage the jobs in the backend.
- corresponding tpus, ssh, gcsfuse, and environment config as attributes.
- all logs saved to `jobs/<user_id>/<job_id>/logs`.

### Caveats
- Since jobs live as tmux sessions, it's suggested that you run this tool on some remote host instead of some local machine, since tmux sessions may die after you shut down your machine.
- On the other hand, `jobman` lives as several local data files inside of `jobs/.jobman` and uses a lock to maintain the consistency. Therefore, please do not mess up with the files in `jobs/.jobman` unless you know what you're doing (if you cannot find `jobs/.jobman`, it's normal since it'll be created after you run your first job).

---

## Queue Mode

Queue mode allows multiple TPU workers to pick tasks from a shared queue instead of running fixed scripts. This is ideal for running many experiments across preemptible TPUs that may be interrupted at any time.

### Why Queue Mode?

- **Automatic task distribution**: Multiple TPUs automatically pick up the next available task
- **Preemption resilience**: If a TPU is preempted, the task is released and picked up by another worker
- **Retry on failure**: Failed tasks are automatically retried up to `max_retries` times
- **Concurrent execution**: Run multiple instances of the same task with `max_jobs > 1`
- **Zone-aware scheduling**: Tasks can be restricted to specific zones
- **Task dependencies**: Tasks can depend on other tasks completing first

### Queue Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        tasks.yaml                                │
│  (defines repos, tasks grouped by accelerator)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     tasks_state.yaml                             │
│  (auto-generated: tracks done/running/history per task)          │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌──────────┐        ┌──────────┐        ┌──────────┐
    │ Worker 1 │        │ Worker 2 │        │ Worker 3 │
    │ (v6e-64) │        │ (v6e-64) │        │ (v6e-128)│
    └──────────┘        └──────────┘        └──────────┘
         │                   │                   │
         └───────────────────┴───────────────────┘
                             │
                    Claim → Run → Release
```

### Setting Up Queue Mode

#### Step 1: Create the Task Queue Configuration

Create a `tasks.yaml` file that defines your repos and tasks:

```yaml
# queue_machines/tasks.yaml
# State tracked in: tasks_state.yaml (same directory)
# View status: jobman queue queue_machines/tasks.yaml

repos:
  myrepo: https://github.com/username/repo

# Defaults applied to ALL tasks (override per-accelerator or per-task)
defaults:
  env: my_venv              # venv/conda environment name
  repo: myrepo              # Default repo key
  branch: main              # Default branch
  workdir: myrepo           # Directory to clone into (~/<workdir>)
  zone: us-central1-b       # Default zone for task matching
  max_jobs: 1               # Max concurrent executions per task
  max_retries: 3            # Retries before marking task as permanently failed

# Tasks grouped by accelerator type
accelerators:
  v6e-64:
    # Optional: override defaults for all tasks under this accelerator
    # env: different_env
    # max_jobs: 2
    tasks:
      - id: train-model-seed42
        run: bash train/run_seed42.sh

      - id: train-model-seed43
        run: bash train/run_seed43.sh

      - id: eval-model-seed42
        run: bash eval/run_eval.sh
        depends_on: train-model-seed42  # Wait for training to complete

  v6e-128:
    tasks:
      - id: train-large-model
        run: bash train/run_large.sh
        max_jobs: 2           # Allow 2 TPUs to run this concurrently
        zone: us-east5-b      # Override zone for this specific task
```

#### Step 2: Create Queue Worker Configuration

Create a worker config that enables queue mode:

```yaml
# queue_machines/worker_v6e64.yaml

job:
  name: myproject-qw-${tpu.accelerator}  # Auto-suffixed with worker_num
  env_type: venv
  loop: true                              # Auto-restart on preemption
  remote_user: your_username

  # Environment variables exported before EACH task
  environment:
    BUCKET_NAME: ${gcsfuse.bucket_name}
    TPU_PREFIX: ${tpu.name}

  # Queue configuration
  queue:
    enabled: true
    config: queue_machines/tasks.yaml

    # Prefix scripts run ONCE when TPU is ready (before any task)
    prefix:
      - name: "gcloud-setup"
        workers: 0              # Run only on worker 0
        # workers: "all"        # Or run on all workers
        # required: true        # If true, job stops if prefix fails
        run: |
          echo "Setting up gcloud..."
          gcloud config set project my-project
          gcloud config set compute/zone ${tpu.zone}

tpu:
  allocation_mode: "queued-resources"
  accelerator: v6e-64
  name: myproject-qw-${tpu.accelerator}
  zone: us-central1-b
  version: v2-alpha-tpuv6e
  pricing: spot
  tags: ["jobman", "queue-worker"]

gcsfuse:
  bucket_name: my-bucket
  mount_path: /home/username/gcs

ssh:
  private_key: ~/.ssh/my_key
  identities:
    - private_key: ~/.ssh/my_key
      public_key: ~/.ssh/my_key.pub
      config_entry: |
        Host 10.* 34.* 35.*
          IdentityFile ~/.ssh/my_key
          IdentitiesOnly yes

venv:
  name: my_venv
  requirements_file: assets/requirements.txt
  python: "python3.10"
```

#### Step 3: Create and Run Queue Workers

```bash
# Create multiple queue workers (they auto-assign unique worker_num)
jobman create queue_machines/worker_v6e64.yaml
jobman create queue_machines/worker_v6e64.yaml
jobman create queue_machines/worker_v6e64.yaml
```

Each worker will:
1. Request a TPU (via queued-resources)
2. Run setup (SSH, gcsfuse, venv)
3. Run prefix scripts once
4. Loop: claim task → clone repo → run command → release task → repeat

### Queue Commands

```bash
# View queue status (shows all tasks, their state, running jobs)
jobman queue queue_machines/tasks.yaml

# View including hidden tasks
jobman queue queue_machines/tasks.yaml --show-all

# Reset a specific task (mark as not done, clear running)
jobman queue queue_machines/tasks.yaml --reset <task_id>

# Reset all tasks matching a pattern
jobman queue queue_machines/tasks.yaml --reset-pattern "eval*"

# Reset ALL tasks
jobman queue queue_machines/tasks.yaml --reset-all

# Reset only failed tasks (so they retry)
jobman queue queue_machines/tasks.yaml --reset-failed

# Mark a task as priority (will be picked up first)
jobman queue queue_machines/tasks.yaml --prioritize <task_id>

# Mark a task as done manually
jobman queue queue_machines/tasks.yaml --done <task_id>

# Clean up stale running entries (for crashed jobs)
jobman queue queue_machines/tasks.yaml --cleanup

# View details of a specific task
jobman queue queue_machines/tasks.yaml --task <task_id>
```

### Queue Worker Commands

```bash
# Requeue a job (release current task, restart to pick new one)
jobman requeue <job_id>

# Reboot with force to re-run prefix scripts
jobman reboot <job_id> --force-prefix

# Reboot with force to re-run ALL setup (SSH, gcsfuse, venv, prefix)
jobman reboot <job_id> --force
```

### Task Configuration Reference

Each task supports the following fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | string | required | Unique task identifier |
| `run` | string | required | Shell command to execute |
| `repo` | string | from defaults | Repository key (defined in `repos:`) |
| `branch` | string | "main" | Git branch to checkout |
| `workdir` | string | repo key | Directory name for clone (`~/<workdir>`) |
| `env` | string | from defaults | venv/conda environment name |
| `zone` | string | from defaults | Restrict task to specific zone |
| `max_jobs` | int | 1 | Max concurrent executions |
| `max_retries` | int | 3 | Retries before permanent failure |
| `depends_on` | string/list | none | Task ID(s) that must complete first |
| `workers` | int/list/"all" | [0] | Which TPU workers run the command |
| `hide` | bool | false | Hide from default queue view |

### Task State File

The state file (`tasks_state.yaml`) is auto-generated and tracks:

```yaml
tasks:
  train-model-seed42:
    done: true                    # Task completed successfully
    running: []                   # Currently running instances
    run_count: 2                  # Total run attempts
    fail_count: 0                 # Consecutive failures (resets on success)
    history:
      - job_id: '000045'
        started: '2025-01-15T08:00:00'
        ended: '2025-01-15T08:45:00'
        status: interrupted       # TPU was preempted
      - job_id: '000047'
        started: '2025-01-15T09:00:00'
        ended: '2025-01-15T14:30:00'
        status: completed

  train-model-seed43:
    done: false
    running:
      - job_id: '000048'
        started: '2025-01-15T14:35:00'
    run_count: 1
    history: []
```

**Status values:**
- `completed`: Task finished successfully (exit code 0)
- `failed`: Task errored (non-zero exit code, counts toward max_retries)
- `interrupted`: TPU preempted or job stopped (does not count as failure)
- `stale_cleanup`: Cleaned up from crashed job

### Task Logs

Task logs are organized by task ID:

```
jobs/tasks/
├── train-model-seed42/
│   ├── run_001_job000045_worker_0.log    # First run, worker 0 output
│   ├── run_001_job000045_meta.yaml       # Run metadata
│   ├── run_002_job000047_worker_0.log    # Second run
│   └── run_002_job000047_meta.yaml
├── train-model-seed43/
│   └── ...
└── hide/                                  # Hidden tasks (hide: true)
    └── ...
```

### Advanced: GCSFuse Caching

For data-intensive workloads, configure gcsfuse caching with a RAM disk:

```yaml
gcsfuse:
  bucket_name: my-bucket
  mount_path: /home/user/gcs-checkpoint  # No cache for checkpoints

  extra_mounts:
    # Smae bucket for cached mount
      cache:
        dir: /mnt/ramdisk
        metadata_ttl_secs: -1            # Never expire metadata
        stat_cache_max_size_mb: -1       # Unlimited stat cache
        type_cache_max_size_mb: -1       # Unlimited type cache
        file_cache_max_size_mb: 550000   # 550GB file cache
        cache_file_for_range_read: true
        enable_parallel_downloads: true
        # Create ramdisk before mounting
        prefix: |
          sudo mkdir -p /mnt/ramdisk
          if ! mountpoint -q /mnt/ramdisk; then
              sudo mount -t tmpfs -o size=550G tmpfs /mnt/ramdisk
          fi
```

### Switching Between Queue and Fixed Script Mode

To switch a job from queue mode to a fixed script:

1. Edit the worker config: set `queue.enabled: false` or remove the `queue` section
2. Add a `command` section with your script
3. Run `jobman reboot <job_id>`

```yaml
job:
  name: my-job
  queue:
    enabled: false    # Disable queue mode

command:
  cmd: |
    echo "Running fixed script"
    python train.py
  workers: [0]        # Or "all" for all workers
```

---

## Standard Job Commands

```bash
# Create and start a job
jobman create <config.yaml>

# List all jobs with status
jobman list

# Run job in foreground (for debugging)
jobman run <job_id>
jobman run <job_id> --force          # Force re-run all setup
jobman run <job_id> --force-prefix   # Force re-run prefix only

# SSH into job's TPU VM (worker 0)
jobman ssh <job_id>

# Resume a stopped job
jobman resume <job_id>

# Stop job (kills tmux session)
jobman stop <job_id>

# Reboot job (stop + resume)
jobman reboot <job_id>

# Delete job (stop + delete TPU)
jobman delete <job_id>

# Clean job (delete + remove logs)
jobman clean <job_id>
```

### Profiling Commands

```bash
jobman quota    # View TPU quota usage
jobman storage  # View GCS storage usage
jobman billing  # View billing report
```

---

## Other Resources

### TPU Intro
Boya Zeng has created [a comprehensive guide](https://github.com/boyazeng/tpu_intro) covering various problems and tips when using tpus. You can find the answers to most of the problems you may have regarding TPUs. This project also provides [a simple job management script](https://github.com/boyazeng/tpu_intro/tree/main/job_management).

### Ultra Create TPU
The design concept of Jobman is somewhat complex, but it aims to provide the easiest user interface s.t. users unfamiliar with TPUs can quickly get started.
For a simpler setup tool, you may refer to [`other_resources/ultra_create_tpu.sh`](other_resources/ultra_create_tpu.sh) by Peter Tong.

### Slack Chatbot
Boyang Zheng has also developed a brilliant Slack Chatbot that 1) automatically deletes dead tpu vms 2) profiles daily usage and sends to their Slack Channel. You may refer to it at [`other_resources/slack_chatbot`](other_resources/slack_chatbot).

---

## FAQ

1. **Q:** I ran `jobman create <config_path>` but nothing happens. What should I do?
   **A:** Under the hood, `jobman create` creates the job directory and starts the job process with tmux in the backend. If the job process fails, it fails silently since it's in tmux.
   The first debugging step is to run `jobman run <job_id>` where `<job_id>` is the id of the job you just created. This will run the job in the front end.

2. **Q:** How do I see which task a queue worker is running?
   **A:** Run `jobman queue <tasks.yaml>` to see the queue status. The "running" column shows which job IDs are currently running each task.

3. **Q:** A task keeps failing. How do I debug?
   **A:** Check the task logs in `jobs/tasks/<task_id>/`. Each run has its own log file with full command output.

4. **Q:** My queue worker is stuck not picking up tasks.
   **A:**
   - Check that the worker's accelerator matches tasks in the queue
   - Check zone matching if tasks have specific zones
   - Run `jobman queue <tasks.yaml> --cleanup` to clear stale entries
   - Check `jobman run <job_id>` for errors

5. **Q:** How do I run the same task on multiple TPUs simultaneously?
   **A:** Set `max_jobs: N` on the task to allow N concurrent executions.

---

## Contributions & Feedback
- If you have any issues with this project or want to contribute to it, please first open an issue in the `Issues` section. This will be of great help to the maintenance of this project!
- You may also contact Yufeng Xu [yx3038@nyu.edu](mailto:yx3038@nyu.edu) for further communication.
- Also, if you would like to contribute to this project, please refer to [CONTRIBUTING.md](CONTRIBUTING.md).
