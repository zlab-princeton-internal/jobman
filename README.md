# Jobman-lite

Lightweight TPU job orchestration. Workers hold TPUs persistently and poll a shared queue for tasks.

Full manual:
- [docs/manual.md](/gpfsnyu/scratch/yx3038/pruning/jobman-lite/docs/manual.md)

## Install

```bash
conda create -n jobman-lite python=3.12 -y
conda activate jobman-lite
pip install -e .
```

System prerequisites:
- `tmux`
- Google Cloud SDK (`gcloud`)
- TPU APIs enabled with sufficient quota and IAM permissions

```bash
tmux -V
gcloud --version
gcloud auth list
gcloud config get-value project
```

## Quick Start

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project <your-gcp-project>
jobman worker start --accelerator=v4-8 --zone=us-central2-b
jobman task submit train.sh
jobman status
jobman task show <task-id>
```

## Usage

```bash
# Start a worker (holds the TPU, polls for tasks)
jobman worker start --accelerator=v4-8 --zone=us-central2-b
jobman worker stop my-tpu
jobman worker stop --all
jobman worker stop -a v4-16 -z us-central2-b
jobman worker resume my-tpu
jobman worker resume -a v4-16
jobman worker reboot my-tpu
jobman worker reboot --all
jobman worker reboot -a v4-16 -z us-central2-b

# Submit tasks
jobman task submit train.sh
jobman task submit train.sh --name=run-1 --accelerator=v4-8 --zone=us-central2-b

# Monitor
jobman status
jobman task show <task-id>
jobman worker show my-tpu

# Manage tasks
jobman task pause <task-id>
jobman task pause -p 'llama3*s50*3e-4*'
jobman task requeue <task-id>   # put a failed/paused task back to pending
jobman task requeue -a v4-128 -z us-central2-b
jobman task delete <task-id>
jobman task delete --all
```

For detailed setup, task headers, email notifications, state layout, failure handling, and troubleshooting, see:
- [docs/manual.md](/gpfsnyu/scratch/yx3038/pruning/jobman-lite/docs/manual.md)
