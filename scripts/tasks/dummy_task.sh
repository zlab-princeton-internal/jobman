#!/bin/bash
#JOBMAN --accelerator=v4-8
#JOBMAN --zone=us-central2-b
#JOBMAN --name=hello-tpu
#JOBMAN --max-retries=3

# Run a simple echo on all TPU hosts to verify connectivity.
# JOBMAN injects: JOBMAN_TPU_NAME, JOBMAN_ZONE, JOBMAN_NUM_WORKERS

echo "=== Running on worker 0 (orchestrator) ==="
echo "TPU name    : $JOBMAN_TPU_NAME"
echo "Zone        : $JOBMAN_ZONE"
echo "Num workers : $JOBMAN_NUM_WORKERS"
echo "Hostname    : $(hostname)"
echo ""

SSH_KEY_FILE="${JOBMAN_GCLOUD_SSH_KEY_FILE:-$HOME/.ssh/google_compute_engine}"
SSH_DIR="$(dirname "$SSH_KEY_FILE")"
mkdir -p "$SSH_DIR"

if [[ ! -f "$SSH_KEY_FILE" || ! -f "${SSH_KEY_FILE}.pub" ]]; then
  rm -f "$SSH_KEY_FILE" "${SSH_KEY_FILE}.pub"
  ssh-keygen -t rsa -f "$SSH_KEY_FILE" -N '' -q
fi

eval "$(ssh-agent -s)" >/dev/null
ssh-add "$SSH_KEY_FILE" >/dev/null

echo "=== Broadcasting echo to all $JOBMAN_NUM_WORKERS workers ==="
gcloud compute tpus tpu-vm ssh "$JOBMAN_TPU_NAME" \
  --zone="$JOBMAN_ZONE" \
  --worker=all \
  --quiet \
  --force-key-file-overwrite \
  --ssh-key-file="$SSH_KEY_FILE" \
  -- 'echo "Hello from $(hostname) (worker $(curl -sf metadata.google.internal/computeMetadata/v1/instance/attributes/worker-id -H Metadata-Flavor:Google 2>/dev/null || echo ?))"'

echo ""
echo "=== Done ==="
