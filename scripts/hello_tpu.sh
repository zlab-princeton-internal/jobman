#!/bin/bash
#JOBMAN --accelerator=v4-8
#JOBMAN --zone=us-central2-b
#JOBMAN --name=hello-tpu
#JOBMAN --max-retries=1

# Run a simple echo on all TPU hosts to verify connectivity.
# JOBMAN injects: JOBMAN_TPU_NAME, JOBMAN_ZONE, JOBMAN_NUM_WORKERS

echo "=== Running on worker 0 (orchestrator) ==="
echo "TPU name    : $JOBMAN_TPU_NAME"
echo "Zone        : $JOBMAN_ZONE"
echo "Num workers : $JOBMAN_NUM_WORKERS"
echo "Hostname    : $(hostname)"
echo ""

echo "=== Broadcasting echo to all $JOBMAN_NUM_WORKERS workers ==="
gcloud compute tpus tpu-vm ssh "$JOBMAN_TPU_NAME" \
  --zone="$JOBMAN_ZONE" \
  --worker=all \
  -- 'echo "Hello from $(hostname) (worker $(curl -sf metadata.google.internal/computeMetadata/v1/instance/attributes/worker-id -H Metadata-Flavor:Google 2>/dev/null || echo ?))"'

echo ""
echo "=== Done ==="