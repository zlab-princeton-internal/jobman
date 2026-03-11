#!/bin/bash
set -euxo pipefail

# Example TPU VM startup script.
# Runs on each TPU VM host during provisioning.

LOG_DIR=$HOME/log/jobman-lite
LOG_FILE="${LOG_DIR}/startup.log"

mkdir -p "${LOG_DIR}"

{
  echo "=== jobman-lite dummy startup ==="
  date -Is
  echo "hostname: $(hostname)"
  echo "user: $(whoami)"
  echo "pwd: $(pwd)"
  echo "HOME: ${HOME:-}"
  echo "PATH: ${PATH:-}"
  echo "=== done ==="
} | tee -a "${LOG_FILE}"
