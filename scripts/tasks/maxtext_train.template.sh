#!/usr/bin/env bash
#JOBMAN --accelerator=__ACCEL__
#JOBMAN --zone=__ZONE__
#JOBMAN --name=__JOB_NAME__
#JOBMAN --max-retries=3

set -euo pipefail

MAXTEXT_REPO_URL="https://github.com/Zephyr271828/maxtext.git"
MAXTEXT_BRANCH="test_new"
MAXTEXT_DIR="$HOME/maxtext"
VENV_NAME="maxtext_env"
VENV_DIR="$HOME/.venvs/$VENV_NAME"
LOG_PREFIX="[jobman-task]"

TRAIN_CMD="__TRAIN_CMD__"

log() {
  printf '%s %s %s\n' "$(date -Is)" "$LOG_PREFIX" "$*"
}

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    log "ERROR: required command '$cmd' is not available"
    exit 1
  fi
}

ensure_maxtext_repo() {
  require_command git

  if [[ ! -d "$MAXTEXT_DIR/.git" ]]; then
    mkdir -p "$(dirname "$MAXTEXT_DIR")"
    git clone -b "$MAXTEXT_BRANCH" "$MAXTEXT_REPO_URL" "$MAXTEXT_DIR"
    log "Cloned MaxText repo to $MAXTEXT_DIR on branch $MAXTEXT_BRANCH"
    return
  fi

  git -C "$MAXTEXT_DIR" fetch origin "$MAXTEXT_BRANCH"
  git -C "$MAXTEXT_DIR" checkout "$MAXTEXT_BRANCH"
  git -C "$MAXTEXT_DIR" reset --hard "origin/$MAXTEXT_BRANCH"
  log "Updated MaxText repo at $MAXTEXT_DIR to origin/$MAXTEXT_BRANCH"
}

activate_venv() {
  if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    log "ERROR: virtualenv not found at $VENV_DIR"
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  log "Activated virtualenv $VENV_DIR"
}

run_training() {
  cd "$MAXTEXT_DIR"
  log "Running training command in $MAXTEXT_DIR"
  log "Command: $TRAIN_CMD"
  bash -lc "$TRAIN_CMD"
}

main() {
  log "Starting task on $(hostname)"
  log "TPU name=$JOBMAN_TPU_NAME zone=$JOBMAN_ZONE workers=$JOBMAN_NUM_WORKERS"
  ensure_maxtext_repo
  activate_venv
  run_training
  log "Task complete"
}

main "$@"
