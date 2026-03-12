#!/usr/bin/env bash
set -euo pipefail

ZONE=""
BUCKET_NAME=""
PRIMARY_MOUNT_PATH="$HOME/gcs-bucket"
DATA_MOUNT_PATH="$HOME/gcs-data"
RAMDISK_PATH="/mnt/ramdisk"
RAMDISK_SIZE="550G"
VENV_NAME="maxtext_env"
VENV_DIR="$HOME/.venvs/$VENV_NAME"
PYTHON_BIN="python3.10"
MAXTEXT_REPO_URL="https://github.com/Zephyr271828/maxtext.git"
MAXTEXT_BRANCH="test_new"
MAXTEXT_DIR="$HOME/maxtext"
REQUIREMENTS_FILE="$MAXTEXT_DIR/requirements.txt"
STATE_DIR="$HOME/.jobman-bootstrap/$VENV_NAME"
REQ_HASH_FILE="$STATE_DIR/requirements.sha256"
LOG_PREFIX="[jobman-bootstrap]"

log() {
  printf '%s %s %s\n' "$(date -Is)" "$LOG_PREFIX" "$*"
}

warn() {
  log "WARN: $*"
}

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    log "ERROR: required command '$cmd' is not available"
    exit 1
  fi
}

unlock_dpkg_if_needed() {
  if sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; then
    local lock_pid
    lock_pid="$(sudo lsof -t /var/lib/dpkg/lock-frontend || true)"
    if [[ -n "$lock_pid" ]]; then
      warn "Killing process $lock_pid holding dpkg lock"
      sudo kill -9 "$lock_pid" || true
      sleep 2
    fi
  fi
}

ensure_state_dir() {
  mkdir -p "$STATE_DIR"
}

detect_zone() {
  if [[ -n "$ZONE" ]]; then
    printf '%s\n' "$ZONE"
    return
  fi

  require_command curl
  curl -fsH "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/zone" |
    awk -F/ '{print $NF}'
}

bucket_for_zone() {
  local zone="$1"
  case "$zone" in
    us-central2-b) printf '%s\n' "llm_pruning_us_central2_b" ;;
    us-central1-b) printf '%s\n' "llm_pruning_us_central1" ;;
    us-east5-b) printf '%s\n' "llm_pruning_us_east5" ;;
    *)
      log "ERROR: no bucket configured for zone $zone"
      log "ERROR: extend bucket_for_zone() for new zones"
      exit 1
      ;;
  esac
}

configure_region_defaults() {
  ZONE="$(detect_zone)"
  if [[ -z "$BUCKET_NAME" ]]; then
    BUCKET_NAME="$(bucket_for_zone "$ZONE")"
  fi
  log "Resolved zone=$ZONE bucket=$BUCKET_NAME"
}

ensure_mount_dir() {
  local mount_path="$1"
  if [[ -d "$mount_path" ]]; then
    log "Mount directory exists: $mount_path"
    return
  fi
  mkdir -p "$mount_path"
  log "Created mount directory: $mount_path"
}

ensure_ramdisk() {
  log "Checking RAM disk at $RAMDISK_PATH"
  require_command mountpoint
  if [[ -d "$RAMDISK_PATH" ]]; then
    log "Mount directory exists: $RAMDISK_PATH"
  else
    sudo mkdir -p "$RAMDISK_PATH"
    log "Created mount directory: $RAMDISK_PATH"
  fi

  if mountpoint -q "$RAMDISK_PATH"; then
    log "RAM disk already mounted at $RAMDISK_PATH"
    return
  fi

  sudo mount -t tmpfs -o "size=$RAMDISK_SIZE" tmpfs "$RAMDISK_PATH"
  log "Mounted RAM disk at $RAMDISK_PATH with size $RAMDISK_SIZE"
}

ensure_gcsfuse() {
  if command -v gcsfuse >/dev/null 2>&1; then
    log "gcsfuse already installed"
    return
  fi

  require_command curl
  require_command lsb_release

  log "Installing gcsfuse"
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt gcsfuse-$(lsb_release -c -s) main" |
    sudo tee /etc/apt/sources.list.d/gcsfuse.list >/dev/null
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg |
    sudo tee /usr/share/keyrings/cloud.google.asc >/dev/null

  unlock_dpkg_if_needed
  sudo apt-get update -y
  unlock_dpkg_if_needed
  sudo apt-get install -y gcsfuse

  require_command gcsfuse
  log "gcsfuse installed successfully"
}

mount_bucket_if_needed() {
  local bucket_name="$1"
  local mount_path="$2"
  shift 2

  log "Checking bucket mount $bucket_name -> $mount_path"
  require_command mountpoint
  ensure_gcsfuse
  ensure_mount_dir "$mount_path"

  if mountpoint -q "$mount_path"; then
    log "Bucket already mounted at $mount_path"
    return
  fi

  gcsfuse \
    --dir-mode=777 \
    --file-mode=777 \
    --kernel-list-cache-ttl-secs=-1 \
    --metadata-cache-ttl-secs=-1 \
    --stat-cache-max-size-mb=-1 \
    --type-cache-max-size-mb=-1 \
    "$@" \
    "$bucket_name" \
    "$mount_path"
  log "Mounted bucket $bucket_name at $mount_path"
}

ensure_primary_bucket_mount() {
  mount_bucket_if_needed \
    "$BUCKET_NAME" \
    "$PRIMARY_MOUNT_PATH"
}

ensure_cached_data_mount() {
  ensure_ramdisk
  mount_bucket_if_needed \
    "$BUCKET_NAME" \
    "$DATA_MOUNT_PATH" \
    --cache-dir="$RAMDISK_PATH" \
    --metadata-cache-ttl-secs=-1 \
    --stat-cache-max-size-mb=-1 \
    --type-cache-max-size-mb=-1 \
    --file-cache-max-size-mb=550000 \
    --file-cache-cache-file-for-range-read=true \
    --file-cache-enable-parallel-downloads=true
}

ensure_python() {
  log "Checking Python interpreter: $PYTHON_BIN"
  require_command "$PYTHON_BIN"
}

ensure_python_venv_support() {
  local venv_pkg="${PYTHON_BIN}-venv"
  if dpkg -s "$venv_pkg" >/dev/null 2>&1; then
    log "$venv_pkg already installed"
    return
  fi

  log "Installing $venv_pkg"
  unlock_dpkg_if_needed
  sudo apt-get update -y
  unlock_dpkg_if_needed
  sudo apt-get install -y "$venv_pkg"
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

ensure_venv() {
  ensure_python
  ensure_python_venv_support

  if [[ -x "$VENV_DIR/bin/python" && -x "$VENV_DIR/bin/pip" ]]; then
    log "Virtualenv already exists: $VENV_DIR"
    return
  fi

  if [[ -d "$VENV_DIR" ]]; then
    warn "Virtualenv at $VENV_DIR is incomplete; removing and recreating it"
    rm -rf "$VENV_DIR"
  fi

  mkdir -p "$(dirname "$VENV_DIR")"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip --version >/dev/null
  log "Created virtualenv: $VENV_DIR"
}

current_requirements_hash() {
  sha256sum "$REQUIREMENTS_FILE" | awk '{print $1}'
}

installed_requirements_hash() {
  if [[ -f "$REQ_HASH_FILE" ]]; then
    cat "$REQ_HASH_FILE"
  fi
}

ensure_requirements() {
  ensure_state_dir
  ensure_maxtext_repo
  ensure_venv

  if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    warn "Requirements file not found, skipping pip install: $REQUIREMENTS_FILE"
    return
  fi

  require_command sha256sum
  local wanted_hash
  wanted_hash="$(current_requirements_hash)"
  local installed_hash
  installed_hash="$(installed_requirements_hash || true)"

  if [[ -n "$installed_hash" && "$installed_hash" == "$wanted_hash" ]]; then
    log "Requirements already installed for hash $wanted_hash"
    return
  fi

  "$VENV_DIR/bin/pip" install --upgrade pip
  "$VENV_DIR/bin/pip" install -r "$REQUIREMENTS_FILE"
  printf '%s\n' "$wanted_hash" > "$REQ_HASH_FILE"
  log "Installed requirements from $REQUIREMENTS_FILE"
}

print_summary() {
  log "Bootstrap summary:"
  log "  zone=$ZONE"
  log "  bucket=$BUCKET_NAME"
  log "  primary_mount=$PRIMARY_MOUNT_PATH"
  log "  data_mount=$DATA_MOUNT_PATH"
  log "  maxtext_dir=$MAXTEXT_DIR"
  log "  maxtext_branch=$MAXTEXT_BRANCH"
  log "  venv_dir=$VENV_DIR"
  log "  requirements_file=$REQUIREMENTS_FILE"
}

main() {
  log "Starting TPU bootstrap on $(hostname)"
  ensure_state_dir
  configure_region_defaults
  ensure_gcsfuse
  ensure_primary_bucket_mount
  ensure_cached_data_mount
  ensure_venv
  ensure_requirements
  print_summary
  log "Bootstrap complete"
}

main "$@"
