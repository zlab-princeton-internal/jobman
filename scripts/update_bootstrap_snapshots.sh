#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SOURCE_SCRIPT="${1:-$REPO_ROOT/scripts/setup/maxtext_bootstrap.sh}"
LOG_ROOT="${JOBMAN_LOG_DIR:-$(pwd)/logs}"
WORKER_ROOT="$LOG_ROOT/workers"

if [[ ! -f "$SOURCE_SCRIPT" ]]; then
  echo "Source bootstrap script not found: $SOURCE_SCRIPT" >&2
  exit 1
fi

if [[ ! -d "$WORKER_ROOT" ]]; then
  echo "Worker log directory not found: $WORKER_ROOT" >&2
  exit 1
fi

script_name="$(basename "$SOURCE_SCRIPT")"
updated=0

for worker_dir in "$WORKER_ROOT"/*; do
  [[ -d "$worker_dir" ]] || continue
  cp "$SOURCE_SCRIPT" "$worker_dir/$script_name"
  echo "Updated: $worker_dir/$script_name"
  updated=$((updated + 1))
done

echo "Updated $updated worker bootstrap snapshot(s)."
