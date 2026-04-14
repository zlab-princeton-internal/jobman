#!/usr/bin/env bash
set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate jobman-lite

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/tasks/maxtext_train.template.sh"
TMP_DIR="${SCRIPT_DIR}/.tmp_submissions"

# Edit these arrays directly.
ACCELS=(
  "v6e-4"
  # "v6e-128"
)

ZONES=(
#   "us-central2-b"
  "us-central1-b"
  "us-east5-b"
)

MODEL_TYPES=(
  # "llama3.1-4b-depth"
  "llama3.1-4b-flap"
)

PRETRAINS=(
  "scratch"
  # "meta"
  "l200"
)

LENGTHS=(
  # "10"
  # "30"
  "50"
  "250"
  "500"
)

LRS=(
  # "1e-4"
  "3e-4"
)

mkdir -p "$TMP_DIR"
for accel in "${ACCELS[@]}"; do
  for zone in "${ZONES[@]}"; do
    for model_type in "${MODEL_TYPES[@]}"; do
      for pretrain in "${PRETRAINS[@]}"; do
        case "$pretrain" in
          scratch) prefix="" ;;
          l200) prefix="L200_" ;;
          meta) prefix="Meta_" ;;
          *)
            echo "Invalid pretrain value: $pretrain" >&2
            exit 1
            ;;
        esac

        for length in "${LENGTHS[@]}"; do
          for lr in "${LRS[@]}"; do
            job_name="eval-${model_type}-${pretrain}-s${length}-lr-${lr//./p}"
            train_cmd="bash scripts/eval/${model_type}/${prefix}S${length}.sh --lr=${lr}"
            tmp_script="$(mktemp "${TMP_DIR}/maxtext.XXXXXX.sh")"

            sed \
              -e "s|__ACCEL__|$accel|g" \
              -e "s|__ZONE__|$zone|g" \
              -e "s|__JOB_NAME__|$job_name|g" \
              -e "s|__TRAIN_CMD__|$train_cmd|g" \
              "$TEMPLATE" > "$tmp_script"

            chmod +x "$tmp_script"
            echo "Submitting: $job_name"
            jobman task submit "$tmp_script"
            rm -f "$tmp_script"
          done
        done
      done
    done
  done
done
