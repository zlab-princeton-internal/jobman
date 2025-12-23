#!/bin/bash

set -euo pipefail

BUCKET_NAME=""
MNT_DIR="~/gcs-bucket"
VENV_PATH="~/venvs/maxtext"

setup_ssh() {
    mkdir -p ~/.ssh 
    chmod 700 ~/.ssh
    
}

setup_gcs() {
    echo "[INFO] Adding gcsfuse apt repo ..."
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt gcsfuse-$(lsb_release -c -s) main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list

    echo "[INFO] Installing repo key ..."
    curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc >/dev/null

    # best-effort unlock dpkg to avoid stuck installs
    if sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; then
        LOCK_PID=$(sudo lsof -t /var/lib/dpkg/lock-frontend || true)
        if [ -n "$LOCK_PID" ]; then
            echo "[WARN] Killing process $LOCK_PID holding dpkg lock"
            sudo kill -9 $LOCK_PID || true
            sleep 2
        fi
    fi

    echo "[INFO] apt update & install gcsfuse ..."
    sudo apt-get update -y && sudo apt-get install -y gcsfuse

    if ! command -v gcsfuse >/dev/null 2>&1; then
        echo "[ERROR] gcsfuse not found after install"
        exit 1
    fi

    set -e
    echo "[INFO] Ensuring mount path exists: {mnt}"
    mkdir -p ${MNT_DIR}

    echo "[INFO] Mounting bucket if not already mounted ..."
    mountpoint -q ${MNT_DIR} || gcsfuse --implicit-dirs --dir-mode=777 --file-mode=777 ${BUCKET_NAME} ${MNT_DIR}

    echo "[INFO] Listing mount ..."
    ls -la ${MNT_DIR} || true
}


check_gcs() {
    command -v gcsfuse >/dev/null 2>&1 && mountpoint -q "${MNT_DIR}"
}

setup_venv() {
    # best-effort unlock dpkg to avoid stuck installs
    if sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; then
        LOCK_PID=$(sudo lsof -t /var/lib/dpkg/lock-frontend || true)
        if [ -n "$LOCK_PID" ]; then
            echo "[WARN] Killing process $LOCK_PID holding dpkg lock"
            sudo kill -9 $LOCK_PID || true
            sleep 2
        fi
    fi

    mkdir -p $(dirname ${VENV_PATH})
    python -m venv ${VENV_PATH}
    source ${VENV_PATH}/bin/activate
    pip install --upgrade pip

    GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
    git clone -b test https://github.com/Zephyr271828/maxtext.git
    cd maxtext
    pip install -r requirements.txt
}

check_venv() {
    [ -x "${VENV_PATH}/bin/activate" ]
}

train() {
    host_id=$(python -c "import socket; print(socket.gethostname())" | awk -F'-' '{print $NF}')
    if [[ $host_id -eq 0 ]]; then
        exit 0
    fi

    source ${VENV_PATH}/bin/activate

    cd maxtext
    GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
    git pull origin test
    git submodule update --init --recursive

    bash scripts/llama3.1-8b_L200.sh --lr=5e-4 --shuffle=True
}

main() {
    setup_ssh

    echo "[CHECK] GCS status..."
    # if ! check_gcs; then
    #     echo "[ACTION] Running GCS setup"
    #     setup_gcs
    # else
    #     echo "[OK] GCS already set up"
    # fi

    echo "[CHECK] Python venv status..."
    if ! check_venv; then
        echo "[ACTION] Running venv setup"
        setup_venv
    else
        echo "[OK] Python venv already exists"
    fi

    # train
}

main "$@"