import json
import logging
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

BREVO_DOCS_URL = "https://developers.brevo.com/docs/getting-started"
BREVO_CONFIG_FILE = ".jobman_brevo.json"

def get_logger(name: str, log_file: "str | None" = None, level: int = logging.INFO,
               file_mode: str = "a") -> logging.Logger:
    """Create a logger with console and optional file output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, mode=file_mode)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def jobman_dir() -> str:
    """Return the jobman state directory."""
    return os.environ.get("JOBMAN_DIR", "/mnt/weka/home/yucheng/yufeng/jobman-lite")


def jobman_log_dir() -> str:
    """Return the jobman log directory."""
    return os.environ.get("JOBMAN_LOG_DIR", os.path.join(jobman_dir(), "logs"))


@contextmanager
def dir_lock(lock_dir: str, timeout: float = 60, stale_timeout: float = 300):
    """Cross-process lock using mkdir (atomic on local and distributed filesystems).

    Uses os.mkdir which is atomic even on NFS/WekaFS, unlike fcntl.flock.
    """
    pid_file = os.path.join(lock_dir, "pid")
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.mkdir(lock_dir)
            # Write PID for stale lock detection
            try:
                with open(pid_file, "w") as f:
                    f.write(str(os.getpid()))
            except OSError:
                pass
            break
        except FileExistsError:
            # Check for stale lock
            try:
                lock_mtime = os.path.getmtime(lock_dir)
                if (time.time() - lock_mtime) > stale_timeout:
                    shutil.rmtree(lock_dir, ignore_errors=True)
                    continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                # Force-break the lock as last resort
                shutil.rmtree(lock_dir, ignore_errors=True)
                continue
            time.sleep(0.05)
    try:
        yield
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)


def brevo_config_path() -> Path:
    """Return the path to the local Brevo config file."""
    return Path.cwd() / BREVO_CONFIG_FILE


def load_brevo_config(path: str | None = None) -> dict[str, str | bool]:
    """Load Brevo config from disk. Returns an empty dict if missing."""
    cfg_path = Path(path) if path else brevo_config_path()
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Brevo config at {cfg_path} must be a JSON object.")
    return {
        "api_key": str(data.get("api_key", "")).strip(),
        "sender_email": str(data.get("sender_email", "")).strip(),
        "disabled": bool(data.get("disabled", False)),
    }


def save_brevo_config(
    api_key: str,
    sender_email: str,
    path: str | None = None,
    *,
    disabled: bool = False,
) -> Path:
    """Persist Brevo config to disk and return the written path."""
    cfg_path = Path(path) if path else brevo_config_path()
    cfg_path.write_text(
        json.dumps(
            {
                "api_key": api_key,
                "sender_email": sender_email,
                "disabled": disabled,
            },
            indent=2,
        ) + "\n"
    )
    try:
        os.chmod(cfg_path, 0o600)
    except OSError:
        pass
    return cfg_path


def send_brevo_email(
    *,
    recipient: str,
    subject: str,
    text_content: str,
    config_path: str | None = None,
    sender_name: str = "jobman-lite",
) -> bool:
    """Send a transactional email via Brevo. Returns True on success."""
    if os.environ.get("DISABLE_EMAIL", "").lower() in ("1", "true", "yes", "on"):
        return False

    config = load_brevo_config(config_path)
    if config.get("disabled"):
        return False
    api_key = config.get("api_key")
    sender_email = config.get("sender_email")
    if not api_key or not sender_email:
        return False

    payload = {
        "sender": {"email": sender_email, "name": sender_name},
        "to": [{"email": recipient}],
        "subject": subject,
        "textContent": text_content,
    }
    req = urllib_request.Request(
        "https://api.brevo.com/v3/smtp/email",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "accept": "application/json",
            "api-key": api_key,
            "content-type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=30):
            return True
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        get_logger(__name__).warning("Brevo email failed (%s): %s", exc.code, body)
    except urllib_error.URLError as exc:
        get_logger(__name__).warning("Brevo email failed: %s", exc)
    return False
