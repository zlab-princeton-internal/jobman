import logging
import os
import sys


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
    return os.environ.get("JOBMAN_DIR", "/scratch/yx3038/pruning/jobman-lite")


def jobman_log_dir() -> str:
    """Return the jobman log directory."""
    return os.environ.get("JOBMAN_LOG_DIR", os.path.join(jobman_dir(), "logs"))
