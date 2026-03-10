"""File-based task queue with fcntl locking for atomicity."""

import fcntl
import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from .utils import get_logger, jobman_dir

logger = get_logger(__name__)

TaskStatus = str  # "pending" | "running" | "done" | "failed" | "cancelled" | "interrupted"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Queue:
    def __init__(self, state_dir: Optional[str] = None):
        self._dir = state_dir or jobman_dir()
        self._path = os.path.join(self._dir, "queue.json")
        os.makedirs(self._dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, task_spec: dict[str, Any]) -> str:
        """Add a task to the queue. Returns the assigned task ID."""
        task_id = "task_" + uuid.uuid4().hex[:8]
        task: dict[str, Any] = {
            "id": task_id,
            "name": task_spec.get("name", task_id),
            "script": task_spec["script"],
            "accelerator": task_spec["accelerator"],
            "zone": task_spec["zone"],
            "tpu_version": task_spec.get("tpu_version", "tpu-ubuntu2204-base"),
            "pricing": task_spec.get("pricing", "spot"),
            "max_retries": task_spec.get("max_retries", 3),
            "status": "pending",
            "worker_id": None,
            "submitted": _now(),
            "started": None,
            "ended": None,
            "fail_count": 0,
            "run_count": 0,
        }
        with self._locked() as tasks:
            tasks[task_id] = task
        logger.info("Submitted task %s (%s)", task_id, task["name"])
        return task_id

    def claim(self, accelerator: str, zone: str, worker_id: str) -> Optional[dict]:
        """Atomically claim first matching pending task. Returns task or None."""
        with self._locked() as tasks:
            for task in tasks.values():
                if (task["status"] == "pending"
                        and task["accelerator"] == accelerator
                        and task["zone"] == zone):
                    task["status"] = "running"
                    task["worker_id"] = worker_id
                    task["started"] = _now()
                    task["run_count"] = task.get("run_count", 0) + 1
                    logger.info("Claimed task %s for worker %s", task["id"], worker_id)
                    return dict(task)
        return None

    def release(self, task_id: str, status: TaskStatus) -> None:
        """Mark task as done/failed/interrupted. Handles retry logic."""
        with self._locked() as tasks:
            task = tasks.get(task_id)
            if task is None:
                logger.warning("release: task %s not found", task_id)
                return
            if status == "failed":
                task["fail_count"] = task.get("fail_count", 0) + 1
                if task["fail_count"] >= task.get("max_retries", 3):
                    task["status"] = "failed"
                    logger.info("Task %s permanently failed after %d retries",
                                task_id, task["fail_count"])
                else:
                    task["status"] = "pending"
                    task["worker_id"] = None
                    logger.info("Task %s failed (%d/%d), re-queuing",
                                task_id, task["fail_count"], task["max_retries"])
                    return
            elif status == "interrupted":
                # Re-queue without incrementing fail_count
                task["status"] = "pending"
                task["worker_id"] = None
                task["started"] = None
                logger.info("Task %s interrupted, re-queuing", task_id)
                return
            else:
                task["status"] = status
            task["ended"] = _now()
            task["worker_id"] = None

    def list(self) -> list:
        """Return all tasks sorted by submission time."""
        data = self._read()
        return sorted(data.values(), key=lambda t: t.get("submitted", ""))

    def cancel(self, task_id: str) -> bool:
        """Cancel a pending or running task. Returns True if cancelled."""
        with self._locked() as tasks:
            task = tasks.get(task_id)
            if task is None:
                return False
            if task["status"] not in ("pending", "running"):
                return False
            task["status"] = "cancelled"
            task["ended"] = _now()
            return True

    def reset(self, task_id: str) -> bool:
        """Reset a failed/cancelled task back to pending. Returns True if reset."""
        with self._locked() as tasks:
            task = tasks.get(task_id)
            if task is None:
                return False
            task["status"] = "pending"
            task["worker_id"] = None
            task["started"] = None
            task["ended"] = None
            task["fail_count"] = 0
            return True

    def get(self, task_id: str) -> Optional[dict]:
        """Return a single task by ID."""
        return self._read().get(task_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read(self) -> dict:
        if not os.path.exists(self._path):
            return {}
        with open(self._path) as f:
            return json.load(f)

    def _write(self, tasks: dict) -> None:
        tmp = self._path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(tasks, f, indent=2)
        os.replace(tmp, self._path)

    @contextmanager
    def _locked(self):
        """Context manager that yields the mutable tasks dict with file lock held."""
        lock_path = self._path + ".lock"
        with open(lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                tasks = self._read()
                yield tasks
                self._write(tasks)
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
