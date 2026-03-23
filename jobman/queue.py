"""File-based task queue with directory-based locking for atomicity."""

import json
import os
import shutil
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, List

from .utils import dir_lock, get_logger, jobman_dir, jobman_log_dir

logger = get_logger(__name__)

TaskStatus = str  # "pending" | "running" | "paused" | "done" | "failed" | "interrupted"


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
        script_path = self._snapshot_task_script(task_id, task_spec["script"])
        task: dict[str, Any] = {
            "id": task_id,
            "name": task_spec.get("name", task_id),
            "script": script_path,
            "source_script": task_spec["script"],
            "accelerator": task_spec["accelerator"],
            "zone": task_spec["zone"],
            "tpu_version": task_spec.get("tpu_version", "tpu-ubuntu2204-base"),
            "pricing": task_spec.get("pricing", "spot"),
            "max_retries": task_spec.get("max_retries", 3),
            "mail_user": task_spec.get("mail_user"),
            "mail_types": task_spec.get("mail_types", []),
            "mail_config_path": task_spec.get("mail_config_path"),
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
        """Atomically claim first matching pending task. Returns task or None.

        If the worker already owns a running task (e.g. from a lost release),
        that stale task is released back to pending first.
        """
        with self._locked() as tasks:
            # Guard: release any stale running tasks owned by this worker
            for task in tasks.values():
                if task.get("worker_id") == worker_id and task["status"] == "running":
                    logger.warning("Releasing stale running task %s from worker %s",
                                   task["id"], worker_id)
                    task["status"] = "pending"
                    task["worker_id"] = None
                    task["started"] = None

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

    def release(self, task_id: str, status: TaskStatus) -> Optional[dict]:
        """Mark task as done/failed/interrupted. Handles retry logic and returns updated task."""
        with self._locked() as tasks:
            task = tasks.get(task_id)
            if task is None:
                logger.warning("release: task %s not found", task_id)
                return None
            paused = task.get("status") == "paused"
            if status == "failed":
                task["fail_count"] = task.get("fail_count", 0) + 1
                if task["fail_count"] >= task.get("max_retries", 3):
                    task["status"] = "failed"
                    logger.info("Task %s permanently failed after %d retries",
                                task_id, task["fail_count"])
                elif paused:
                    task["worker_id"] = None
                    task["ended"] = _now()
                    logger.info("Task %s failed while paused; keeping it paused", task_id)
                    return dict(task)
                else:
                    task["status"] = "pending"
                    task["worker_id"] = None
                    logger.info("Task %s failed (%d/%d), re-queuing",
                                task_id, task["fail_count"], task["max_retries"])
                    return dict(task)
            elif status == "interrupted":
                # Re-queue without incrementing fail_count unless paused.
                task["status"] = "paused" if paused else "pending"
                task["worker_id"] = None
                task["started"] = None
                logger.info("Task %s interrupted, %s",
                            task_id, "keeping paused" if paused else "re-queuing")
                return dict(task)
            else:
                task["status"] = status
            task["ended"] = _now()
            task["worker_id"] = None
            return dict(task)

    def list(self) -> list:
        """Return all tasks sorted by submission time."""
        data = self._read()
        return sorted(data.values(), key=lambda t: t.get("submitted", ""))

    def cancel(self, task_id: str) -> bool:
        """Delete a task from the queue. Returns True if deleted."""
        with self._locked() as tasks:
            task = tasks.get(task_id)
            if task is None:
                return False
            del tasks[task_id]
            logger.info("Deleted task %s from queue", task_id)
            return True

    def pause(self, task_id: str) -> bool:
        """Pause a pending or running task. Returns True if paused."""
        with self._locked() as tasks:
            task = tasks.get(task_id)
            if task is None:
                return False
            if task["status"] not in ("pending", "running", "paused"):
                return False
            task["status"] = "paused"
            task["worker_id"] = None
            task["ended"] = _now()
            logger.info("Paused task %s", task_id)
            return True

    def reset(self, task_id: str) -> bool:
        """Reset a failed/paused task back to pending. Returns True if reset."""
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

    def release_worker_tasks(self, worker_id: str) -> List[str]:
        """Re-queue running tasks currently assigned to a worker."""
        released: List[str] = []
        with self._locked() as tasks:
            for task in tasks.values():
                if task.get("worker_id") != worker_id or task.get("status") != "running":
                    continue
                task["status"] = "pending"
                task["worker_id"] = None
                task["started"] = None
                released.append(task["id"])
        for task_id in released:
            logger.info("Released running task %s from worker %s", task_id, worker_id)
        return released

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

    def _snapshot_task_script(self, task_id: str, script_path: str) -> str:
        """Copy the submitted script into the task directory and return the snapshot path."""
        task_dir = os.path.join(jobman_log_dir(), "tasks", task_id)
        os.makedirs(task_dir, exist_ok=True)
        src = Path(script_path)
        dst = Path(task_dir) / src.name
        shutil.copy2(src, dst)
        return str(dst)

    @contextmanager
    def _locked(self):
        """Context manager that yields the mutable tasks dict with file lock held."""
        lock_dir = self._path + ".d.lock"
        with dir_lock(lock_dir):
            tasks = self._read()
            yield tasks
            self._write(tasks)
