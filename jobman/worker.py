"""Worker loop: TPU holder + task executor."""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

from .queue import Queue
from .tpu import TPU, AllocationMode, Pricing, DEFAULT_TPU_VERSION
from .utils import get_logger, jobman_dir, jobman_log_dir, send_brevo_email

logger = get_logger(__name__)
_RETRYABLE_FAILURE_PATTERN = "Main command finished with errors, check the logs located in"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tee_to_stdout(proc, buffer: list[str] | None = None) -> None:
    """Read proc.stdout lines and write them to sys.stdout."""
    for line in proc.stdout:
        if buffer is not None:
            buffer.append(line)
        sys.stdout.write(line)
        sys.stdout.flush()


class Worker:
    _TASK_CONTROL_POLL_INTERVAL = 5
    _LOOP_ERROR_SLEEP_SECS = 10

    def __init__(
        self,
        tpu_name: str,
        accelerator: str,
        zone: str,
        tpu_version: str = DEFAULT_TPU_VERSION,
        pricing: Pricing = "spot",
        allocation_mode: AllocationMode = "tpu-vm",
        startup_script: str | None = None,
        debug: bool = False,
    ):
        self.worker_id = tpu_name
        self.accelerator = accelerator
        self.zone = zone
        self.debug = debug
        self.startup_script = startup_script
        self.tpu = TPU(
            tpu_name,
            zone,
            accelerator,
            tpu_version,
            pricing,
            allocation_mode,
        )
        self.queue = Queue()
        self._state_dir = jobman_dir()
        self._task_log_root = os.path.join(jobman_log_dir(), "tasks")
        self._log_dir = os.path.join(jobman_log_dir(), "workers", tpu_name)
        self._timeline_path = os.path.join(self._log_dir, "timeline.jsonl")
        self._bootstrap_complete = False
        os.makedirs(self._log_dir, exist_ok=True)
        if self.startup_script:
            self._snapshot_startup_script()

        # Attach file handler to root jobman logger so all submodules
        # (tpu, queue, worker) log to the same file.
        log_file = None if debug else os.path.join(self._log_dir, "worker.log")
        level = logging.DEBUG if debug else logging.INFO
        get_logger("jobman", log_file=log_file, level=level, file_mode="w")
        if debug:
            # Module-level loggers were created before debug mode was known;
            # update them so DEBUG messages are emitted.
            for name in list(logging.Logger.manager.loggerDict):
                if name.startswith("jobman"):
                    logging.getLogger(name).setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _snapshot_startup_script(self) -> None:
        src = self.startup_script
        if not src:
            return
        dst = os.path.join(self._log_dir, os.path.basename(src))
        if os.path.exists(dst):
            return
        try:
            shutil.copy2(src, dst)
        except OSError:
            logger.warning("Failed to snapshot startup script %s into %s", src, dst, exc_info=True)

    def run(self) -> None:
        self._register(status="running")
        self._record_timeline("worker_started",
                              accelerator=self.accelerator,
                              zone=self.zone,
                              pricing=self.tpu.pricing,
                              allocation_mode=self.tpu.mode)
        logger.info("Worker %s started (accelerator=%s, zone=%s)",
                    self.worker_id, self.accelerator, self.zone)
        try:
            while True:
                task: dict | None = None
                try:
                    self._ensure_tpu_ready()
                    bootstrap_success, bootstrap_preempted = self._ensure_bootstrap_ready()
                    if not bootstrap_success:
                        if bootstrap_preempted:
                            self._handle_preemption()
                        else:
                            logger.warning("Worker bootstrap failed on TPU %s; retrying in 30s",
                                           self.worker_id)
                            time.sleep(30)
                        continue
                    task = self.queue.claim(self.accelerator, self.zone, self.worker_id)
                    if task is None:
                        logger.debug("No pending tasks, sleeping 30s...")
                        time.sleep(30)
                        continue

                    self._send_task_notification(task, "BEGIN")
                    outcome, preempted = self._run_task(task)

                    if outcome == "deleted":
                        logger.info("Task %s was deleted while running; skipping queue release", task["id"])
                        continue
                    if outcome == "paused":
                        logger.info("Task %s was paused while running; leaving it paused", task["id"])
                        continue
                    if preempted:
                        self.queue.release(task["id"], "interrupted")
                        self._handle_preemption()
                    else:
                        final_task = self.queue.release(task["id"], "done" if outcome == "done" else "failed")
                        if outcome == "done":
                            self._send_task_notification(final_task or task, "END")
                        elif final_task is not None and final_task.get("status") == "failed":
                            self._send_task_notification(final_task, "FAIL")

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.exception("Unhandled error in worker loop: %s", e)
                    self._recover_after_loop_exception(task, e)
                    time.sleep(self._LOOP_ERROR_SLEEP_SECS)
        except KeyboardInterrupt:
            logger.info("Worker %s shutting down", self.worker_id)
        finally:
            self._record_timeline("worker_stopped")
            self._register(status="stopped")

    # ------------------------------------------------------------------
    # TPU management
    # ------------------------------------------------------------------

    def _recover_after_loop_exception(self, task: dict | None, exc: Exception) -> None:
        """Best-effort recovery after an unexpected loop exception."""
        self._record_timeline("worker_loop_exception", error=str(exc))
        if task is None:
            return

        task_id = task["id"]
        try:
            if self._is_preempted():
                self.queue.release(task_id, "interrupted")
                self._handle_preemption()
            else:
                final_task = self.queue.release(task_id, "failed")
                if final_task is not None and final_task.get("status") == "failed":
                    self._send_task_notification(final_task, "FAIL")
        except Exception:
            logger.exception("Failed to recover task %s after loop exception", task_id)

    def _ensure_tpu_ready(self) -> None:
        status = self.tpu.status()
        if status == "READY":
            return
        self._record_timeline("tpu_status", status=status)
        logger.info("TPU %s status=%s, provisioning...", self.worker_id, status)
        if status not in ("NOT_FOUND", "CREATING"):
            self._record_timeline("tpu_delete_requested", reason=f"status={status}")
            self.tpu.delete()
            self._record_timeline("tpu_deleted", reason=f"status={status}")
        if status != "CREATING":
            self._record_timeline("tpu_requesting")
            self.tpu.request()
        self.tpu.wait_ready()
        self._bootstrap_complete = False
        self._record_timeline("tpu_ready")

    def _handle_preemption(self) -> None:
        logger.info("Handling preemption for TPU %s", self.worker_id)
        self._bootstrap_complete = False
        try:
            self._record_timeline("tpu_delete_requested", reason="preemption_recovery")
            self.tpu.delete()
            self._record_timeline("tpu_deleted", reason="preemption_recovery")
        except Exception as e:
            logger.warning("Error deleting preempted TPU: %s", e)
            self._record_timeline("tpu_delete_failed", reason="preemption_recovery", error=str(e))

    def _scp_cmd(self, local_path: str, remote_path: str, worker_index: int) -> list[str]:
        return [
            "gcloud", "compute", "tpus", "tpu-vm", "scp",
            local_path,
            f"{self.worker_id}:{remote_path}",
            f"--zone={self.zone}",
            f"--worker={worker_index}",
            "--scp-flag=-o ConnectionAttempts=2",
            "--scp-flag=-o ConnectTimeout=30",
            "--scp-flag=-o StrictHostKeyChecking=no",
            "--scp-flag=-o UserKnownHostsFile=/dev/null",
            "--scp-flag=-o BatchMode=yes",
        ]

    def _ssh_cmd(self, worker_index: int, inline: str) -> list[str]:
        return [
            "gcloud", "compute", "tpus", "tpu-vm", "ssh", self.worker_id,
            f"--zone={self.zone}",
            f"--worker={worker_index}",
            "--ssh-flag=-o ConnectionAttempts=2",
            "--ssh-flag=-o ConnectTimeout=30",
            "--ssh-flag=-o StrictHostKeyChecking=no",
            "--ssh-flag=-o UserKnownHostsFile=/dev/null",
            "--ssh-flag=-o BatchMode=yes",
            "--command", inline,
        ]

    def _format_cmd(self, cmd: list[str]) -> str:
        return shlex.join(cmd)

    def _extract_process_error(self, result: subprocess.CompletedProcess, fallback: str) -> str:
        parts = []
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if stderr:
            parts.append(f"stderr: {stderr}")
        if stdout:
            parts.append(f"stdout: {stdout}")
        return "\n".join(parts) if parts else fallback

    def _log_gcloud_failure(
        self,
        *,
        op: str,
        cmd: list[str],
        returncode: int,
        detail: str,
        phase: str,
        worker_index: int,
        log_handle=None,
    ) -> None:
        cmd_str = self._format_cmd(cmd)
        message = (
            f"{op} failed on worker {worker_index} with exit code {returncode}\n"
            f"command: {cmd_str}\n"
            f"{detail}"
        )
        logger.warning("%s", message)
        self._record_timeline(
            f"{op.lower()}_failed",
            phase=phase,
            worker_index=worker_index,
            returncode=returncode,
            error=detail,
            command=cmd_str,
        )
        if log_handle is not None:
            log_handle.write(f"=== {op} failed on worker {worker_index} ===\n")
            log_handle.write(f"command: {cmd_str}\n")
            log_handle.write(detail)
            if not detail.endswith("\n"):
                log_handle.write("\n")
            log_handle.flush()

    def _run_gcloud_scp(
        self,
        local_path: str,
        remote_path: str,
        worker_index: int,
        *,
        phase: str,
        log_handle=None,
    ) -> subprocess.CompletedProcess:
        cmd = self._scp_cmd(local_path, remote_path, worker_index)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            detail = self._extract_process_error(result, "gcloud scp failed")
            self._log_gcloud_failure(
                op="SCP",
                cmd=cmd,
                returncode=result.returncode,
                detail=detail,
                phase=phase,
                worker_index=worker_index,
                log_handle=log_handle,
            )
        return result

    def _run_gcloud_ssh(
        self,
        worker_index: int,
        inline: str,
        *,
        phase: str,
        log_handle=None,
    ) -> subprocess.CompletedProcess:
        cmd = self._ssh_cmd(worker_index, inline)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            detail = self._extract_process_error(result, "gcloud ssh failed")
            self._log_gcloud_failure(
                op="SSH",
                cmd=cmd,
                returncode=result.returncode,
                detail=detail,
                phase=phase,
                worker_index=worker_index,
                log_handle=log_handle,
            )
        return result

    def _popen_gcloud_ssh(self, worker_index: int, inline: str, **kwargs) -> subprocess.Popen:
        cmd = self._ssh_cmd(worker_index, inline)
        logger.info("Running SSH command on worker %d: %s", worker_index, self._format_cmd(cmd))
        return subprocess.Popen(cmd, **kwargs)

    def _ensure_bootstrap_ready(self) -> tuple[bool, bool]:
        if self._bootstrap_complete or not self.startup_script:
            return True, False
        success, preempted = self._run_bootstrap(self.tpu.get_num_workers())
        if success:
            self._bootstrap_complete = True
            self._record_timeline("bootstrap_succeeded", script=self.startup_script)
            logger.info("Worker bootstrap completed for TPU %s", self.worker_id)
        else:
            self._record_timeline("bootstrap_failed",
                                  script=self.startup_script,
                                  preempted=preempted)
        return success, preempted

    def _run_bootstrap(self, num_workers: int) -> tuple[bool, bool]:
        if not self.startup_script:
            return True, False

        remote_setup = "/tmp/jobman_worker_setup.sh"
        self._record_timeline("bootstrap_started",
                              script=self.startup_script,
                              num_workers=num_workers)
        logger.info("Running worker bootstrap on %d TPU hosts: %s", num_workers, self.startup_script)

        lf = None
        if not self.debug:
            bootstrap_log = os.path.join(self._log_dir, "bootstrap.log")
            lf = open(bootstrap_log, "w")
            lf.write(
                f"=== Bootstrap started at {_now()} ===\n"
                f"Script: {self.startup_script}\n"
                f"TPU: {self.worker_id} ({self.zone})\n\n"
            )
            lf.flush()
        else:
            print(f"=== Bootstrap phase: {self.startup_script} ===")

        results: list[tuple[int, bool, str | None]] = []
        preempted = False

        def record_bootstrap_result(worker_index: int, ok: bool, error_text: str | None = None) -> None:
            status = "ok" if ok else "failed"
            if lf is not None:
                lf.write(f"worker {worker_index}: {status}\n")
                if error_text:
                    lf.write(f"  detail: {error_text}\n")
                lf.flush()
            elif self.debug:
                print(f"worker {worker_index}: {status}")
                if error_text:
                    print(f"  detail: {error_text}")

        try:
            for worker_index in range(num_workers):
                scp_result = self._run_gcloud_scp(
                    self.startup_script,
                    remote_setup,
                    worker_index,
                    phase="bootstrap",
                    log_handle=lf,
                )
                if scp_result.returncode != 0:
                    stderr = self._extract_process_error(scp_result, "SCP failed")
                    results.append((worker_index, False, stderr))
                    record_bootstrap_result(worker_index, False, stderr)
                    preempted = preempted or self._is_preempted()
                    continue

                setup_inline = f"chmod +x {remote_setup} && bash {remote_setup}"
                if worker_index == 0:
                    if self.debug:
                        proc = self._popen_gcloud_ssh(
                            worker_index, setup_inline,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                        )
                        tee_thread = threading.Thread(target=_tee_to_stdout, args=(proc,), daemon=True)
                        tee_thread.start()
                        exit_code = proc.wait()
                        tee_thread.join()
                        error_text = "See console output above for merged gcloud ssh output." if exit_code != 0 else None
                    else:
                        proc = self._popen_gcloud_ssh(
                            worker_index, setup_inline,
                            stdout=lf,
                            stderr=subprocess.STDOUT,
                            text=True,
                        )
                        exit_code = proc.wait()
                        error_text = "See bootstrap.log above for merged gcloud ssh output." if exit_code != 0 else None
                else:
                    result = self._run_gcloud_ssh(
                        worker_index,
                        setup_inline,
                        phase="bootstrap",
                        log_handle=lf,
                    )
                    exit_code = result.returncode
                    error_text = self._extract_process_error(result, "gcloud ssh failed") if exit_code != 0 else None

                if exit_code != 0:
                    preempted = preempted or (exit_code == 255 and self._is_preempted())
                    results.append((worker_index, False, error_text))
                    record_bootstrap_result(worker_index, False, error_text)
                else:
                    results.append((worker_index, True, None))
                    record_bootstrap_result(worker_index, True, None)

            if lf is not None:
                lf.write("\n=== Bootstrap summary ===\n")
                success = all(ok for _, ok, _ in results)
                lf.write(f"\n=== Bootstrap ended at {_now()}, success={success} ===\n\n")
                lf.flush()
            elif self.debug:
                print("=== Bootstrap summary ===")
                print("")

            success = all(ok for _, ok, _ in results)
            return success, preempted
        finally:
            if lf is not None:
                lf.close()

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    def _send_task_notification(self, task: dict, event: str) -> None:
        mail_user = task.get("mail_user")
        mail_types = {str(v).upper() for v in task.get("mail_types", [])}
        if not mail_user or event not in mail_types:
            return

        subject = f"[jobman-lite] {event} {task['id']} ({task.get('name', task['id'])})"
        body = "\n".join([
            f"event: {event}",
            f"task_id: {task['id']}",
            f"name: {task.get('name', task['id'])}",
            f"status: {task.get('status', '-')}",
            f"worker: {self.worker_id}",
            f"accelerator: {task.get('accelerator', self.accelerator)}",
            f"zone: {task.get('zone', self.zone)}",
            f"run_count: {task.get('run_count', 0)}",
            f"submitted: {task.get('submitted') or '-'}",
            f"started: {task.get('started') or '-'}",
            f"ended: {task.get('ended') or '-'}",
        ])
        try:
            sent = send_brevo_email(
                recipient=mail_user,
                subject=subject,
                text_content=body,
                config_path=task.get("mail_config_path"),
            )
        except Exception as exc:
            logger.warning("Failed to send %s email for task %s: %s", event, task["id"], exc)
            return
        if not sent:
            logger.warning("Email notification %s for task %s was not sent", event, task["id"])

    def _run_task(self, task: dict) -> tuple[str, bool]:
        """Run a task. Returns (outcome, preempted)."""
        task_id = task["id"]
        run_count = task.get("run_count", 1)

        logger.info("Running task %s (%s) on TPU %s", task_id, task["name"], self.worker_id)

        num_workers = self.tpu.get_num_workers()
        script_path = task["script"]
        remote_script = f"/tmp/jobman_{task_id}.sh"
        control_state = self._task_control_state(task_id)
        if control_state is not None:
            return control_state, False
        if self.debug:
            output_buffer: list[str] = []
            scp_result = self._run_gcloud_scp(
                script_path, remote_script, 0, phase="task"
            )
            if scp_result.returncode != 0:
                logger.error("SCP failed for task %s: %s", task_id, self._extract_process_error(scp_result, "SCP failed"))
                if self._is_preempted():
                    return "failed", True
                return "failed", False

            env_str = (
                f"JOBMAN_TPU_NAME={self.worker_id} "
                f"JOBMAN_ZONE={self.zone} "
                f"JOBMAN_NUM_WORKERS={num_workers}"
            )
            control_state = self._task_control_state(task_id)
            if control_state is not None:
                return control_state, False
            inline = f"chmod +x {remote_script} && {env_str} bash {remote_script}"
            proc = self._popen_gcloud_ssh(
                0, inline,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            tee_thread = threading.Thread(target=_tee_to_stdout, args=(proc, output_buffer), daemon=True)
            tee_thread.start()
            exit_code, control_state = self._wait_for_task_process(proc, task_id)
            tee_thread.join()
            if control_state is not None:
                return control_state, False
        else:
            log_dir = os.path.join(self._task_log_root, task_id)
            os.makedirs(log_dir, exist_ok=True)
            worker_suffix = self.worker_id.split("-")[-1]  # e.g. "00001" from "v4-8-us-central2-b-00001"
            log_file = os.path.join(log_dir, f"run_{run_count}_worker_{worker_suffix}.log")
            with open(log_file, "w") as lf:
                lf.write(
                    f"=== Task {task_id} started at {_now()} ===\n"
                    f"Script: {script_path}\n"
                    f"TPU: {self.worker_id} ({self.zone})\n\n"
                )
                lf.flush()
                scp_result = self._run_gcloud_scp(
                    script_path,
                    remote_script,
                    0,
                    phase="task",
                    log_handle=lf,
                )
                if scp_result.returncode != 0:
                    detail = self._extract_process_error(scp_result, "SCP failed")
                    logger.error("SCP failed for task %s: %s", task_id, detail)
                    lf.write(f"\n=== Task {task_id} ended at {_now()}, exit_code={scp_result.returncode} ===\n")
                    if self._is_preempted():
                        return "failed", True
                    return "failed", False

                env_str = (
                    f"JOBMAN_TPU_NAME={self.worker_id} "
                    f"JOBMAN_ZONE={self.zone} "
                    f"JOBMAN_NUM_WORKERS={num_workers}"
                )
                control_state = self._task_control_state(task_id)
                if control_state is not None:
                    lf.write(f"=== Task {task_id} externally {control_state} before remote execution ===\n")
                    return control_state, False
                inline = f"chmod +x {remote_script} && {env_str} bash {remote_script}"
                proc = self._popen_gcloud_ssh(
                    0, inline,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                exit_code, control_state = self._wait_for_task_process(proc, task_id)
                lf.write(f"\n=== Task {task_id} ended at {_now()}, exit_code={exit_code} ===\n")
                if control_state is not None:
                    lf.write(f"=== Task {task_id} externally {control_state} during execution ===\n")
                    return control_state, False

        pattern_detected = False
        if exit_code == 0:
            if self.debug:
                pattern_detected = self._has_retryable_failure_pattern(text="".join(output_buffer))
            else:
                pattern_detected = self._has_retryable_failure_pattern(log_file=log_file)
            if pattern_detected:
                logger.warning("Task %s matched retryable failure pattern despite exit code 0", task_id)
                self._record_timeline("task_retryable_pattern_detected", task_id=task_id)
                return "failed", False

        if exit_code == 255:
            # Possible preemption
            if self._is_preempted():
                logger.warning("Task %s: SSH exit 255 + TPU preempted → interrupted", task_id)
                return "failed", True
        elif exit_code != 0:
            self._record_timeline("ssh_failed",
                                  phase="task",
                                  task_id=task_id,
                                  returncode=exit_code)

        success = exit_code == 0
        if success:
            logger.info("Task %s completed successfully", task_id)
        else:
            logger.warning("Task %s failed with exit code %d", task_id, exit_code)
        return ("done" if success else "failed"), False

    def _task_control_state(self, task_id: str) -> str | None:
        task = self.queue.get(task_id)
        if task is None:
            self._record_timeline("task_deleted_while_running", task_id=task_id)
            return "deleted"
        if task.get("status") == "paused":
            self._record_timeline("task_paused_while_running", task_id=task_id)
            return "paused"
        return None

    def _terminate_task_process(self, proc: subprocess.Popen, task_id: str, action: str) -> int:
        logger.info("Stopping task %s due to %s request", task_id, action)
        self._record_timeline("task_process_stop_requested", task_id=task_id, action=action)
        proc.terminate()
        try:
            return proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            return proc.wait()

    def _wait_for_task_process(self, proc: subprocess.Popen, task_id: str) -> tuple[int, str | None]:
        while True:
            exit_code = proc.poll()
            if exit_code is not None:
                return exit_code, None
            control_state = self._task_control_state(task_id)
            if control_state is not None:
                exit_code = self._terminate_task_process(proc, task_id, control_state)
                return exit_code, control_state
            time.sleep(self._TASK_CONTROL_POLL_INTERVAL)

    def _has_retryable_failure_pattern(self, text: str = "", log_file: str = "") -> bool:
        if text:
            return _RETRYABLE_FAILURE_PATTERN in text
        if log_file and os.path.exists(log_file):
            try:
                with open(log_file) as f:
                    return _RETRYABLE_FAILURE_PATTERN in f.read()
            except OSError:
                logger.warning("Failed to read task log %s for retryable failure detection", log_file)
        return False

    def _is_preempted(self) -> bool:
        status = self.tpu.status()
        if status in ("PREEMPTED", "TERMINATED", "NOT_FOUND", "SUSPENDED", "FAILED"):
            self._record_timeline("tpu_unavailable", status=status)
        return status in ("PREEMPTED", "TERMINATED", "NOT_FOUND", "SUSPENDED", "FAILED")

    def _record_timeline(self, event: str, **fields) -> None:
        entry = {"time": _now(), "event": event, "worker_id": self.worker_id}
        entry.update(fields)
        try:
            os.makedirs(self._log_dir, exist_ok=True)
            with open(self._timeline_path, "a") as f:
                f.write(json.dumps(entry, sort_keys=True) + "\n")
        except OSError:
            logger.debug("Failed to write timeline event %s", event, exc_info=True)

    # ------------------------------------------------------------------
    # Worker registry
    # ------------------------------------------------------------------

    def _register(self, status: str = "running") -> None:
        registry_path = os.path.join(self._state_dir, "workers.json")
        os.makedirs(self._state_dir, exist_ok=True)

        # Read existing
        registry: dict = {}
        if os.path.exists(registry_path):
            try:
                with open(registry_path) as f:
                    registry = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        registry[self.worker_id] = {
            "worker_id": self.worker_id,
            "tpu_name": self.worker_id,
            "accelerator": self.accelerator,
            "zone": self.zone,
            "tpu_version": self.tpu.version,
            "pricing": self.tpu.pricing,
            "mode": self.tpu.mode,
            "startup_script": self.startup_script,
            "status": status,
            "registered": _now(),
            "pid": os.getpid(),
        }

        tmp = registry_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(registry, f, indent=2)
        os.replace(tmp, registry_path)
