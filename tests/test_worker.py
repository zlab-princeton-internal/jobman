from __future__ import annotations

import json
import logging
import multiprocessing
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobman.worker import Worker, _MidTaskHealthMonitor


def _register_worker_once(state_dir: str, worker_id: str) -> None:
    os.environ["JOBMAN_DIR"] = state_dir
    worker = Worker(
        tpu_name=worker_id,
        accelerator="v4-8",
        zone="us-central2-b",
        allocation_mode="tpu-vm",
    )
    worker._register()


class FakePopen:
    def __init__(self, cmd, stdout=None, stderr=None, text=None, lines=None, returncode=0):
        self.cmd = cmd
        self.stdout = iter(lines or [])
        self._target = stdout
        self._lines = lines or []
        self.returncode = returncode
        self._terminated = False
        self._killed = False
        self._poll_count = 0

    def wait(self, timeout=None):
        if self._target is not None:
            for line in self._lines:
                self._target.write(line)
                self._target.flush()
        return self.returncode

    def poll(self):
        if self._terminated or self._killed:
            return self.returncode
        self._poll_count += 1
        return self.returncode if self._poll_count > 1 else None

    def terminate(self):
        self._terminated = True

    def kill(self):
        self._killed = True


class BlockingFakePopen(FakePopen):
    def __init__(self, *args, release_event: threading.Event, **kwargs):
        super().__init__(*args, **kwargs)
        self._release_event = release_event

    def wait(self, timeout=None):
        self._release_event.wait(timeout=5)
        return super().wait(timeout=timeout)


class SilentFakePopen(FakePopen):
    def __init__(self, *args, returncode=124, **kwargs):
        super().__init__(*args, lines=[], returncode=returncode, **kwargs)

    def poll(self):
        if self._terminated or self._killed:
            return self.returncode
        return None


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.state_dir = Path(self.tmpdir.name)
        self.env_patch = patch.dict(os.environ, {"JOBMAN_DIR": str(self.state_dir)})
        self.env_patch.start()
        self.previous_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(self.previous_disable)
        self.env_patch.stop()
        self.tmpdir.cleanup()

    def _make_worker(self, debug: bool = False, startup_script: str | None = None) -> Worker:
        return Worker(
            tpu_name="v4-8-us-central2-b-00001",
            accelerator="v4-8",
            zone="us-central2-b",
            allocation_mode="tpu-vm",
            startup_script=startup_script,
            debug=debug,
        )

    def test_ensure_tpu_ready_noop_when_ready(self):
        worker = self._make_worker()
        worker.tpu = Mock()
        worker.tpu.status.return_value = "READY"

        worker._ensure_tpu_ready()

        worker.tpu.request.assert_not_called()
        worker.tpu.wait_ready.assert_not_called()
        worker.tpu.delete.assert_not_called()

    def test_ensure_tpu_ready_requests_missing_tpu(self):
        worker = self._make_worker()
        worker.tpu = Mock()
        worker.tpu.status.return_value = "NOT_FOUND"

        worker._ensure_tpu_ready()

        worker.tpu.request.assert_called_once()
        worker.tpu.wait_ready.assert_called_once()
        worker.tpu.delete.assert_not_called()

    def test_ensure_tpu_ready_recreates_preempted_tpu(self):
        worker = self._make_worker()
        worker.tpu = Mock()
        # First call returns PREEMPTED (triggers delete+request),
        # second call (post-delete verification) returns NOT_FOUND.
        worker.tpu.status.side_effect = ["PREEMPTED", "NOT_FOUND"]

        worker._ensure_tpu_ready()

        worker.tpu.delete.assert_called_once()
        worker.tpu.request.assert_called_once()
        worker.tpu.wait_ready.assert_called_once()

    def test_ensure_tpu_ready_recreates_unhealthy_tpu(self):
        worker = self._make_worker()
        worker.tpu = Mock()
        worker.tpu.status.side_effect = ["UNHEALTHY", "NOT_FOUND"]

        worker._ensure_tpu_ready()

        worker.tpu.delete.assert_called_once()
        worker.tpu.request.assert_called_once()
        worker.tpu.wait_ready.assert_called_once()

    def test_ensure_tpu_ready_recreates_when_wait_ready_turns_unhealthy(self):
        worker = self._make_worker()
        worker.tpu = Mock()
        worker.tpu.status.side_effect = ["PROVISIONING", "NOT_FOUND"]
        worker.tpu.wait_ready.side_effect = [
            RuntimeError("TPU v4-8-us-central2-b-00001 entered unhealthy state: maintenance event"),
            None,
        ]

        worker._ensure_tpu_ready()

        self.assertEqual(worker.tpu.delete.call_count, 1)
        self.assertEqual(worker.tpu.request.call_count, 1)
        self.assertEqual(worker.tpu.wait_ready.call_count, 2)

    def test_run_records_task_lifecycle_events(self):
        worker = self._make_worker()
        task = {"id": "task-1", "name": "demo-task"}

        worker._ensure_tpu_ready = Mock()
        worker._ensure_bootstrap_ready = Mock(return_value=(True, False))
        worker._check_host_health = Mock(return_value=True)
        worker._run_task = Mock(return_value=("done", None))
        worker._send_task_notification = Mock()
        worker._register = Mock()
        worker.queue = Mock()
        worker.queue.claim.side_effect = [task, KeyboardInterrupt()]
        worker.queue.release.return_value = task

        worker.run()

        timeline = Path(worker._timeline_path).read_text().splitlines()
        self.assertTrue(any('"event": "task_started"' in line for line in timeline))
        self.assertTrue(any('"event": "task_completed"' in line for line in timeline))

    def test_run_refreshes_host_health_before_claiming_tasks(self):
        worker = self._make_worker()

        worker._ensure_tpu_ready = Mock()
        worker._ensure_bootstrap_ready = Mock(return_value=(True, False))
        worker._check_host_health = Mock(return_value=False)
        worker._recreate_tpu = Mock(side_effect=KeyboardInterrupt())
        worker._register = Mock()
        worker.queue = Mock()

        worker.run()

        worker.queue.claim.assert_not_called()
        worker._recreate_tpu.assert_called_once_with(reason="host_health_check_failed")

    def test_register_preserves_all_workers_across_processes(self):
        ctx = multiprocessing.get_context("fork")
        worker_ids = [f"worker-{idx}" for idx in range(6)]
        processes = [
            ctx.Process(target=_register_worker_once, args=(str(self.state_dir), worker_id))
            for worker_id in worker_ids
        ]

        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=10)
            self.assertEqual(process.exitcode, 0)

        registry = json.loads((self.state_dir / "workers.json").read_text())
        self.assertEqual(set(registry), set(worker_ids))

    def test_run_bootstrap_logs_summary(self):
        setup_script = self.state_dir / "setup.sh"
        setup_script.write_text("#!/bin/bash\necho setup\n")
        worker = self._make_worker(startup_script=str(setup_script))
        worker.tpu = Mock()
        worker.tpu.status.return_value = "READY"

        def run_side_effect(cmd, capture_output=True, text=True):
            command = " ".join(cmd)
            if " --command true" in command:
                return SimpleNamespace(returncode=0, stderr="", stdout="")
            if "scp" in command:
                return SimpleNamespace(returncode=0, stderr="", stdout="")
            if "--worker=1" in command and "jobman_worker_setup.sh" in command:
                return SimpleNamespace(returncode=0, stderr="", stdout="")
            raise AssertionError(f"unexpected subprocess.run call: {command}")

        def popen_side_effect(cmd, stdout=None, stderr=None, text=None):
            command = " ".join(cmd)
            if "jobman_worker_setup.sh" in command:
                return FakePopen(cmd, stdout=stdout, lines=["setup worker0 ok\n"], returncode=0)
            if "JOBMAN_TPU_NAME=" in command and "jobman_task_123.sh" in command:
                return FakePopen(cmd, stdout=stdout, lines=["main task ok\n"], returncode=0)
            raise AssertionError(f"unexpected subprocess.Popen call: {command}")

        with patch("jobman.worker.subprocess.run", side_effect=run_side_effect), patch(
            "jobman.worker.subprocess.Popen", side_effect=popen_side_effect
        ):
            success, preempted = worker._run_bootstrap(2)

        self.assertTrue(success)
        self.assertFalse(preempted)
        log_file = Path(worker._log_dir) / "bootstrap.log"
        content = log_file.read_text()
        self.assertIn("=== SSH reachability check ===", content)
        self.assertIn("ssh-check worker 0: ok", content)
        self.assertIn("ssh-check worker 1: ok", content)
        self.assertIn(f"Script: {Path(worker._log_dir) / setup_script.name}", content)
        self.assertNotIn(f"Script: {setup_script}", content)
        self.assertIn("=== Bootstrap started", content)
        self.assertIn("setup worker0 ok", content)
        self.assertIn("worker 0: ok", content)
        self.assertIn("worker 1: ok", content)

    def test_run_bootstrap_fails_when_nonzero_host_fails(self):
        setup_script = self.state_dir / "setup.sh"
        setup_script.write_text("#!/bin/bash\necho setup\n")
        worker = self._make_worker(startup_script=str(setup_script))
        worker.tpu = Mock()
        worker.tpu.status.return_value = "READY"

        def run_side_effect(cmd, capture_output=True, text=True):
            command = " ".join(cmd)
            if " --command true" in command:
                return SimpleNamespace(returncode=0, stderr="", stdout="")
            if "scp" in command:
                return SimpleNamespace(returncode=0, stderr="", stdout="")
            if "--worker=1" in command and "jobman_worker_setup.sh" in command:
                return SimpleNamespace(returncode=1, stderr="host1 failed", stdout="")
            raise AssertionError(f"unexpected subprocess.run call: {command}")

        def popen_side_effect(cmd, stdout=None, stderr=None, text=None):
            command = " ".join(cmd)
            if "jobman_worker_setup.sh" in command:
                return FakePopen(cmd, stdout=stdout, lines=["setup worker0 ok\n"], returncode=0)
            raise AssertionError(f"unexpected subprocess.Popen call: {command}")

        with patch("jobman.worker.subprocess.run", side_effect=run_side_effect), patch(
            "jobman.worker.subprocess.Popen",
            side_effect=popen_side_effect,
        ) as popen_mock:
            success, preempted = worker._run_bootstrap(2)

        self.assertFalse(success)
        self.assertFalse(preempted)
        self.assertEqual(popen_mock.call_count, 1)
        log_file = Path(worker._log_dir) / "bootstrap.log"
        content = log_file.read_text()
        self.assertIn("ssh-check worker 0: ok", content)
        self.assertIn("ssh-check worker 1: ok", content)
        self.assertIn("worker 0: ok", content)
        self.assertIn("worker 1: failed", content)
        self.assertIn("detail: stderr: host1 failed", content)

    def test_run_bootstrap_runs_hosts_in_parallel(self):
        setup_script = self.state_dir / "setup.sh"
        setup_script.write_text("#!/bin/bash\necho setup\n")
        worker = self._make_worker(startup_script=str(setup_script))
        worker.tpu = Mock()
        worker.tpu.status.return_value = "READY"

        worker0_release = threading.Event()
        worker1_started = threading.Event()

        def run_side_effect(cmd, capture_output=True, text=True):
            command = " ".join(cmd)
            if " --command true" in command:
                return SimpleNamespace(returncode=0, stderr="", stdout="")
            if "scp" in command:
                return SimpleNamespace(returncode=0, stderr="", stdout="")
            if "--worker=1" in command and "jobman_worker_setup.sh" in command:
                worker1_started.set()
                worker0_release.set()
                return SimpleNamespace(returncode=0, stderr="", stdout="")
            raise AssertionError(f"unexpected subprocess.run call: {command}")

        def popen_side_effect(cmd, stdout=None, stderr=None, text=None):
            command = " ".join(cmd)
            if "--worker=0" in command and "jobman_worker_setup.sh" in command:
                return BlockingFakePopen(
                    cmd,
                    stdout=stdout,
                    lines=["setup worker0 ok\n"],
                    returncode=0,
                    release_event=worker0_release,
                )
            raise AssertionError(f"unexpected subprocess.Popen call: {command}")

        with patch("jobman.worker.subprocess.run", side_effect=run_side_effect), patch(
            "jobman.worker.subprocess.Popen", side_effect=popen_side_effect
        ):
            success, preempted = worker._run_bootstrap(2)

        self.assertTrue(worker1_started.is_set())
        self.assertTrue(success)
        self.assertFalse(preempted)
        log_file = Path(worker._log_dir) / "bootstrap.log"
        content = log_file.read_text()
        self.assertIn("ssh-check worker 0: ok", content)
        self.assertIn("ssh-check worker 1: ok", content)
        self.assertIn("worker 0: ok", content)
        self.assertIn("worker 1: ok", content)

    def test_run_bootstrap_fails_when_ssh_check_fails(self):
        setup_script = self.state_dir / "setup.sh"
        setup_script.write_text("#!/bin/bash\necho setup\n")
        worker = self._make_worker(startup_script=str(setup_script))
        worker.tpu = Mock()
        worker.tpu.status.return_value = "READY"

        def run_side_effect(cmd, capture_output=True, text=True):
            command = " ".join(cmd)
            if "--worker=0" in command and " --command true" in command:
                return SimpleNamespace(returncode=0, stderr="", stdout="")
            if "--worker=1" in command and " --command true" in command:
                return SimpleNamespace(returncode=1, stderr="ssh handshake failed", stdout="")
            raise AssertionError(f"unexpected subprocess.run call: {command}")

        with patch("jobman.worker.subprocess.run", side_effect=run_side_effect), patch(
            "jobman.worker.subprocess.Popen"
        ) as popen_mock:
            success, preempted = worker._run_bootstrap(2)

        self.assertFalse(success)
        self.assertFalse(preempted)
        popen_mock.assert_not_called()
        log_file = Path(worker._log_dir) / "bootstrap.log"
        content = log_file.read_text()
        self.assertIn("ssh-check worker 0: ok", content)
        self.assertIn("ssh-check worker 1: failed", content)
        self.assertIn("detail: stderr: ssh handshake failed", content)
        self.assertNotIn("\nworker 0: ok\n", content)

    def test_run_task_writes_task_log_and_main_output(self):
        worker = self._make_worker()
        worker.tpu = Mock()
        worker.tpu.get_num_workers.return_value = 2
        worker.tpu.status.return_value = "READY"

        task_script = self.state_dir / "train.sh"
        task_script.write_text("#!/bin/bash\necho train\n")
        task = {"id": "task_789", "name": "train", "script": str(task_script), "run_count": 1}

        def run_side_effect(cmd, capture_output=True, text=True):
            command = " ".join(cmd)
            if "scp" in command:
                return SimpleNamespace(returncode=0, stderr="", stdout="")
            raise AssertionError(f"unexpected subprocess.run call: {command}")

        def popen_side_effect(cmd, stdout=None, stderr=None, text=None):
            command = " ".join(cmd)
            if "JOBMAN_TPU_NAME=" in command and "jobman_task_789.sh" in command:
                return FakePopen(cmd, stdout=stdout, lines=["main task ok\n"], returncode=0)
            raise AssertionError(f"unexpected subprocess.Popen call: {command}")

        with patch("jobman.worker.subprocess.run", side_effect=run_side_effect), patch(
            "jobman.worker.subprocess.Popen", side_effect=popen_side_effect
        ), patch.object(worker, "_task_control_state", return_value=None):
            success, preempted = worker._run_task(task)

        self.assertTrue(success)
        self.assertFalse(preempted)
        log_file = Path(worker._task_log_root) / "task_789" / "run_1_worker_00001.log"
        content = log_file.read_text()
        self.assertIn("=== Task task_789 started", content)
        self.assertIn("main task ok", content)

    def test_run_task_treats_detected_disconnect_as_interrupted(self):
        worker = self._make_worker()
        worker.tpu = Mock()
        worker.tpu.get_num_workers.return_value = 2
        worker.tpu.status.return_value = "READY"

        task_script = self.state_dir / "train.sh"
        task_script.write_text("#!/bin/bash\necho train\n")
        task = {"id": "task_disconnect", "name": "train", "script": str(task_script), "run_count": 1}

        def run_side_effect(cmd, capture_output=True, text=True):
            command = " ".join(cmd)
            if "scp" in command:
                return SimpleNamespace(returncode=0, stderr="", stdout="")
            raise AssertionError(f"unexpected subprocess.run call: {command}")

        def popen_side_effect(cmd, stdout=None, stderr=None, text=None):
            command = " ".join(cmd)
            if "JOBMAN_TPU_NAME=" in command and "jobman_task_disconnect.sh" in command:
                return FakePopen(
                    cmd,
                    stdout=stdout,
                    lines=["training output\n"],
                    returncode=255,
                )
            raise AssertionError(f"unexpected subprocess.Popen call: {command}")

        with patch("jobman.worker.subprocess.run", side_effect=run_side_effect), patch(
            "jobman.worker.subprocess.Popen", side_effect=popen_side_effect
        ), patch.object(worker, "_wait_for_task_process", return_value=(255, None)), patch.object(
            worker, "_task_control_state", return_value=None
        ), patch.object(worker, "_has_worker_disconnect_pattern", return_value=True):
            outcome, preempted = worker._run_task(task)

        self.assertEqual(outcome, "failed")
        self.assertTrue(preempted)

    def test_run_task_fails_after_output_inactivity_timeout(self):
        worker = self._make_worker()
        worker.tpu = Mock()
        worker.tpu.get_num_workers.return_value = 2
        worker.tpu.status.return_value = "READY"
        worker._task_inactivity_timeout_secs = 1
        worker._TASK_CONTROL_POLL_INTERVAL = 0

        task_script = self.state_dir / "train.sh"
        task_script.write_text("#!/bin/bash\necho train\n")
        task = {"id": "task_silent", "name": "train", "script": str(task_script), "run_count": 1}

        def run_side_effect(cmd, capture_output=True, text=True):
            command = " ".join(cmd)
            if "scp" in command:
                return SimpleNamespace(returncode=0, stderr="", stdout="")
            raise AssertionError(f"unexpected subprocess.run call: {command}")

        def popen_side_effect(cmd, stdout=None, stderr=None, text=None):
            command = " ".join(cmd)
            if "JOBMAN_TPU_NAME=" in command and "jobman_task_silent.sh" in command:
                return SilentFakePopen(cmd, stdout=stdout, returncode=124)
            raise AssertionError(f"unexpected subprocess.Popen call: {command}")

        monotonic_values = iter([0.0, 2.0])
        timeline = []

        with patch("jobman.worker.subprocess.run", side_effect=run_side_effect), patch(
            "jobman.worker.subprocess.Popen", side_effect=popen_side_effect
        ), patch("jobman.worker.time.monotonic", side_effect=lambda: next(monotonic_values)), patch.object(
            worker, "_task_control_state", return_value=None
        ), patch.object(
            worker, "_record_timeline", side_effect=lambda event, **fields: timeline.append((event, fields))
        ):
            outcome, preempted = worker._run_task(task)

        self.assertEqual(outcome, "failed")
        self.assertEqual(preempted, "task")
        self.assertIn(
            ("task_inactivity_timeout", {"task_id": "task_silent", "timeout_secs": 1}),
            timeline,
        )
        log_file = Path(worker._task_log_root) / "task_silent" / "run_1_worker_00001.log"
        content = log_file.read_text()
        self.assertIn("terminated due to inactivity timeout", content)

    def test_has_worker_disconnect_pattern_matches_broken_pipe(self):
        worker = self._make_worker()

        detected = worker._has_worker_disconnect_pattern(
            text="some output\nclient_loop: send disconnect: Broken pipe\nmore output\n"
        )

        self.assertTrue(detected)

    def test_run_task_exit137_multihost_is_infra(self):
        """Exit 137 on multi-host tasks should be classified as infra failure."""
        worker = self._make_worker()
        worker.tpu = Mock()
        worker.tpu.get_num_workers.return_value = 4
        worker.tpu.status.return_value = "READY"

        task_script = self.state_dir / "train.sh"
        task_script.write_text("#!/bin/bash\necho train\n")
        task = {"id": "task_oom137", "name": "train", "script": str(task_script), "run_count": 1}

        def run_side_effect(cmd, capture_output=True, text=True):
            command = " ".join(cmd)
            if "scp" in command:
                return SimpleNamespace(returncode=0, stderr="", stdout="")
            raise AssertionError(f"unexpected subprocess.run call: {command}")

        def popen_side_effect(cmd, stdout=None, stderr=None, text=None):
            command = " ".join(cmd)
            if "JOBMAN_TPU_NAME=" in command and "jobman_task_oom137.sh" in command:
                return FakePopen(cmd, stdout=stdout, lines=["training...\n"], returncode=137)
            raise AssertionError(f"unexpected subprocess.Popen call: {command}")

        with patch("jobman.worker.subprocess.run", side_effect=run_side_effect), \
             patch("jobman.worker.subprocess.Popen", side_effect=popen_side_effect), \
             patch.object(worker, "_wait_for_task_process", return_value=(137, None)), \
             patch.object(worker, "_task_control_state", return_value=None):
            outcome, failure_reason = worker._run_task(task)

        self.assertEqual(outcome, "failed")
        self.assertEqual(failure_reason, "infra")

    def test_run_task_exit137_singlehost_is_task_failure(self):
        """Exit 137 on single-host tasks should remain a task failure."""
        worker = self._make_worker()
        worker.tpu = Mock()
        worker.tpu.get_num_workers.return_value = 1
        worker.tpu.status.return_value = "READY"

        task_script = self.state_dir / "train.sh"
        task_script.write_text("#!/bin/bash\necho train\n")
        task = {"id": "task_oom137s", "name": "train", "script": str(task_script), "run_count": 1}

        def run_side_effect(cmd, capture_output=True, text=True):
            command = " ".join(cmd)
            if "scp" in command:
                return SimpleNamespace(returncode=0, stderr="", stdout="")
            raise AssertionError(f"unexpected subprocess.run call: {command}")

        def popen_side_effect(cmd, stdout=None, stderr=None, text=None):
            command = " ".join(cmd)
            if "JOBMAN_TPU_NAME=" in command and "jobman_task_oom137s.sh" in command:
                return FakePopen(cmd, stdout=stdout, lines=["training...\n"], returncode=137)
            raise AssertionError(f"unexpected subprocess.Popen call: {command}")

        with patch("jobman.worker.subprocess.run", side_effect=run_side_effect), \
             patch("jobman.worker.subprocess.Popen", side_effect=popen_side_effect), \
             patch.object(worker, "_wait_for_task_process", return_value=(137, None)), \
             patch.object(worker, "_task_control_state", return_value=None):
            outcome, failure_reason = worker._run_task(task)

        self.assertEqual(outcome, "failed")
        self.assertEqual(failure_reason, "task")

    def test_run_task_infra_pattern_in_output_is_infra(self):
        """Known infra failure patterns in output should cause infra classification."""
        worker = self._make_worker()
        worker.tpu = Mock()
        worker.tpu.get_num_workers.return_value = 2
        worker.tpu.status.return_value = "READY"

        task_script = self.state_dir / "train.sh"
        task_script.write_text("#!/bin/bash\necho train\n")
        task = {"id": "task_deadline", "name": "train", "script": str(task_script), "run_count": 1}

        def run_side_effect(cmd, capture_output=True, text=True):
            command = " ".join(cmd)
            if "scp" in command:
                return SimpleNamespace(returncode=0, stderr="", stdout="")
            raise AssertionError(f"unexpected subprocess.run call: {command}")

        def popen_side_effect(cmd, stdout=None, stderr=None, text=None):
            command = " ".join(cmd)
            if "JOBMAN_TPU_NAME=" in command and "jobman_task_deadline.sh" in command:
                return FakePopen(cmd, stdout=stdout, lines=["DEADLINE_EXCEEDED: Barrier timed out\n"], returncode=1)
            raise AssertionError(f"unexpected subprocess.Popen call: {command}")

        with patch("jobman.worker.subprocess.run", side_effect=run_side_effect), \
             patch("jobman.worker.subprocess.Popen", side_effect=popen_side_effect), \
             patch.object(worker, "_wait_for_task_process", return_value=(1, None)), \
             patch.object(worker, "_task_control_state", return_value=None), \
             patch.object(worker, "_has_infra_failure_pattern", return_value=True):
            outcome, failure_reason = worker._run_task(task)

        self.assertEqual(outcome, "failed")
        self.assertEqual(failure_reason, "infra")

    def test_run_task_repeated_exit_code_is_infra(self):
        """A task that fails with the same exit code as a previous run should be systemic infra."""
        worker = self._make_worker()
        worker.tpu = Mock()
        worker.tpu.get_num_workers.return_value = 1
        worker.tpu.status.return_value = "READY"

        task_script = self.state_dir / "train.sh"
        task_script.write_text("#!/bin/bash\necho train\n")
        task = {"id": "task_repeat", "name": "train", "script": str(task_script), "run_count": 2}

        # Seed the exit code history with a previous failure
        ec_dir = Path(worker._task_log_root) / "task_repeat"
        ec_dir.mkdir(parents=True, exist_ok=True)
        (ec_dir / "exit_codes.json").write_text("[1]")

        def run_side_effect(cmd, capture_output=True, text=True):
            command = " ".join(cmd)
            if "scp" in command:
                return SimpleNamespace(returncode=0, stderr="", stdout="")
            raise AssertionError(f"unexpected subprocess.run call: {command}")

        def popen_side_effect(cmd, stdout=None, stderr=None, text=None):
            command = " ".join(cmd)
            if "JOBMAN_TPU_NAME=" in command and "jobman_task_repeat.sh" in command:
                return FakePopen(cmd, stdout=stdout, lines=["training...\n"], returncode=1)
            raise AssertionError(f"unexpected subprocess.Popen call: {command}")

        with patch("jobman.worker.subprocess.run", side_effect=run_side_effect), \
             patch("jobman.worker.subprocess.Popen", side_effect=popen_side_effect), \
             patch.object(worker, "_wait_for_task_process", return_value=(1, None)), \
             patch.object(worker, "_task_control_state", return_value=None):
            outcome, failure_reason = worker._run_task(task)

        self.assertEqual(outcome, "failed")
        self.assertEqual(failure_reason, "infra")

    def test_run_task_first_failure_is_task_not_infra(self):
        """First occurrence of a non-special exit code should be task failure, not infra."""
        worker = self._make_worker()
        worker.tpu = Mock()
        worker.tpu.get_num_workers.return_value = 1
        worker.tpu.status.return_value = "READY"

        task_script = self.state_dir / "train.sh"
        task_script.write_text("#!/bin/bash\necho train\n")
        task = {"id": "task_first", "name": "train", "script": str(task_script), "run_count": 1}

        def run_side_effect(cmd, capture_output=True, text=True):
            command = " ".join(cmd)
            if "scp" in command:
                return SimpleNamespace(returncode=0, stderr="", stdout="")
            raise AssertionError(f"unexpected subprocess.run call: {command}")

        def popen_side_effect(cmd, stdout=None, stderr=None, text=None):
            command = " ".join(cmd)
            if "JOBMAN_TPU_NAME=" in command and "jobman_task_first.sh" in command:
                return FakePopen(cmd, stdout=stdout, lines=["error output\n"], returncode=1)
            raise AssertionError(f"unexpected subprocess.Popen call: {command}")

        with patch("jobman.worker.subprocess.run", side_effect=run_side_effect), \
             patch("jobman.worker.subprocess.Popen", side_effect=popen_side_effect), \
             patch.object(worker, "_wait_for_task_process", return_value=(1, None)), \
             patch.object(worker, "_task_control_state", return_value=None):
            outcome, failure_reason = worker._run_task(task)

        self.assertEqual(outcome, "failed")
        self.assertEqual(failure_reason, "task")

        # Verify exit code was recorded for future runs
        ec_file = Path(worker._task_log_root) / "task_first" / "exit_codes.json"
        self.assertTrue(ec_file.exists())
        codes = json.loads(ec_file.read_text())
        self.assertEqual(codes, [1])

    def test_has_infra_failure_pattern_matches_deadline(self):
        """_has_infra_failure_pattern should detect DEADLINE_EXCEEDED."""
        worker = self._make_worker()
        self.assertTrue(worker._has_infra_failure_pattern(
            text="Error: DEADLINE_EXCEEDED: Barrier timed out\n"
        ))

    def test_has_infra_failure_pattern_matches_oom(self):
        """_has_infra_failure_pattern should detect Out of memory."""
        worker = self._make_worker()
        self.assertTrue(worker._has_infra_failure_pattern(
            text="kernel: Out of memory: Killed process 12345\n"
        ))

    def test_has_infra_failure_pattern_no_match(self):
        """_has_infra_failure_pattern should not match normal task output."""
        worker = self._make_worker()
        self.assertFalse(worker._has_infra_failure_pattern(
            text="Step 100: loss=0.5, accuracy=0.9\n"
        ))

    def test_run_task_midtask_tpu_unhealthy_is_infra(self):
        """Mid-task TPU health degradation should kill the task and classify as infra failure."""
        worker = self._make_worker()
        worker.tpu = Mock()
        worker.tpu.get_num_workers.return_value = 2
        worker.tpu.status.return_value = "READY"
        worker.tpu.health_status.return_value = "UNHEALTHY"
        worker.tpu.health_description.return_value = "maintenance event"
        worker._TASK_CONTROL_POLL_INTERVAL = 0

        task_script = self.state_dir / "train.sh"
        task_script.write_text("#!/bin/bash\necho train\n")
        task = {"id": "task_health1", "name": "train", "script": str(task_script), "run_count": 1}

        def run_side_effect(cmd, **kwargs):
            command = " ".join(cmd)
            if "scp" in command:
                return SimpleNamespace(returncode=0, stderr="", stdout="")
            raise AssertionError(f"unexpected subprocess.run call: {command}")

        def popen_side_effect(cmd, stdout=None, stderr=None, text=None):
            command = " ".join(cmd)
            if "JOBMAN_TPU_NAME=" in command and "jobman_task_health1.sh" in command:
                return SilentFakePopen(cmd, stdout=stdout, returncode=0)
            raise AssertionError(f"unexpected subprocess.Popen call: {command}")

        with patch("jobman.worker.subprocess.run", side_effect=run_side_effect), \
             patch("jobman.worker.subprocess.Popen", side_effect=popen_side_effect), \
             patch.object(_MidTaskHealthMonitor, 'HEALTH_CHECK_INTERVAL_SECS', 0.1), \
             patch.object(worker, "_task_control_state", return_value=None):
            outcome, failure_reason = worker._run_task(task)

        self.assertEqual(outcome, "failed")
        self.assertEqual(failure_reason, "infra")

    def test_run_task_midtask_memory_pressure_is_infra(self):
        """Mid-task host memory pressure should kill the task and classify as infra failure."""
        worker = self._make_worker()
        worker.tpu = Mock()
        worker.tpu.get_num_workers.return_value = 2
        worker.tpu.status.return_value = "READY"
        worker.tpu.health_status.return_value = "HEALTHY"
        worker._TASK_CONTROL_POLL_INTERVAL = 0

        task_script = self.state_dir / "train.sh"
        task_script.write_text("#!/bin/bash\necho train\n")
        task = {"id": "task_memlow", "name": "train", "script": str(task_script), "run_count": 1}

        def run_side_effect(cmd, **kwargs):
            command = " ".join(cmd)
            if "scp" in command:
                return SimpleNamespace(returncode=0, stderr="", stdout="")
            if "MemAvailable" in command:
                return SimpleNamespace(returncode=0, stderr="", stdout="500000\n")
            if "storage.googleapis.com" in command:
                return SimpleNamespace(returncode=0, stderr="", stdout="OK\n")
            raise AssertionError(f"unexpected subprocess.run call: {command}")

        def popen_side_effect(cmd, stdout=None, stderr=None, text=None):
            command = " ".join(cmd)
            if "JOBMAN_TPU_NAME=" in command and "jobman_task_memlow.sh" in command:
                return SilentFakePopen(cmd, stdout=stdout, returncode=0)
            raise AssertionError(f"unexpected subprocess.Popen call: {command}")

        with patch("jobman.worker.subprocess.run", side_effect=run_side_effect), \
             patch("jobman.worker.subprocess.Popen", side_effect=popen_side_effect), \
             patch.object(_MidTaskHealthMonitor, 'HEALTH_CHECK_INTERVAL_SECS', 0.1), \
             patch.object(worker, "_task_control_state", return_value=None):
            outcome, failure_reason = worker._run_task(task)

        self.assertEqual(outcome, "failed")
        self.assertEqual(failure_reason, "infra")

    def test_run_task_midtask_gcs_unreachable_is_infra(self):
        """Mid-task GCS unreachability should kill the task and classify as infra failure."""
        worker = self._make_worker()
        worker.tpu = Mock()
        worker.tpu.get_num_workers.return_value = 2
        worker.tpu.status.return_value = "READY"
        worker.tpu.health_status.return_value = "HEALTHY"
        worker._TASK_CONTROL_POLL_INTERVAL = 0

        task_script = self.state_dir / "train.sh"
        task_script.write_text("#!/bin/bash\necho train\n")
        task = {"id": "task_gcs", "name": "train", "script": str(task_script), "run_count": 1}

        def run_side_effect(cmd, **kwargs):
            command = " ".join(cmd)
            if "scp" in command:
                return SimpleNamespace(returncode=0, stderr="", stdout="")
            if "MemAvailable" in command:
                return SimpleNamespace(returncode=0, stderr="", stdout="8000000\n")
            if "storage.googleapis.com" in command:
                return SimpleNamespace(returncode=1, stderr="Connection refused", stdout="")
            raise AssertionError(f"unexpected subprocess.run call: {command}")

        def popen_side_effect(cmd, stdout=None, stderr=None, text=None):
            command = " ".join(cmd)
            if "JOBMAN_TPU_NAME=" in command and "jobman_task_gcs.sh" in command:
                return SilentFakePopen(cmd, stdout=stdout, returncode=0)
            raise AssertionError(f"unexpected subprocess.Popen call: {command}")

        with patch("jobman.worker.subprocess.run", side_effect=run_side_effect), \
             patch("jobman.worker.subprocess.Popen", side_effect=popen_side_effect), \
             patch.object(_MidTaskHealthMonitor, 'HEALTH_CHECK_INTERVAL_SECS', 0.1), \
             patch.object(worker, "_task_control_state", return_value=None):
            outcome, failure_reason = worker._run_task(task)

        self.assertEqual(outcome, "failed")
        self.assertEqual(failure_reason, "infra")
