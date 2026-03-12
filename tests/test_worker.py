from __future__ import annotations

import logging
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

from jobman.worker import Worker


class FakePopen:
    def __init__(self, cmd, stdout=None, stderr=None, text=None, lines=None, returncode=0):
        self.cmd = cmd
        self.stdout = iter(lines or [])
        self._target = stdout
        self._lines = lines or []
        self.returncode = returncode

    def wait(self):
        if self._target is not None:
            for line in self._lines:
                self._target.write(line)
                self._target.flush()
        return self.returncode


class BlockingFakePopen(FakePopen):
    def __init__(self, *args, release_event: threading.Event, **kwargs):
        super().__init__(*args, **kwargs)
        self._release_event = release_event

    def wait(self):
        self._release_event.wait(timeout=5)
        return super().wait()


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

    def test_ensure_tpu_ready_recreates_preempted_tpu(self):
        worker = self._make_worker()
        worker.tpu = Mock()
        worker.tpu.status.return_value = "PREEMPTED"

        worker._ensure_tpu_ready()

        worker.tpu.delete.assert_called_once()
        worker.tpu.request.assert_called_once()
        worker.tpu.wait_ready.assert_called_once()

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
        ):
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

    def test_has_worker_disconnect_pattern_matches_broken_pipe(self):
        worker = self._make_worker()

        detected = worker._has_worker_disconnect_pattern(
            text="some output\nclient_loop: send disconnect: Broken pipe\nmore output\n"
        )

        self.assertTrue(detected)
