from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobman.cli import cli


class CLITests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.state_dir = Path(self.tmpdir.name)
        self.env_patch = patch.dict(os.environ, {"JOBMAN_DIR": str(self.state_dir)})
        self.env_patch.start()
        self.previous_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        self.runner = CliRunner()

    def tearDown(self):
        logging.disable(self.previous_disable)
        self.env_patch.stop()
        self.tmpdir.cleanup()

    def _write_script(self, name: str = "train.sh") -> Path:
        script = self.state_dir / name
        script.write_text(
            "#!/bin/bash\n"
            "#JOBMAN --accelerator=v4-8\n"
            "#JOBMAN --zone=us-central2-b\n"
            "#JOBMAN --max-retries=2\n"
            "echo hello\n"
        )
        return script

    def test_submit_snapshots_script_into_task_dir(self):
        script = self._write_script()

        result = self.runner.invoke(cli, ["task", "submit", str(script)])

        self.assertEqual(result.exit_code, 0, result.output)
        queue_data = json.loads((self.state_dir / "queue.json").read_text())
        self.assertEqual(len(queue_data), 1)
        task = next(iter(queue_data.values()))
        self.assertEqual(task["source_script"], str(script.resolve()))
        self.assertNotEqual(task["script"], task["source_script"])
        snapshot_path = Path(task["script"])
        self.assertTrue(snapshot_path.exists())
        self.assertEqual(snapshot_path.read_text(), script.read_text())

    def test_task_show_hides_source_script(self):
        script = self._write_script()

        submit_result = self.runner.invoke(cli, ["task", "submit", str(script)])
        self.assertEqual(submit_result.exit_code, 0, submit_result.output)

        queue_data = json.loads((self.state_dir / "queue.json").read_text())
        task = next(iter(queue_data.values()))

        show_result = self.runner.invoke(cli, ["task", "show", task["id"]])

        self.assertEqual(show_result.exit_code, 0, show_result.output)
        self.assertIn(f"Script            : {task['script']}", show_result.output)
        self.assertNotIn("Source script", show_result.output)

    def test_analytics_availability_command_exists(self):
        result = self.runner.invoke(cli, ["analytics", "availability"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Availability analytics is not implemented yet.", result.output)

    def test_submit_rejects_invalid_accelerator_in_script(self):
        script = self.state_dir / "bad_accel.sh"
        script.write_text(
            "#!/bin/bash\n"
            "#JOBMAN --accelerator=bad-value\n"
            "#JOBMAN --zone=us-central2-b\n"
            "echo hello\n"
        )

        result = self.runner.invoke(cli, ["task", "submit", str(script)])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid accelerator 'bad-value'", result.output)

    def test_submit_rejects_invalid_zone_in_script(self):
        script = self.state_dir / "bad_zone.sh"
        script.write_text(
            "#!/bin/bash\n"
            "#JOBMAN --accelerator=v4-8\n"
            "#JOBMAN --zone=invalid_zone\n"
            "echo hello\n"
        )

        result = self.runner.invoke(cli, ["task", "submit", str(script)])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid zone 'invalid_zone'", result.output)

    def test_status_rejects_workers_only_and_task_only_together(self):
        result = self.runner.invoke(cli, ["status", "-wo", "-to"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--workers-only and --task-only cannot be used together", result.output)

    def test_worker_start_debug_execs_in_foreground(self):
        with patch("jobman.cli.os.execvp", side_effect=SystemExit(0)) as execvp:
            result = self.runner.invoke(
                cli,
                [
                    "worker",
                    "start",
                    "--accelerator=v4-8",
                    "--zone=us-central2-b",
                    "--tpu-name=test-worker",
                    "--debug",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        execvp.assert_called_once()
        args = execvp.call_args.args[1]
        self.assertIn("--debug", args)
        self.assertIn("--tpu-name=test-worker", args)

    def test_worker_delete_requeues_running_tasks_and_removes_registry(self):
        workers_path = self.state_dir / "workers.json"
        workers_path.write_text(
            json.dumps(
                {
                    "worker-a": {
                        "worker_id": "worker-a",
                        "zone": "us-central2-b",
                        "accelerator": "v4-8",
                        "tpu_version": "tpu-ubuntu2204-base",
                        "pricing": "spot",
                        "mode": "tpu-vm",
                        "status": "running",
                    }
                }
            )
        )
        queue_path = self.state_dir / "queue.json"
        queue_path.write_text(
            json.dumps(
                {
                    "task_x": {
                        "id": "task_x",
                        "name": "x",
                        "script": "/tmp/task_x.sh",
                        "accelerator": "v4-8",
                        "zone": "us-central2-b",
                        "status": "running",
                        "worker_id": "worker-a",
                        "fail_count": 0,
                        "max_retries": 1,
                    }
                }
            )
        )

        with patch("jobman.cli.subprocess.run") as run_mock, patch("jobman.cli.TPU") as tpu_cls:
            run_mock.return_value.returncode = 0
            result = self.runner.invoke(cli, ["worker", "delete", "worker-a"])

        self.assertEqual(result.exit_code, 0, result.output)
        queue_data = json.loads(queue_path.read_text())
        self.assertEqual(queue_data["task_x"]["status"], "pending")
        self.assertIsNone(queue_data["task_x"]["worker_id"])
        workers_data = json.loads(workers_path.read_text())
        self.assertEqual(workers_data, {})
        tpu_cls.return_value.delete.assert_called_once()
