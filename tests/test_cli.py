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

from jobman.cli import cli, _worker_display_status


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

    def test_analytics_cost_ownership_filter_adds_bq_parameter(self):
        payload = json.dumps(
            [
                {
                    "usage_date": "2026-02-24",
                    "ownership": "mine",
                    "cost": "1.23",
                    "usage_amount": "10",
                    "usage_unit": "bytes",
                    "location": "us-east5",
                    "service": "Cloud Storage",
                    "resource_name": "llm_pruning_us_east5",
                    "sku": "Standard Storage Columbus",
                }
            ]
        )

        with patch("jobman.cli.subprocess.run") as run_mock:
            run_mock.return_value.returncode = 0
            run_mock.return_value.stdout = payload
            run_mock.return_value.stderr = ""

            result = self.runner.invoke(
                cli,
                [
                    "analytics",
                    "cost",
                    "--date=2026-02-24",
                    "--ownership=mine",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        cmd = run_mock.call_args.args[0]
        self.assertIn("--parameter=ownership:STRING:mine", cmd)
        self.assertIn("OWNER", result.output)
        self.assertIn("llm_pruning_us_east5", result.output)

    def test_analytics_egress_unknown_filter_keeps_ownership_column(self):
        payload = json.dumps(
            [
                {
                    "usage_date": "2026-02-24",
                    "ownership": "unknown",
                    "cost": "0.52",
                    "usage_amount": "5042090223",
                    "usage_unit": "bytes",
                    "location": "us-central1",
                    "service": "Compute Engine",
                    "resource_name": "t1v-n-67b5edcd-w-0",
                    "sku": "Network Internet Data Transfer Out from Americas to Americas",
                }
            ]
        )

        with patch("jobman.cli.subprocess.run") as run_mock:
            run_mock.return_value.returncode = 0
            run_mock.return_value.stdout = payload
            run_mock.return_value.stderr = ""

            result = self.runner.invoke(
                cli,
                [
                    "analytics",
                    "egress",
                    "--date=2026-02-24",
                    "--ownership=unknown",
                    "--kind=internet",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        cmd = run_mock.call_args.args[0]
        self.assertIn("--parameter=ownership:STRING:unknown", cmd)
        self.assertIn("unknown", result.output)
        self.assertIn("t1v-n-67b5edcd-w-0", result.output)

    def test_analytics_cost_queries_bigquery_and_renders_table(self):
        payload = [
            {
                "usage_date": "2026-02-24",
                "service": "Compute Engine",
                "sku": "Network Internet Data Transfer Out to Americas",
                "location": "us-east5",
                "project_id": "demo-project",
                "resource_name": "demo-worker",
                "global_name": "",
                "cost": "12.34",
                "usage_amount": "56.78",
                "usage_unit": "GiBy",
            }
        ]

        with patch("jobman.cli.subprocess.run") as run_mock:
            run_mock.return_value.returncode = 0
            run_mock.return_value.stdout = json.dumps(payload)
            run_mock.return_value.stderr = ""

            result = self.runner.invoke(
                cli,
                [
                    "analytics",
                    "cost",
                    "--date=2026-02-24",
                    "--billing-table=billing_export_us.gcp_billing_export_resource_v1_demo",
                    "--network-only",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        invoked = run_mock.call_args.args[0]
        self.assertIn("bq", invoked[0])
        self.assertTrue(
            any(
                arg == "--parameter=target_date:DATE:2026-02-24"
                for arg in invoked
            )
        )
        self.assertIn("Network Internet Data Transfer Out to Americas", result.output)
        self.assertIn("demo-worker", result.output)
        self.assertIn("12.34", result.output)

    def test_analytics_egress_json_output_and_location_filter(self):
        payload = [
            {
                "usage_date": "2026-02-24",
                "service": "Compute Engine",
                "sku": "Network Inter Zone Data Transfer",
                "location": "us-east5",
                "project_id": "demo-project",
                "resource_name": "",
                "global_name": "//tpu.googleapis.com/projects/demo/locations/us-east5-b/nodes/demo-worker",
                "cost": "3.21",
                "usage_amount": "10",
                "usage_unit": "GiBy",
            }
        ]

        with patch("jobman.cli.subprocess.run") as run_mock:
            run_mock.return_value.returncode = 0
            run_mock.return_value.stdout = json.dumps(payload)
            run_mock.return_value.stderr = ""

            result = self.runner.invoke(
                cli,
                [
                    "analytics",
                    "egress",
                    "--date=2026-02-24",
                    "--billing-table=billing_export_us.gcp_billing_export_resource_v1_demo",
                    "--kind=inter-zone",
                    "--location=us-east5",
                    "--output=json",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        invoked = run_mock.call_args.args[0]
        self.assertTrue(
            any(
                arg == "--parameter=location:STRING:us-east5"
                for arg in invoked
            )
        )
        self.assertIn('"sku": "Network Inter Zone Data Transfer"', result.output)
        self.assertIn('"location": "us-east5"', result.output)

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

    def test_worker_display_status_pending_when_tpu_not_active(self):
        status = _worker_display_status(
            {"status": "running"},
            process_status="running",
            vm_status="READY",
            qr_status="WAITING_FOR_RESOURCES",
            has_running_task=False,
        )

        self.assertEqual(status, "pending")

    def test_worker_display_status_setup_during_bootstrap(self):
        status = _worker_display_status(
            {"status": "running"},
            process_status="setup",
            vm_status="READY",
            qr_status="ACTIVE",
            has_running_task=False,
        )

        self.assertEqual(status, "setup")

    def test_worker_display_status_idle_without_running_task(self):
        status = _worker_display_status(
            {"status": "running"},
            process_status="running",
            vm_status="READY",
            qr_status="ACTIVE",
            has_running_task=False,
        )

        self.assertEqual(status, "idle")

    def test_worker_display_status_running_with_running_task(self):
        status = _worker_display_status(
            {"status": "running"},
            process_status="running",
            vm_status="READY",
            qr_status="ACTIVE",
            has_running_task=True,
        )

        self.assertEqual(status, "running")

    def test_worker_display_status_unhealthy_when_health_flags_issue(self):
        status = _worker_display_status(
            {"status": "running"},
            process_status="running",
            vm_status="READY",
            qr_status="ACTIVE",
            health_status="UNHEALTHY",
            has_running_task=False,
        )

        self.assertEqual(status, "unhealthy")

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

    def test_worker_show_hides_internal_only_host0_ip(self):
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

        def run_side_effect(cmd, *args, **kwargs):
            if cmd[:2] == ["tmux", "has-session"]:
                return unittest.mock.Mock(returncode=0, stdout="", stderr="")
            if cmd[:5] == ["gcloud", "compute", "tpus", "tpu-vm", "describe"]:
                return unittest.mock.Mock(
                    returncode=0,
                    stdout=json.dumps({"networkEndpoints": [{"ipAddress": "10.1.2.3"}]}),
                    stderr="",
                )
            if cmd[:4] == ["gcloud", "config", "get-value", "project"]:
                return unittest.mock.Mock(returncode=0, stdout="demo-project\n", stderr="")
            return unittest.mock.Mock(returncode=1, stdout="", stderr="not found")

        with patch("jobman.cli.subprocess.run", side_effect=run_side_effect):
            result = self.runner.invoke(cli, ["worker", "show", "worker-a"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Host0 IP          : -", result.output)

    def test_worker_show_prefers_external_host0_ip(self):
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

        def run_side_effect(cmd, *args, **kwargs):
            if cmd[:2] == ["tmux", "has-session"]:
                return unittest.mock.Mock(returncode=0, stdout="", stderr="")
            if cmd[:5] == ["gcloud", "compute", "tpus", "tpu-vm", "describe"]:
                return unittest.mock.Mock(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "networkEndpoints": [
                                {
                                    "ipAddress": "10.1.2.3",
                                    "accessConfig": {"externalIp": "34.5.6.7"},
                                }
                            ]
                        }
                    ),
                    stderr="",
                )
            if cmd[:4] == ["gcloud", "config", "get-value", "project"]:
                return unittest.mock.Mock(returncode=0, stdout="demo-project\n", stderr="")
            return unittest.mock.Mock(returncode=1, stdout="", stderr="not found")

        with patch("jobman.cli.subprocess.run", side_effect=run_side_effect):
            result = self.runner.invoke(cli, ["worker", "show", "worker-a"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Host0 IP          : 34.5.6.7", result.output)
