from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from jobman.availability import compute_availability


class AvailabilityTests(unittest.TestCase):
    def test_compute_availability_tracks_task_runtime_and_wait_states(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            worker_dir = Path(tmpdir) / "logs" / "workers" / "worker-1"
            worker_dir.mkdir(parents=True)
            (worker_dir / "timeline.jsonl").write_text(
                "\n".join(
                    [
                        '{"time":"2026-03-01T00:00:00+00:00","event":"worker_started","worker_id":"worker-1","accelerator":"v4-8","zone":"us-central2-b"}',
                        '{"time":"2026-03-01T00:00:05+00:00","event":"tpu_requesting","worker_id":"worker-1"}',
                        '{"time":"2026-03-01T00:01:00+00:00","event":"tpu_status","worker_id":"worker-1","status":"WAITING_FOR_RESOURCES"}',
                        '{"time":"2026-03-01T00:03:00+00:00","event":"tpu_status","worker_id":"worker-1","status":"PROVISIONING"}',
                        '{"time":"2026-03-01T00:05:00+00:00","event":"tpu_ready","worker_id":"worker-1"}',
                        '{"time":"2026-03-01T00:06:00+00:00","event":"task_started","worker_id":"worker-1","task_id":"t1"}',
                        '{"time":"2026-03-01T00:16:00+00:00","event":"task_completed","worker_id":"worker-1","task_id":"t1"}',
                    ]
                )
                + "\n"
            )

            fake_now = datetime.fromisoformat("2026-03-01T00:20:00+00:00")
            with patch("jobman.availability.jobman_log_dir", return_value=str(Path(tmpdir) / "logs")), patch(
                "jobman.availability.datetime"
            ) as mock_datetime:
                mock_datetime.now.return_value = fake_now
                mock_datetime.fromisoformat.side_effect = datetime.fromisoformat
                stats_by_worker, stats_by_accel = compute_availability()

        worker = stats_by_worker["worker-1"]
        self.assertEqual(worker["tasks"], 1)
        self.assertEqual(worker["phases"]["waiting_for_resources"], 120)
        self.assertEqual(worker["phases"]["provisioning"], 120)
        self.assertEqual(worker["phases"]["running_task"], 600)
        self.assertEqual(worker["phases"]["ready_idle"], 300)
        self.assertIn(("v4-8", "us-central2-b"), stats_by_accel)
