import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import persistent_operator as op


class PersistentOperatorV2Tests(unittest.TestCase):
    def test_executor_rotates_by_attempt(self):
        self.assertEqual(op.executor_for_attempt(1), "workspace:01-delivery-director")
        self.assertEqual(op.executor_for_attempt(2), "workspace:deus-intus-founder-os")
        self.assertEqual(op.executor_for_attempt(3), "workspace:02-ai-automation-and-integration")

    def test_restart_requeues_lost_inflight_receipt(self):
        submitted = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        mission = {
            "id": "OP-RESTART",
            "runtime": {
                "attempts": 1,
                "progress": [
                    {
                        "type": "EXECUTION_DISPATCHED",
                        "receipt_id": "abcdef123456",
                        "submitted_at": submitted,
                        "worker_instance": "previous-worker",
                    }
                ],
            },
        }
        with (
            patch.object(op, "_receipt_text", return_value=(None, {"status": "running_or_lost"})),
            patch.object(op, "_retry_or_block") as retry,
        ):
            op._poll_execution(mission)

        retry.assert_called_once()
        self.assertIn("worker restarted", retry.call_args.args[1])


if __name__ == "__main__":
    unittest.main()