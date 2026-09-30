import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import persistent_operator as op


class PersistentOperatorV2Tests(unittest.TestCase):
    def test_executor_rotates_by_attempt(self):
        self.assertEqual(op.executor_for_attempt(1), "workspace:01-delivery-director")
        self.assertEqual(op.executor_for_attempt(2), "workspace:deus-intus-founder-os")
        self.assertEqual(op.executor_for_attempt(3), "workspace:02-ai-automation-and-integration")

    def test_parse_marker_tolerates_markdown(self):
        text = """Work done.

**OPERATOR_RESULT**
`{"status":"COMPLETE","summary":"ok","evidence":["x"],"next":""}`
"""
        result = op.parse_marker(text, "OPERATOR_RESULT")
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "COMPLETE")

    def test_prompt_carries_existing_mission_approval(self):
        mission = {
            "id": "OP-APPROVED",
            "objective": "Inspect status only",
            "approval": "GRANTED",
            "runtime": {"acceptance_criteria": ["Return a count"]},
        }
        prompt = op._executor_prompt(mission)
        self.assertIn("MISSION APPROVAL: GRANTED", prompt)
        self.assertIn("Do not ask the user to approve ordinary read-only", prompt)

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