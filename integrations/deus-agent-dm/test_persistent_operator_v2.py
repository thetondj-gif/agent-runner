import unittest
import persistent_operator as op

class PersistentOperatorV2Tests(unittest.TestCase):
    def test_executor_rotates_by_attempt(self):
        self.assertEqual(op.executor_for_attempt(1), "workspace:01-delivery-director")
        self.assertEqual(op.executor_for_attempt(2), "workspace:deus-intus-founder-os")
        self.assertEqual(op.executor_for_attempt(3), "workspace:02-ai-automation-and-integration")

if __name__ == "__main__":
    unittest.main()