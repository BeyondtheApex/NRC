import copy
import unittest

from probe import apply_result, new_ledger, prepare, result_for, validate_worker_run


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.sid = "probe-test"
        self.sha = "a" * 40
        self.ledger = new_ledger()
        prepare(self.ledger, self.sid, self.sha)
        self.result = result_for(self.sid, self.ledger["scenarios"][self.sid], 123)

    def test_checkpoint_is_sufficient_after_serialization(self):
        import json
        restored = json.loads(json.dumps(self.ledger))
        self.assertEqual(restored["scenarios"][self.sid]["dispatch"]["state"], "pending")
        self.assertEqual(apply_result(restored, self.result), "accepted")

    def test_repeated_prepare_does_not_reset_existing_work(self):
        apply_result(self.ledger, self.result)
        prepare(self.ledger, self.sid, "b" * 40)
        self.assertEqual(self.ledger["scenarios"][self.sid]["source_sha"], self.sha)
        self.assertEqual(self.ledger["scenarios"][self.sid]["phase"], "completed")

    def test_duplicate_result_has_one_completion(self):
        self.assertEqual(apply_result(self.ledger, self.result), "accepted")
        self.assertEqual(apply_result(self.ledger, self.result), "duplicate")
        item = self.ledger["scenarios"][self.sid]
        self.assertEqual(sum(e["type"] == "task.completed" for e in item["events"]), 1)
        self.assertEqual(item["duplicate_deliveries"], 1)

    def test_stale_goal_does_not_complete_task(self):
        self.assertEqual(apply_result(self.ledger, dict(self.result, goal_revision=0)), "rejected")
        self.assertEqual(self.ledger["scenarios"][self.sid]["phase"], "ready")

    def test_wrong_attempt_and_source_are_rejected(self):
        for changes in ({"attempt_id": 2}, {"source_sha": "b" * 40}, {"task_id": "another"}):
            with self.subTest(changes=changes):
                self.assertEqual(apply_result(self.ledger, dict(self.result, **changes)), "rejected")

    def test_wrong_answer_and_non_numeric_answer_are_rejected(self):
        for value in (43, "42", True):
            self.assertEqual(apply_result(self.ledger, dict(self.result, value=value)), "rejected")

    def test_repeated_rejection_does_not_grow_semantic_state(self):
        bad = dict(self.result, goal_revision=0)
        apply_result(self.ledger, bad)
        apply_result(self.ledger, bad)
        self.assertEqual(len(self.ledger["scenarios"][self.sid]["rejected_results"]), 1)

    def test_stale_result_after_completion_does_not_undo_completion(self):
        apply_result(self.ledger, self.result)
        apply_result(self.ledger, dict(self.result, goal_revision=0))
        self.assertEqual(self.ledger["scenarios"][self.sid]["accepted_result"], self.result)

    def test_conflicting_success_is_not_silently_overwritten(self):
        apply_result(self.ledger, self.result)
        with self.assertRaises(ValueError):
            apply_result(self.ledger, dict(self.result, worker_run_id=999))

    def test_result_requires_expected_workflow_commit_and_successful_job(self):
        run = {"path": ".github/workflows/agent-probe-worker.yml", "event": "workflow_dispatch", "head_sha": self.sha}
        jobs = [{"name": "deterministic-worker", "status": "completed", "conclusion": "success"}]
        item = self.ledger["scenarios"][self.sid]
        validate_worker_run(run, jobs, item)
        for bad in (dict(run, head_sha="b" * 40), dict(run, path="other.yml"), dict(run, event="push")):
            with self.assertRaises(ValueError):
                validate_worker_run(bad, jobs, item)
        with self.assertRaises(ValueError):
            validate_worker_run(run, [], item)

    def test_invalid_identifiers_cannot_become_paths(self):
        for sid in ("../main", "probe-a/b", "probe-", "other"):
            with self.assertRaises(ValueError):
                prepare(copy.deepcopy(self.ledger), sid, self.sha)


if __name__ == "__main__":
    unittest.main()

