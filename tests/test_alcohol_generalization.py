"""Verify strict scoring of counterfactual trios."""

import unittest
from collections import Counter
from pathlib import Path

from alcohol_ft.data import read_cases
from eval.alcohol.generalization_eval import group_metrics


class GroupMetricsTests(unittest.TestCase):
    def test_locked_fixture_has_thirty_complete_trios(self):
        cases = read_cases(Path("eval/alcohol/generalization.jsonl"))
        self.assertEqual(len(cases), 90)
        groups = {}
        for case in cases:
            groups.setdefault(case["group_id"], []).append(case["label"])
        self.assertEqual(len(groups), 30)
        self.assertTrue(all(Counter(labels) == Counter({"direct": 1, "indirect": 1, "negative": 1})
                            for labels in groups.values()))

    def test_all_three_must_be_correct(self):
        labels = ("direct", "indirect", "negative")
        cases = [{"case_id": f"g1-{label}", "group_id": "g1"} for label in labels]
        correct = [{"case_id": f"g1-{label}", "expected": label, "predicted": label}
                   for label in labels]
        self.assertEqual(group_metrics(cases, correct)["all_three_correct_rate"], 1.0)
        wrong = [dict(row) for row in correct]
        wrong[0]["predicted"] = "negative"
        scored = group_metrics(cases, wrong)
        self.assertEqual(scored["all_three_correct_rate"], 0.0)
        self.assertEqual(scored["predicted_three_distinct_rate"], 0.0)

    def test_reject_incomplete_trio(self):
        with self.assertRaisesRegex(ValueError, "expected one case per label"):
            group_metrics([{"case_id": "a", "group_id": "g1"}],
                          [{"case_id": "a", "expected": "direct", "predicted": "direct"}])


if __name__ == "__main__":
    unittest.main()
