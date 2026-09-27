"""Checks for split integrity, label scoring, and deterministic generation."""

import tempfile
import unittest
from pathlib import Path

from alcohol_ft.benchmark import metrics, paired_accuracy_interval
from alcohol_ft.data import validate_splits, write_jsonl
from alcohol_ft.generate import generate_split
from alcohol_ft.task import case_state


class AlcoholDataTests(unittest.TestCase):
    def test_generated_split_balanced_and_reproducible(self):
        first = generate_split("train", 20, 4)
        self.assertEqual(first, generate_split("train", 20, 4))
        self.assertEqual(len(first), 60)
        self.assertEqual({label: sum(row["label"] == label for row in first)
                          for label in ("direct", "indirect", "negative")},
                         {"direct": 20, "indirect": 20, "negative": 20})
        self.assertTrue(any(len(row["encounters"]) > 1 for row in first))

    def test_patient_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path1, path2 = Path(tmp) / "train.jsonl", Path(tmp) / "test.jsonl"
            row = {"case_id": "a", "patient_id": "same", "label": "negative",
                   "encounters": [{"note": "No alcohol use."}]}
            write_jsonl(path1, [row])
            write_jsonl(path2, [{**row, "case_id": "b", "encounters": [{"note": "Seen for rash."}]}])
            with self.assertRaisesRegex(ValueError, "patient_id leakage"):
                validate_splits({"train": path1, "test": path2})

    def test_multiple_visits_same_patient_in_one_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            write_jsonl(path, [
                {"case_id": "a", "patient_id": "p", "label": "negative",
                 "encounters": [{"note": "Seen for rash."}]},
                {"case_id": "b", "patient_id": "p", "label": "negative",
                 "encounters": [{"note": "Seen for ankle pain."}]},
            ])
            self.assertEqual(validate_splits({"train": path})["train"]["total"], 2)

    def test_state_keeps_encounter_boundaries(self):
        state = case_state([{"type": "ED", "note": "Denied drinking."},
                            {"type": "consult", "note": "Later admitted wine."}])
        self.assertIn("Encounter 2 (consult):", state)
        self.assertIn("Later admitted wine.", state)


class AlcoholMetricsTests(unittest.TestCase):
    def test_confusion_and_brier(self):
        rows = [
            {"case_id": "a", "scenario": "one", "expected": "direct", "predicted": "direct",
             "probabilities": {"direct": 0.8, "indirect": 0.1, "negative": 0.1}},
            {"case_id": "b", "scenario": "two", "expected": "indirect", "predicted": "negative",
             "probabilities": {"direct": 0.1, "indirect": 0.2, "negative": 0.7}},
        ]
        report = metrics(rows)
        self.assertEqual(report["accuracy"], 0.5)
        self.assertEqual(report["indirect_missed"], 1)
        self.assertEqual(report["confusion_matrix"]["indirect"]["negative"], 1)
        self.assertGreater(report["brier"], 0)
        self.assertEqual(paired_accuracy_interval(rows, rows)["delta"], 0)


if __name__ == "__main__":
    unittest.main()
