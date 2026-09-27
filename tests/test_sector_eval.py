import unittest

from eval.run_eval import score_cases
from eval.sector_eval import load_sector_cases, load_tasks


class SectorEvaluationTests(unittest.TestCase):
    def test_every_sector_has_fifty_balanced_cases(self):
        tasks = load_tasks()
        self.assertEqual(len(tasks), 5)
        for sector, task in tasks.items():
            with self.subTest(sector=sector):
                cases = load_sector_cases(sector, tuple(task["criteria"]))
                self.assertEqual(len(cases), 50)

    def test_metrics_support_five_categories(self):
        labels = ("a", "b", "c", "d", "e")
        rows = [{"expected": label, "predicted": label} for label in labels]
        rows.append({"expected": "a", "predicted": "b"})
        report = score_cases(rows, labels)
        self.assertEqual(report["correct"], 5)
        self.assertEqual(report["confusion_matrix"]["a"]["b"], 1)
        self.assertEqual(report["per_label"]["e"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()

