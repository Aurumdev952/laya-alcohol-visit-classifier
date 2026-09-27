import unittest

from eval.structured_eval import load_cases, load_tasks, score_records


class StructuredEvaluationTests(unittest.TestCase):
    def test_three_tasks_have_fifty_valid_records(self):
        tasks = load_tasks()
        self.assertEqual(len(tasks), 3)
        for name, task in tasks.items():
            with self.subTest(task=name):
                self.assertEqual(len(load_cases(name, task["schema"])), 50)

    def test_exact_and_field_scores(self):
        schema = {"properties": {
            "topic": {"type": "string", "enum": ["a", "b"]},
            "urgency": {"type": "integer", "minimum": 0, "maximum": 2},
            "refund": {"type": "boolean"},
        }}
        rows = [
            {"expected": {"topic": "a", "urgency": 0, "refund": False},
             "predicted": {"topic": "a", "urgency": 0, "refund": False}},
            {"expected": {"topic": "b", "urgency": 2, "refund": True},
             "predicted": {"topic": "b", "urgency": 1, "refund": True}},
        ]
        score = score_records(rows, schema)
        self.assertEqual(score["exact_records"], 1)
        self.assertEqual(score["by_field"]["urgency"]["correct"], 1)
        self.assertEqual(score["field_accuracy"], 0.8333)


if __name__ == "__main__":
    unittest.main()

