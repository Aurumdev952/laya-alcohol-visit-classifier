import unittest
from unittest.mock import patch

from eval.run_eval import DEFAULT_CASES, evaluate, load_cases, score_cases
from eval.repeat_eval import summarize_runs


class EvaluationTests(unittest.TestCase):
    def test_dataset_has_fifty_balanced_cases(self):
        cases = load_cases(DEFAULT_CASES)
        self.assertEqual(len(cases), 50)
        counts = {label: sum(case["label"] == label for case in cases) for label in ("positive", "neutral", "negative")}
        self.assertEqual(counts, {"positive": 17, "neutral": 16, "negative": 17})

    def test_metrics_count_mistakes(self):
        rows = [
            {"id": "1", "expected": "positive", "predicted": "positive"},
            {"id": "2", "expected": "neutral", "predicted": "negative"},
            {"id": "3", "expected": "negative", "predicted": "negative"},
        ]
        report = score_cases(rows)
        self.assertEqual(report["correct"], 2)
        self.assertEqual(report["accuracy"], 0.6667)
        self.assertEqual(report["confusion_matrix"]["neutral"]["negative"], 1)
        self.assertEqual(len(report["mistakes"]), 1)

    def test_runner_requests_all_fifty_cases(self):
        cases = load_cases(DEFAULT_CASES)
        requests = []

        def fake_get_json(url, payload, timeout):
            if url.endswith("/health"):
                return {"status": "ok", "device": "cuda", "gpu": "RTX 5090", "model": "test"}
            requests.append(payload["text"])
            return {
                "sentiment": "neutral",
                "probabilities": {"positive": 0.1, "neutral": 0.8, "negative": 0.1},
            }

        with patch("eval.run_eval.get_json", side_effect=fake_get_json):
            report = evaluate("http://127.0.0.1:8000", cases, timeout=1.0, required_gpu="5090")
        self.assertEqual(len(requests), 50)
        self.assertEqual(requests, [case["text"] for case in cases])
        self.assertEqual(report["total"], 50)
        self.assertEqual(report["correct"], 16)

    def test_repeated_summary_averages_scores_and_detects_label_changes(self):
        runs = [
            {"accuracy": 0.8, "macro_f1": 0.7, "total": 2,
             "cases": [{"id": "a", "predicted": "positive"}, {"id": "b", "predicted": "neutral"}]},
            {"accuracy": 1.0, "macro_f1": 0.9, "total": 2,
             "cases": [{"id": "a", "predicted": "positive"}, {"id": "b", "predicted": "negative"}]},
        ]
        summary = summarize_runs(runs)
        self.assertEqual(summary["mean_accuracy"], 0.9)
        self.assertEqual(summary["accuracy_stddev"], 0.1)
        self.assertEqual(summary["mean_macro_f1"], 0.8)
        self.assertEqual(summary["prediction_changes"], ["b"])


if __name__ == "__main__":
    unittest.main()
