import unittest

from sentiment_api.service import (
    GpuUnavailableError,
    SentimentService,
    TextTooLongError,
)


class FakeTokenizer:
    mask_token = "[MASK]"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


class FakeAgent:
    def __init__(self):
        self.tok = FakeTokenizer()
        self.device = type("Device", (), {"type": "cuda"})()
        self.calls = 0
        self.fallback = False

    def predict(self, text, questions):
        self.calls += 1
        assert questions["sentiment"]["type"] == "choice"
        if self.fallback:
            self.device.type = "cpu"
        return {
            "answers": {
                "sentiment": {
                    "choice": "positive",
                    "probabilities": {
                        "positive": 0.8,
                        "neutral": 0.15,
                        "negative": 0.05,
                    },
                }
            }
        }


class SentimentServiceTests(unittest.TestCase):
    def setUp(self):
        self.agent = FakeAgent()
        self.service = SentimentService(self.agent, max_text_tokens=3, gpu_name="RTX 5090")

    def test_maps_laya_choice_and_probabilities(self):
        result = self.service.predict("Very good product")
        self.assertEqual(result["sentiment"], "positive")
        self.assertEqual(result["probabilities"]["positive"], 0.8)
        self.assertEqual(set(result["probabilities"]), {"positive", "neutral", "negative"})
        self.assertEqual(self.agent.calls, 1)

    def test_rejects_blank_text(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            self.service.predict(" \t ")
        self.assertEqual(self.agent.calls, 0)

    def test_rejects_oversize_without_inference(self):
        with self.assertRaises(TextTooLongError):
            self.service.predict("one two three four")
        self.assertEqual(self.agent.calls, 0)

    def test_detects_cpu_fallback(self):
        self.agent.fallback = True
        with self.assertRaises(GpuUnavailableError):
            self.service.predict("good")
        self.assertFalse(self.service.is_ready())


if __name__ == "__main__":
    unittest.main()

