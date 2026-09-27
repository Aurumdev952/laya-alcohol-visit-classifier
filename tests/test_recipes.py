import unittest

from sentiment_api.recipes import RECIPE_QUESTIONS
from sentiment_api.service import GpuUnavailableError, SentimentService, TextTooLongError


class FakeTokenizer:
    mask_token = "[MASK]"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


class FakeAgent:
    def __init__(self):
        self.tok = FakeTokenizer()
        self.device = type("Device", (), {"type": "cuda"})()
        self.calls = []
        self.fallback = False

    def predict(self, state, questions):
        self.calls.append((state, questions))
        if self.fallback:
            self.device.type = "cpu"
        answers = {}
        for name, question in questions.items():
            kind = question["type"]
            if kind == "choice":
                options = list(question["criteria"])
                answers[name] = {
                    "type": "choice",
                    "choice": options[0],
                    "probabilities": {option: 1 / len(options) for option in options},
                }
            elif kind == "score":
                levels = question["criteria"]
                answers[name] = {
                    "type": "score",
                    "score": 1.0,
                    "probabilities": {str(i): 1 / len(levels) for i in range(len(levels))},
                    "legend": {str(i): level for i, level in enumerate(levels)},
                }
            else:
                answers[name] = {"type": "noul", "noul": 0.75}
        return {"answers": answers}


class RecipeServiceTests(unittest.TestCase):
    def setUp(self):
        self.agent = FakeAgent()
        self.service = SentimentService(
            self.agent, 20, "RTX 5090", {name: 20 for name in RECIPE_QUESTIONS}
        )

    def test_every_recipe_returns_its_typed_answers(self):
        for name, questions in RECIPE_QUESTIONS.items():
            with self.subTest(recipe=name):
                result = self.service.predict_recipe(name, {"body": "Example state"})
                self.assertEqual(set(result["answers"]), set(questions))
                self.assertIs(self.agent.calls[-1][1], questions)
        self.assertEqual(len(self.agent.calls), 6)

    def test_rejects_oversize_state_before_inference(self):
        self.service.recipe_token_budgets["rag-filtering"] = 2
        with self.assertRaises(TextTooLongError):
            self.service.predict_recipe(
                "rag-filtering", {"question": "What is the period?", "passage": "Thirty days"}
            )
        self.assertEqual(self.agent.calls, [])

    def test_detects_cuda_fallback(self):
        self.agent.fallback = True
        with self.assertRaises(GpuUnavailableError):
            self.service.predict_recipe("model-routing", {"request": "Explain this"})


if __name__ == "__main__":
    unittest.main()
