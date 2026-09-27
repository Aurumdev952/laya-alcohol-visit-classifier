"""Laya-backed sentiment inference and checkpoint-aware input validation."""

from __future__ import annotations

import json
from threading import Lock
from typing import Any

from sentiment_api.recipes import RECIPE_QUESTIONS


MODEL_ID = "convaiinnovations/laya"
LABELS = ("positive", "neutral", "negative")
QUESTION = {
    "sentiment": {
        "type": "choice",
        "instructions": "What is the overall sentiment expressed by this text?",
        "criteria": {
            "positive": "mainly favorable, pleased, or approving",
            "neutral": "mainly factual, mixed, or without a clear positive or negative opinion",
            "negative": "mainly unfavorable, disappointed, or disapproving",
        },
    }
}


class TextTooLongError(ValueError):
    """The complete text will not fit in the checkpoint's input window."""


class GpuUnavailableError(RuntimeError):
    """CUDA inference is not available."""


class SentimentService:
    def __init__(
        self,
        agent: Any,
        max_text_tokens: int,
        gpu_name: str,
        recipe_token_budgets: dict[str, int] | None = None,
    ):
        self.agent = agent
        self.max_text_tokens = max_text_tokens
        self.gpu_name = gpu_name
        self.recipe_token_budgets = recipe_token_budgets or {}
        self._lock = Lock()

    def is_ready(self) -> bool:
        return self.agent.device.type == "cuda"

    def predict(self, text: str) -> dict[str, Any]:
        if not text.strip():
            raise ValueError("text must not be empty or whitespace only")

        # Laya replaces literal mask tokens in state text before tokenization.
        mask_token = self.agent.tok.mask_token
        safe_text = text.replace(mask_token, " ") if mask_token else text
        token_count = len(self.agent.tok(safe_text, add_special_tokens=False)["input_ids"])
        if token_count > self.max_text_tokens:
            raise TextTooLongError(
                f"text has {token_count} model tokens; maximum is {self.max_text_tokens}"
            )

        # One worker at a time protects the shared model and makes a CUDA fallback visible.
        with self._lock:
            if not self.is_ready():
                raise GpuUnavailableError("Laya is not running on CUDA")
            result = self.agent.predict(text, QUESTION)
            if not self.is_ready():
                raise GpuUnavailableError("Laya fell back to CPU during inference")

        answer = result["answers"]["sentiment"]
        label = answer["choice"]
        probabilities = answer["probabilities"]
        if label not in LABELS or set(probabilities) != set(LABELS):
            raise RuntimeError("Laya returned an unexpected sentiment response")
        return {
            "sentiment": label,
            "probabilities": {name: float(probabilities[name]) for name in LABELS},
        }

    def predict_recipe(self, recipe: str, state: dict[str, str]) -> dict[str, Any]:
        """Answer a fixed recipe without silently truncating any state field."""
        questions = RECIPE_QUESTIONS[recipe]
        safe_state = json.dumps(state, ensure_ascii=False)
        mask_token = self.agent.tok.mask_token
        if mask_token:
            safe_state = safe_state.replace(mask_token, " ")
        token_count = len(self.agent.tok(safe_state, add_special_tokens=False)["input_ids"])
        budget = self.recipe_token_budgets[recipe]
        if token_count > budget:
            raise TextTooLongError(
                f"state has {token_count} model tokens; maximum for {recipe} is {budget}"
            )

        with self._lock:
            if not self.is_ready():
                raise GpuUnavailableError("Laya is not running on CUDA")
            result = self.agent.predict(state, questions)
            if not self.is_ready():
                raise GpuUnavailableError("Laya fell back to CPU during inference")

        answers = result["answers"]
        if set(answers) != set(questions):
            raise RuntimeError(f"Laya returned incomplete answers for {recipe}")
        for name, question in questions.items():
            answer = answers[name]
            kind = question["type"]
            if kind == "choice" and (
                answer.get("choice") not in question["criteria"]
                or set(answer.get("probabilities", {})) != set(question["criteria"])
            ):
                raise RuntimeError(f"Laya returned an invalid {name} choice")
            if kind == "score" and (
                not isinstance(answer.get("score"), (int, float))
                or set(answer.get("probabilities", {}))
                != {str(index) for index in range(len(question["criteria"]))}
            ):
                raise RuntimeError(f"Laya returned an invalid {name} score")
            if kind == "noul" and (
                not isinstance(answer.get("noul"), (int, float))
                or not 0 <= answer["noul"] <= 1
            ):
                raise RuntimeError(f"Laya returned an invalid {name} probability")
        return {"answers": answers}


def load_service() -> SentimentService:
    """Load the English checkpoint and fail if it cannot stay on CUDA."""
    import torch
    import laya
    from laya.common import build_sequence

    if not torch.cuda.is_available():
        raise GpuUnavailableError(
            "CUDA is unavailable. Install a Blackwell-compatible PyTorch CUDA build "
            "and verify the NVIDIA driver with nvidia-smi."
        )

    agent = laya.load(MODEL_ID, device="cuda")
    if agent.device.type != "cuda":
        raise GpuUnavailableError("Laya loaded the model on CPU instead of CUDA")

    def question_budget(question: dict[str, Any]) -> int:
        internal_question = agent._to_internal(question)
        empty_sequence, _ = build_sequence(
            agent.tok,
            "",
            internal_question,
            max_len=agent.cfg.get("max_len", 512),
            head_max_len=agent.cfg.get("head_max_len", 192),
        )
        return agent.cfg.get("max_len", 512) - len(empty_sequence)

    max_text_tokens = question_budget(QUESTION["sentiment"])
    if max_text_tokens < 1:
        raise RuntimeError("Laya checkpoint has no room for sentiment text")
    recipe_token_budgets = {
        name: min(question_budget(question) for question in questions.values())
        for name, questions in RECIPE_QUESTIONS.items()
    }
    if any(budget < 1 for budget in recipe_token_budgets.values()):
        raise RuntimeError("Laya checkpoint has no room for a recipe state")

    gpu_name = torch.cuda.get_device_name(agent.device)
    return SentimentService(agent, max_text_tokens, gpu_name, recipe_token_budgets)
