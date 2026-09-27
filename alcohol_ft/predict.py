"""Classify a visit case report with a trained local Laya checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alcohol_ft.task import LABELS, QUESTION, case_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to runs/.../final")
    parser.add_argument("--input", type=Path, required=True, help="one JSON case or JSONL cases")
    parser.add_argument("--review-below", type=float, default=0.8)
    args = parser.parse_args()
    if not 0 <= args.review_below <= 1:
        parser.error("review-below must be between 0 and 1")

    import laya
    import torch
    from laya.common import build_sequence

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    agent = laya.load(args.model, device="cuda")
    if agent.device.type != "cuda":
        raise RuntimeError("model did not load on CUDA")
    q = QUESTION["alcohol_attribution"]
    internal = {"t": "choice", "ins": q["instructions"], "crit": q["criteria"]}
    max_len = agent.cfg.get("max_len", 512)
    empty, _ = build_sequence(agent.tok, "", internal, max_len, agent.cfg.get("head_max_len", 192))
    budget = max_len - len(empty)
    contents = args.input.read_text(encoding="utf-8").strip()
    try:
        parsed = json.loads(contents)
        cases = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        cases = [json.loads(line) for line in contents.splitlines() if line.strip()]
    for case in cases:
        state = case_state(case["encounters"])
        safe_state = state.replace(agent.tok.mask_token, " ") if agent.tok.mask_token else state
        count = len(agent.tok(safe_state, add_special_tokens=False)["input_ids"])
        if count > budget:
            raise ValueError(f"{case.get('case_id', '?')}: {count} tokens exceed {budget}; split or review manually")
        answer = agent.predict(state, QUESTION)["answers"]["alcohol_attribution"]
        if agent.device.type != "cuda":
            raise RuntimeError("Laya fell back to CPU")
        probs = answer["probabilities"]
        if answer["choice"] not in LABELS or set(probs) != set(LABELS):
            raise RuntimeError("unexpected Laya response")
        confidence = float(probs[answer["choice"]])
        print(json.dumps({"case_id": case.get("case_id"), "label": answer["choice"],
                          "probabilities": probs, "confidence": confidence,
                          "review_required": confidence < args.review_below}))


if __name__ == "__main__":
    main()
