"""Checks attribution targets, group isolation, and benchmark aggregation."""

import json
import warnings
from pathlib import Path

import pytest

from alcohol_ft.benchmark_v2 import summarize
from alcohol_ft.data import read_cases, validate_splits, write_jsonl
from alcohol_ft.generate_v2 import generate_split
from alcohol_ft.task import LABELS, combine_components, combine_probabilities, target_for
from alcohol_ft.train_classifier import encode, load_config as load_classifier_config
from alcohol_ft.train_v2 import _training_loss, load_config as load_laya_config, prepare_items, visit_predictions
from alcohol_ft.select_v2 import assess


class TinyTokenizer:
    mask_token = "[MASK]"
    mask_token_id = 3
    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        tokens = [10 + sum(map(ord, word)) % 1000 for word in text.split()]
        if kwargs.get("truncation"):
            tokens = tokens[:kwargs["max_length"]]
        return {"input_ids": tokens}


def test_v2_counterfactuals_have_correct_component_targets():
    rows = generate_split("train", 18, 31)
    assert rows == generate_split("train", 18, 31)
    assert len(rows) == 54
    for row in rows:
        annotation = row["annotation"]
        assert combine_components(annotation["patient_evidence"], annotation["other_person_causal"]) == row["label"]
        assert target_for(row, "patient_alcohol") == (
            "documented" if annotation["patient_evidence"] else "not_documented")
        assert target_for(row, "other_person_causal") == (
            "causal" if annotation["other_person_causal"] else "not_causal")
    assert any(r["label"] == "direct" and r["annotation"]["other_person_causal"] for r in rows)
    assert any(r["signal_category"] == "prenatal" for r in rows)
    assert all(r["scenario"] not in {"thigh", "jaw", "neurodevelopment"} for r in rows)
    assert any(r["signal_category"] == "prenatal" and not r["family_seen_in_training"]
               for r in generate_split("test", 6, 31))


def test_group_leakage_and_span_integrity(tmp_path):
    a, b = generate_split("train", 1, 22)[:2]
    p1, p2 = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    write_jsonl(p1, [a])
    write_jsonl(p2, [b])
    with pytest.raises(ValueError, match="group_id leakage"):
        validate_splits({"one": p1, "two": p2})
    broken = json.loads(json.dumps(next(row for row in (a, b) if row["label"] != "negative")))
    broken["annotation"]["evidence"][0]["text"] = "evidence that is not present"
    write_jsonl(p1, [broken])
    with pytest.raises(ValueError, match="evidence span"):
        read_cases(p1)


def test_decomposed_training_and_visit_decision():
    from alcohol_ft.task import case_state

    rows = generate_split("train", 1, 25)
    for row in rows:
        row["state"] = case_state(row["encounters"])
    items = prepare_items(rows, TinyTokenizer(), {"max_len": 1024, "head_max_len": 256}, "decomposed")
    assert len(items) == 6
    assert all(len(item["target"]) == 2 for item in items)
    logits = [[7 if x else -7 for x in item["target"]] for item in items]
    predictions = visit_predictions(items, logits, "decomposed")
    assert {r["expected"]: r["predicted"] for r in predictions} == {label: label for label in LABELS}
    assert combine_probabilities(0.6, 0.8) == pytest.approx(
        {"direct": 0.6, "indirect": 0.32, "negative": 0.08})


def test_complete_group_score_and_negative_false_positive():
    from alcohol_ft.task import case_state

    cases = generate_split("validation", 2, 41)
    for case in cases:
        case["state"] = case_state(case["encounters"])
    predictions = [{"case_id": c["case_id"], "expected": c["label"], "predicted": c["label"],
                    "scenario": c["scenario"],
                    "probabilities": {x: 0.9 if x == c["label"] else 0.05 for x in LABELS}}
                   for c in cases]
    negative = next(r for r in predictions if r["expected"] == "negative")
    negative["predicted"] = "direct"
    negative["probabilities"] = {"direct": 0.9, "indirect": 0.05, "negative": 0.05}
    report = summarize(cases, predictions)
    assert report["counterfactual"]["complete_correct"] == 1
    assert report["negative_to_direct"] == 1


def test_experiment_configs_parse():
    folder = Path(__file__).resolve().parents[1] / "experiments" / "alcohol_v2"
    for path in folder.glob("*.json"):
        cfg = load_classifier_config(path) if path.stem.endswith("_classifier") else load_laya_config(path)
        assert cfg["epochs"] > 0


def test_classifier_refuses_truncation():
    rows = [{"state": "one two three four", "label": "negative", "case_id": "case-1"}]
    with pytest.raises(ValueError, match="exceed"):
        encode(rows, TinyTokenizer(), max_len=3)


def test_selection_requires_negative_recall_without_sacrificing_other_classes():
    def report(direct, indirect, negative):
        return {"split": "validation", "data_dir": "same", "model": "candidate",
                "metrics": {"macro_f1": 0.8, "negative_to_direct": 10,
                            "per_label": {"direct": {"recall": direct},
                                          "indirect": {"recall": indirect},
                                          "negative": {"recall": negative}}}}

    baseline = report(0.94, 0.97, 0.70)
    assert assess(baseline, report(0.93, 0.96, 0.91))["passes_research_targets"]
    assert not assess(baseline, report(0.90, 0.96, 0.91))["passes_research_targets"]
    assert not assess(baseline, report(0.93, 0.96, 0.89))["passes_research_targets"]


@pytest.mark.parametrize("objective", ["ce", "rlcd_ce"])
def test_both_objectives_backpropagate_on_labeled_choices(objective):
    import torch

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.2, -0.2]))

        def forward(self, ids, att, pos, mask, qtype):
            return self.weight.expand(ids.shape[0], -1), self.weight.sum().expand(ids.shape[0], 2)

    model = TinyModel()
    batch = (torch.ones((2, 3), dtype=torch.long), torch.ones((2, 3), dtype=torch.long),
             torch.ones((2, 2), dtype=torch.long), torch.ones((2, 2), dtype=torch.bool),
             torch.tensor([[1., 0.], [0., 1.]]), torch.zeros(2, dtype=torch.long))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loss = _training_loss(model, batch, torch, {"objective": objective,
                                                  "noise_samples": 4, "rl_weight": 1.0}, 0.2)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.weight.grad).all()
