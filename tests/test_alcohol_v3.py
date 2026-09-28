"""Check v3 causal counterfactuals and dataset integrity."""

from collections import Counter, defaultdict
from pathlib import Path

from alcohol_ft.audit_v3 import audit
from alcohol_ft.data import read_cases
from alcohol_ft.generate_v3 import HARD_CONTEXTS, TEST_ONLY_CONTEXTS, generate_split
from alcohol_ft.task import LABELS, combine_components


def test_hard_groups_are_matched_and_have_incidental_negative_mentions():
    rows = generate_split("train", 48, 17)
    assert rows == generate_split("train", 48, 17)
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
        annotation = row["annotation"]
        assert combine_components(annotation["patient_evidence"],
                                  annotation["other_person_causal"]) == row["label"]
    assert len(rows) == 144
    assert Counter(row["label"] for row in rows) == {label: 48 for label in LABELS}
    assert sum(group[0]["hard_group"] for group in groups.values()) == 24
    for group in groups.values():
        assert {row["label"] for row in group} == set(LABELS)
        assert len({row["scenario"] for row in group}) == 1
        if group[0]["hard_group"]:
            negative = next(row for row in group if row["label"] == "negative")
            indirect = next(row for row in group if row["label"] == "indirect")
            assert len(negative["annotation"]["decoys"]) == 1
            assert not negative["annotation"]["evidence"]
            assert len(indirect["annotation"]["evidence"]) == 2


def test_nonprenatal_indirect_contexts_document_impairment():
    for context in (*HARD_CONTEXTS, *TEST_ONLY_CONTEXTS):
        alcohol_note = context[4].lower()
        assert any(word in alcohol_note for word in
                   ("impaired", "intoxicated", "drunk", "unsteady"))


def test_v3_keeps_reserved_families_in_test():
    # The complete 30k corpus is audited in the runbook; a small build checks
    # deterministic split and hard-case logic without training a model.
    rows = generate_split("test", 60, 23)
    assert all(row["scenario"] not in {"thigh", "jaw", "neurodevelopment"}
               for row in rows if row["family_seen_in_training"])
    assert any(not row["family_seen_in_training"] for row in rows)
    assert all(row["render_style"] in (7, 8) for row in rows)


def test_v3_training_reports_do_not_copy_authored_examples():
    states = {row["state"] for row in read_cases(Path("data/alcohol_synthetic_v3/train.jsonl"))}
    for name in ("curated", "generalization"):
        authored = read_cases(Path("eval/alcohol") / f"{name}.jsonl")
        assert states.isdisjoint(row["state"] for row in authored)


def test_v3_full_corpus_manifest_and_hard_coverage():
    report = audit(Path("data/alcohol_synthetic_v3"))
    assert report["manifest_sha256_verified"]
    assert report["splits"]["train"]["hard_groups"] == 3500
    assert report["splits"]["test"]["hard_groups"] == 500
