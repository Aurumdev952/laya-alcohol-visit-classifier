"""Audit v3 synthetic data, hard-negative coverage, and split separation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from alcohol_ft.data import read_cases, validate_splits
from alcohol_ft.generate_v2 import SEEN_FAMILIES, SPLITS, UNSEEN_FAMILIES
from alcohol_ft.generate_v3 import TEST_ONLY_CONTEXTS
from alcohol_ft.task import LABELS


def audit(data_dir: Path, tokenizer_path: str | None = None) -> dict:
    paths = {part: data_dir / f"{part}.jsonl" for part in SPLITS}
    counts = validate_splits(paths)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 3 or manifest.get("counts") != counts:
        raise ValueError("v3 manifest version/count mismatch")
    tokenizer = None
    if tokenizer_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    report = {"counts": counts, "splits": {}, "manifest_sha256_verified": True}
    known = {family[0] for family in SEEN_FAMILIES}
    reserved = {family[0] for family in UNSEEN_FAMILIES}
    test_only = {context[0] for context in TEST_ONLY_CONTEXTS}
    for part, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"][part]:
            raise ValueError(f"manifest hash mismatch: {part}")
        rows = read_cases(path)
        groups = defaultdict(list)
        for row in rows:
            if row.get("source") != "fully_synthetic_v3":
                raise ValueError(f"incorrect provenance: {row['case_id']}")
            if row.get("hard_group") not in (True, False):
                raise ValueError(f"missing hard-group flag: {row['case_id']}")
            if row["scenario"] in reserved and part != "test":
                raise ValueError("reserved presentation family leaked")
            if row["family_seen_in_training"] != (row["scenario"] in known):
                raise ValueError(f"incorrect seen-family flag: {row['case_id']}")
            if bool(row["annotation"]["evidence"]) != (row["label"] != "negative"):
                raise ValueError(f"incorrect evidence flag: {row['case_id']}")
            groups[row["group_id"]].append(row)
        if len(groups) != SPLITS[part] or Counter(r["label"] for r in rows) != {
                label: SPLITS[part] for label in LABELS}:
            raise ValueError(f"count/class mismatch: {part}")
        for group in groups.values():
            if {r["label"] for r in group} != set(LABELS) or len(group) != 3:
                raise ValueError(f"incomplete counterfactual group: {group[0]['group_id']}")
            if len({r["scenario"] for r in group}) != 1 or len({r["hard_group"] for r in group}) != 1:
                raise ValueError(f"counterfactual context mismatch: {group[0]['group_id']}")
            if group[0]["hard_group"] and (len({r["hard_category"] for r in group}) != 1 or
                                          not next(r for r in group if r["label"] == "negative")[
                                              "annotation"]["decoys"]):
                raise ValueError(f"hard negative missing context: {group[0]['group_id']}")
        hard = [r for r in rows if r["hard_group"] and r["label"] == "negative"]
        if len(hard) != (SPLITS[part] + 1) // 2:
            raise ValueError(f"hard-negative coverage mismatch: {part}")
        seen_test_only = {r["hard_category"] for r in hard} & test_only
        if (part == "test" and seen_test_only != test_only) or (part != "test" and seen_test_only):
            raise ValueError(f"test-only hard context leakage/absence: {part}")
        lengths = sorted(len(tokenizer(r["state"], add_special_tokens=False)["input_ids"])
                         for r in rows) if tokenizer else []
        report["splits"][part] = {
            "groups": len(groups), "hard_groups": len(hard),
            "hard_negative_categories": dict(Counter(r["hard_category"] for r in hard)),
            "negative_alcohol_word": sum(r["label"] == "negative" and
                                         "alcohol" in r["state"].lower() for r in rows),
            "token_length": {"max": lengths[-1], "p95": lengths[int(len(lengths) * 0.95)]}
                            if lengths else None,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/alcohol_synthetic_v3"))
    parser.add_argument("--tokenizer")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.data_dir, args.tokenizer)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": report["counts"],
                      "hard_groups": {part: values["hard_groups"]
                                      for part, values in report["splits"].items()}}, indent=2))


if __name__ == "__main__":
    main()
