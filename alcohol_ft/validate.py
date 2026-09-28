"""Check visit JSONL labels, case structure, and patient-level split isolation."""

import argparse
import json
from pathlib import Path

from alcohol_ft.data import validate_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/alcohol_synthetic"))
    parser.add_argument("--curated", type=Path)
    args = parser.parse_args()
    parts = ("train", "validation", "calibration", "test")
    paths = {split: args.data_dir / f"{split}.jsonl" for split in parts
             if (args.data_dir / f"{split}.jsonl").exists()}
    if args.curated:
        paths["curated"] = args.curated
    print(json.dumps(validate_splits(paths), indent=2))


if __name__ == "__main__":
    main()
