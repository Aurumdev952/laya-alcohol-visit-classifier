"""Run one v2 experiment at a time; dry-run by default.

Examples:
  python experiments/alcohol_v2/run.py laya_rlcd
  python experiments/alcohol_v2/run.py laya_rlcd --execute
  python experiments/alcohol_v2/run.py laya_rlcd --execute --seed 43
  python experiments/alcohol_v2/run.py laya_rlcd --test --execute
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent


def commands(name: str, seed: int, *, test: bool) -> tuple[list[list[str]], Path, Path]:
    config_path = CONFIG_DIR / f"{name}.json"
    if not config_path.is_file():
        raise ValueError(f"unknown experiment: {name}")
    family = "classifier" if name.endswith("_classifier") else "laya"
    trainer = "alcohol_ft.train_classifier" if family == "classifier" else "alcohol_ft.train_v2"
    run_dir = ROOT / "runs" / "alcohol-v2" / f"{name}-seed{seed}"
    effective = ROOT / "runs" / "alcohol-v2" / "configs" / f"{name}-seed{seed}.json"
    model = run_dir / "final"
    split = "test" if test else "validation"
    result = run_dir / f"{split}_benchmark.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["seed"] = seed
    if test:
        if not model.exists():
            raise ValueError(f"final checkpoint missing: {model}")
        if result.exists():
            raise ValueError(f"benchmark already exists: {result}")
        cmd = [sys.executable, "-m", "alcohol_ft.benchmark_v2", "--model", str(model),
               "--family", family, "--split", "test", "--unlock-test", "--output", str(result)]
        return [cmd], effective, result
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"run already started; resume manually with trainer --resume: {run_dir}")
    if result.exists():
        raise ValueError(f"benchmark already exists: {result}")
    cmd_train = [sys.executable, "-m", trainer, "--config", str(effective),
                 "--output-dir", str(run_dir)]
    cmd_val = [sys.executable, "-m", "alcohol_ft.benchmark_v2", "--model", str(model),
               "--family", family, "--split", "validation", "--output", str(result)]
    return [cmd_train, cmd_val], effective, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="stem of experiments/alcohol_v2/<name>.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true", help="run the locked test on a frozen finalist")
    parser.add_argument("--execute", action="store_true", help="actually start GPU work")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    steps, effective, result = commands(args.name, args.seed, test=args.test)
    config = json.loads((CONFIG_DIR / f"{args.name}.json").read_text(encoding="utf-8"))
    config["seed"] = args.seed
    print(json.dumps({"config": config, "commands": steps, "report": str(result),
                      "will_execute": args.execute}, indent=2))
    if not args.execute:
        return
    if not args.test:
        effective.parent.mkdir(parents=True, exist_ok=True)
        effective.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for command in steps:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
