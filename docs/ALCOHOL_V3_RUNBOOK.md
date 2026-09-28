# Alcohol attribution v3: causal-context retraining

Run outcomes and remaining errors: [ALCOHOL_V3_RUN_RESULTS.md](ALCOHOL_V3_RUN_RESULTS.md).

The v3 corpus targets the remaining negative-to-indirect errors found on the independently authored cases. It keeps the three-label, visit-level task and 30,000 synthetic reports. Half the matched groups contain two-note hard contexts: a documented non-alcohol mechanism plus an incidental alcohol mention for the negative variant, or an event linked to another person's drinking for the indirect variant. The direct variant documents the patient's own use in the same presentation. Each group stays in one split.

The 21,000/3,000/3,000/3,000 train/validation/calibration/test split is unchanged in size. Training contains 3,500 hard groups; each other split has 500. The locked test has held-out sentence surfaces, three reserved presentation families, and additional mechanisms that do not appear in training. The old curated and counterfactual sets remain development diagnostics; none of their exact reports appears in v3. A separate 72-case authored synthetic challenge, frozen before v3 scoring, tests 12 further causal mechanisms in two note orders. All reports remain synthetic and cannot establish hospital-note accuracy.

## Build and audit

```bash
.venv/bin/python -m alcohol_ft.generate_v3 --output-dir data/alcohol_synthetic_v3
.venv/bin/python -m alcohol_ft.audit_v3 --data-dir data/alcohol_synthetic_v3 \
  --tokenizer runs/alcohol-v2/laya_rlcd-seed42/final/tokenizer \
  --output runs/alcohol-v3-data-audit.json
.venv/bin/python -m pytest -q tests/test_alcohol_v3.py tests/test_alcohol_v2.py
```

The generator refuses to overwrite existing data. Use a fresh output directory to reproduce and compare manifest hashes.
The authored challenge is stored at `eval/alcohol/v3_adversarial.jsonl`, with a SHA-256 manifest. To rebuild it in a fresh location, pass `--output` to `eval/alcohol/generate_v3_adversarial.py`.

## Controlled retraining and evaluation

The v3 `laya_rlcd` config keeps the v2 base revision, objective, optimizer defaults, batch size, 1,024-token limit, seed, and early-stopping rule. Only the synthetic corpus changes. A baseline v2 checkpoint is scored on v3 validation before training.

```bash
.venv/bin/python -m alcohol_ft.benchmark_v2 \
  --model runs/alcohol-v2/laya_rlcd-seed42/final --family laya \
  --data-dir data/alcohol_synthetic_v3 --split validation \
  --output runs/alcohol-v3/v2_on_v3_validation.json
.venv/bin/python experiments/alcohol_v3/run.py laya_rlcd --execute
```

The runner trains, selects an epoch by validation macro F1, fits temperature on calibration, and saves a validation benchmark. If interrupted, resume from its saved optimizer state:

```bash
.venv/bin/python -m alcohol_ft.train_v2 \
  --config runs/alcohol-v3/configs/laya_rlcd-seed42.json \
  --output-dir runs/alcohol-v3/laya_rlcd-seed42 --resume
```

After selection, score the old authored diagnostics without using them to train, and open v3 test once for the selected checkpoint:

```bash
.venv/bin/python -m alcohol_ft.benchmark_v2 \
  --model runs/alcohol-v3/laya_rlcd-seed42/final --family laya \
  --split custom --cases eval/alcohol/curated.jsonl \
  --output runs/alcohol-v3/laya_rlcd-seed42/curated_diagnostic.json
.venv/bin/python -m alcohol_ft.benchmark_v2 \
  --model runs/alcohol-v3/laya_rlcd-seed42/final --family laya \
  --split custom --cases eval/alcohol/generalization.jsonl \
  --output runs/alcohol-v3/laya_rlcd-seed42/generalization_diagnostic.json
.venv/bin/python -m alcohol_ft.benchmark_v2 \
  --model runs/alcohol-v3/laya_rlcd-seed42/final --family laya \
  --split custom --cases eval/alcohol/v3_adversarial.jsonl \
  --output runs/alcohol-v3/laya_rlcd-seed42/v3_adversarial_benchmark.json
.venv/bin/python experiments/alcohol_v3/run.py laya_rlcd --test --execute
```

Report per-class recall, negative-to-indirect errors, high-confidence errors, complete matched triplets, hard-negative categories, and held-out mechanisms. Synthetic test accuracy can remain high despite errors on independently authored reports; prioritize the latter for diagnosing the causal-link failure. A real, clinician-adjudicated hospital holdout remains necessary before deployment.
