# Alcohol attribution v2: preparation and runbook

The v2 experiment code and data are prepared. The first v2 GPU fine-tune completed on September 28, 2026; its measured results are in [the run results](ALCOHOL_V2_RUN_RESULTS.md). The completed v1 checkpoint and results remain in `runs/laya-alcohol-001` and `alcohol_ft/RUN_RESULTS.md`.

## Prepared assets

- `data/alcohol_synthetic_v2/`: 30,000 synthetic visits in 10,000 matched three-class groups. Splits are 21,000 train, 3,000 validation, 3,000 calibration, and 3,000 locked test. The manifest records hashes and provenance; the audit checks group completeness and split isolation.
- `alcohol_ft/generate_v2.py`, `audit_v2.py`: deterministic regeneration and data checks.
- `alcohol_ft/train_v2.py`: single-GPU Laya training with separate validation, epoch checkpoints, early stopping, CE/RLCD objective choice, resume state, and post-selection calibration.
- `alcohol_ft/train_classifier.py`: clinical and general ModernBERT three-class baselines using the same case splits.
- `alcohol_ft/clinical_laya.py`: experimental compatibility check and encoder-graft constructor. A successful graft still needs a full task fine-tune.
- `alcohol_ft/benchmark_v2.py`: shared visit, class, slice, and complete-group metrics. The locked test requires an explicit flag.
- `alcohol_ft/compare_v2.py`, `predict_v2.py`: paired group-aware comparison and inference for Laya or classifier checkpoints.
- `experiments/alcohol_v2/*.json` and `run.py`: ready experiment configurations and a dry-run-first sequential runner.

The new generator has 20 presentation families, each with direct, indirect, and negative variants. Developmental families include prenatal exposure; trauma families include another person's alcohol-related collision, assault, or injury. Three families are reserved for half of the locked test; the other half uses familiar families with distinct report styles. Evidence sentences vary by actor, wording, context, and encounter placement. The data is still synthetic and uses authored patterns, so its test result cannot establish accuracy on hospital notes.

## Regenerate and verify data

The generator refuses to overwrite an existing dataset directory. Use a fresh destination to reproduce and compare hashes:

```bash
.venv/bin/python -m alcohol_ft.generate_v2 --output-dir /tmp/alcohol_synthetic_v2_check
.venv/bin/python -m alcohol_ft.audit_v2 --data-dir data/alcohol_synthetic_v2
.venv/bin/python -m alcohol_ft.validate --data-dir data/alcohol_synthetic_v2
```

For tokenizer length auditing, point to the already downloaded Laya tokenizer:

```bash
.venv/bin/python -m alcohol_ft.audit_v2 \
  --tokenizer runs/laya-alcohol-001/final/tokenizer \
  --output runs/alcohol-v2/data_audit.json
```

The generator's `annotation` stores the actor flags and evidence references for validation and decomposed-question training. `case_state()` passes only encounters to the model. For real notes, obtain independent annotation and adjudication before using these flags.

## Training order

The runner prints commands and configurations without executing by default. `--execute` begins GPU training and then scores the validation split. Run one experiment at a time:

First score the completed v1 checkpoint on v2 validation as the common baseline when CUDA is available:

```bash
.venv/bin/python -m alcohol_ft.benchmark_v2 \
  --model runs/laya-alcohol-001/final --family laya \
  --split validation --output runs/alcohol-v2/v1_on_v2_validation.json
```

```bash
.venv/bin/python experiments/alcohol_v2/run.py laya_rlcd
.venv/bin/python experiments/alcohol_v2/run.py laya_rlcd --execute
.venv/bin/python experiments/alcohol_v2/run.py laya_ce --execute
.venv/bin/python experiments/alcohol_v2/run.py laya_decomposed --execute
```

Each run writes epoch checkpoints, `resume.pt`, `validation_history.json`, `selection.json`, a calibrated `final/` checkpoint, and a validation benchmark under `runs/alcohol-v2/<name>-seed42/`. The output path must be new. If a run is interrupted, use its effective configuration and the trainer's `--resume` flag:

```bash
.venv/bin/python -m alcohol_ft.train_v2 \
  --config runs/alcohol-v2/configs/laya_rlcd-seed42.json \
  --output-dir runs/alcohol-v2/laya_rlcd-seed42 --resume
```

After the first two objective runs, compare validation metrics and epoch histories. Run `laya_encoder_low`, `laya_head_low`, and `laya_both_low` as controlled learning-rate experiments. The preset comparison uses the same v2 data, seed, question, and objective. Choose the winning objective before interpreting its learning-rate sweep; if CE wins, copy the learning-rate presets with `"objective": "ce"` into new named config files.

```bash
.venv/bin/python -m alcohol_ft.compare_v2 \
  --reference runs/alcohol-v2/v1_on_v2_validation.json \
  --candidate runs/alcohol-v2/laya_rlcd-seed42/validation_benchmark.json \
  --output runs/alcohol-v2/laya_rlcd-seed42/paired_comparison.json
```

`python -m alcohol_ft.select_v2 --reference <baseline.json> --candidate <validation_benchmark.json>` checks the predefined negative-recall and direct/indirect-recall targets. Use validation macro F1 to rank candidates that pass all three checks.

Then run `laya_typed` to compare Laya starting weights. Compare `clinical_classifier` with `general_classifier` to isolate the effect of clinical pretraining within the same classifier architecture. These classifiers are separate baselines and cannot be loaded with `laya.load`.

The optional clinical-Laya candidate needs a compatible local initialization:

```bash
.venv/bin/python -m alcohol_ft.clinical_laya
.venv/bin/python -m alcohol_ft.clinical_laya \
  --base-revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851 \
  --clinical-revision c684f470c2ee60531d8c47c32187dba746c84201 \
  --output /tmp/clinical-laya-init-check
.venv/bin/python experiments/alcohol_v2/run.py laya_clinical_graft --execute
```

The first command checks model architecture and tokenizer metadata. The build command additionally checks every encoder tensor key and shape before writing a Laya checkpoint with clinical encoder weights and the original decision head. A mismatch is a research finding; do not bypass it by partially loading weights. The pinned clinical encoder passed these checks, and its initialization is in `runs/alcohol-v2/clinical-laya-init`.

All presets use BF16 and gradient checkpointing. Laya starts with micro-batch 8 and accumulation 4; the classifier starts with 4 and 8. Profile peak memory and reduce micro-batch while raising accumulation for long cases if needed. Do not change the 1,024-token input budget without a length audit and a separate validation comparison.

## Selection and final evaluation

Choose checkpoints using validation macro F1 and inspect negative recall, negative-to-direct errors, direct/indirect recall, complete triplets, and the reserved-family slice. The research target is negative recall of at least 0.90 without losing more than 0.02 direct or indirect recall against the current checkpoint on the same v2 validation cases. A selected epoch's calibration temperature comes only from the calibration split. Decomposed visit probabilities are a diagnostic approximation; validate component probabilities and any joint calibration separately.

After choosing two finalists, repeat each with seeds 43 and 44 by adding `--seed 43` or `--seed 44` to the runner command. Freeze the recipe before opening the test. Run each final checkpoint on the locked test once:

```bash
.venv/bin/python experiments/alcohol_v2/run.py laya_rlcd --test --execute
```

The test command requires `--unlock-test` internally and refuses to overwrite an existing report. Interpret correlated counterfactual rows by group, and report seed variability. The old authored challenge sets are development diagnostics because their errors are already known.

For an unlabeled visit JSON or JSONL file, run `python -m alcohol_ft.predict_v2 --model <run>/final --input <visits.jsonl>`. The command detects Laya versus a Transformers classifier from the saved checkpoint. The review threshold and the decomposed model's joint probabilities remain provisional until checked on real calibration data.

## Known limits

- The v2 data uses authored patterns and neutral-note variation. It improves controlled attribution coverage, but a model can still learn synthetic style. Independently labeled, de-identified hospital visits with patient and time separation are required to assess clinical generalization.
- The clinical-Laya graft is an experimental initialization; matching tensor shapes do not establish that it improves this task. It needs its own fine-tune and comparison.
- Temperature scaling adjusts probability sharpness, not the winning class. Classification errors remain the primary target.
- The experiment presets pin the Hub revisions used for comparisons. Local custom checkpoints use their saved initialization metadata.
- `resume.pt` uses PyTorch serialization and must be loaded only from a trusted local run directory.
- The RTX 5090 completed the first v2 Laya fine-tune. The v1-on-v2 validation baseline was 0.8072 macro F1 with negative recall 0.607; the v2 checkpoint reached 1.000 macro F1 on synthetic validation and 0.9993 on the locked synthetic test. See the run results for authored-case failures.

See [the run results](ALCOHOL_V2_RUN_RESULTS.md) for measured scores and [the full improvement plan](ACCURACY_IMPROVEMENT_PLAN.md) for data policy, acceptance gates, and the experiment matrix.
