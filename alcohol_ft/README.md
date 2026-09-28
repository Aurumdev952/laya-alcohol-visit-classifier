# Alcohol attribution from hospital visit notes

**V2 preparation:** The 30,000-case attribution dataset, validation-aware Laya trainer, decomposed-question experiment, clinical classifier baselines, and single-GPU run configurations are ready. GPU fine-tuning for v2 has not run. See [the v2 runbook](../docs/ALCOHOL_V2_RUNBOOK.md) and [experiment plan](../docs/ACCURACY_IMPROVEMENT_PLAN.md). The instructions and measured results below describe the completed v1 run.

This is a visit-level, three-class Laya fine-tuning setup for English clinical notes. It follows the [Laya fine-tuning guide](https://laya-ai.com/guides/fine-tune-laya) and the [upstream 2×T4 RLCD notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb), adapted for **one 24 GB RTX 5090**. The choice question in [task.py](task.py) is identical in training and inference. The training loop uses the upstream proper-scoring reward, four noisy logit samples, cross-entropy guidance, encoder and head checkpointing, and a separate calibration split. It exports a Laya-loadable checkpoint with the newly fitted choice temperature and removes inherited per-option temperatures so they cannot override the fit.

## Label policy

One row is one hospital visit case report. `encounters` is an ordered list of notes for that visit. Review **all** notes before labeling. The internal label `direct` means **alcohol positive**, `indirect` means **alcohol positive but indirect**, and `negative` means **alcohol negative**.

| Label | Rule | Examples |
| --- | --- | --- |
| `direct` | Any note documents the patient's own alcohol consumption, alcohol use disorder, withdrawal, intoxication, or a condition linked to the patient's drinking, including remote history. | Patient admits wine; alcohol-related cirrhosis; AUD in remission. |
| `indirect` | No direct evidence, but someone else's drinking causally affected this patient or this presentation. | Sober patient struck by drunk driver; child harmed by an intoxicated caregiver; fetal alcohol exposure from maternal drinking. |
| `negative` | Neither rule is met. | Explicit denial, negative ethanol test, routine screening, generic counseling, alcohol swab, or unrelated family history. |

If both direct and indirect evidence appear, label `direct`. A denial in one note does not erase positive history in another. A question about alcohol, an unfilled intake field, or general family history does not establish use. These definitions are task-specific: if your clinical team wants a different time horizon or meaning for “positive,” revise [task.py](task.py), relabel the data, and retrain. Ambiguous or contradictory real cases should be adjudicated by two reviewers before they enter a gold set.

## Data delivered

[data/alcohol_synthetic](../data/alcohol_synthetic) contains **30,000 fully synthetic visits**: 24,000 train, 3,000 calibration, and 3,000 test. Each split is balanced: 8,000/1,000/1,000 per label. Most rows have multiple encounters. The seed, counts, and SHA-256 hashes are in [manifest.json](../data/alcohol_synthetic/manifest.json). Generation is deterministic:

```bash
python3 -m alcohol_ft.generate
python3 -m eval.alcohol.build_curated
python3 -m alcohol_ft.validate --curated eval/alcohol/curated.jsonl
```

The synthetic test uses separately held-out wording. The 90 authored [challenge cases](../eval/alcohol/curated.jsonl) use additional wording and targeted edge cases. These are **development diagnostics, not evidence of clinical accuracy**. Template generation can create strong lexical shortcuts. The first meaningful deployment benchmark needs de-identified, clinician-labeled real visits from the target hospital, sampled across services and time periods, with patient-level separation from training and calibration.

JSONL schema (one row per visit):

```json
{"case_id":"v001","patient_id":"p001","label":"indirect","source":"reviewed_clinical","scenario":"impaired_driver","encounters":[{"type":"ED physician note","note":"Sober pedestrian struck by an intoxicated driver."},{"type":"discharge summary","note":"Patient denies alcohol use."}]}
```

Use stable **pseudonymous** `patient_id` values so visits for the same patient stay in one split. Replace the synthetic files with adjudicated data of this exact schema, or pass another directory with `--data-dir`. Keep private records under `data/clinical/`, which is ignored by Git. Do not put names, dates of birth, addresses, medical record numbers, or raw identifiers into the training files, run reports, public repositories, or Hugging Face Hub. The validator rejects repeated case IDs and exact rendered reports, and rejects patient IDs crossing splits while allowing multiple visits for one patient within a split. It cannot detect near-duplicates or hidden identity leakage, so review those before clinical benchmarking.

For a real benchmark, reserve a chronologically later set from different patients, maintain the natural class prevalence in a separate prevalence sample, double-label difficult cases, record disagreements, and lock the test set before tuning. Use the 3,000 synthetic test and 90 challenge cases for regression checks only.

## Install and train on the 5090

Use the repository's [RTX 5090 setup](../README.md#run-on-an-rtx-5090) to fix the NVIDIA driver and install the CUDA PyTorch wheel. Then install the training extra and verify CUDA:

```bash
.venv/bin/python -m pip install -e '.[train,test]'
nvidia-smi
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
.venv/bin/python -m pytest -q tests/test_alcohol_ft.py
```

Start a fresh run directory. The default micro-batch is eight short synthetic reports, accumulated four times for effective batch 32. The encoder and head use activation checkpointing and BF16 autocast. If real cases are longer or memory is tight, reduce `--micro-batch` to 4 or 2 and raise `--grad-accum` to keep the effective batch similar. The trainer refuses silently truncated inputs and writes a rolling checkpoint after each epoch. It currently starts fresh; the rolling checkpoint preserves weights for inspection, but resuming optimizer state is not implemented.

```bash
.venv/bin/python -m alcohol_ft.train \
  --data-dir data/alcohol_synthetic \
  --output-dir runs/laya-alcohol-001 \
  --epochs 4 --micro-batch 8 --grad-accum 4
```

The final checkpoint is `runs/laya-alcohol-001/final`. Calibration is fit only on the 3,000 calibration visits, never on train or test. The trainer checks for at least 20 GiB of GPU memory. It does **not** publish weights or data.

## Benchmark

Run the base and fine-tuned checkpoint against the same locked test cases:

```bash
.venv/bin/python -m alcohol_ft.benchmark \
  --finetuned runs/laya-alcohol-001/final \
  --data-dir data/alcohol_synthetic \
  --curated eval/alcohol/curated.jsonl \
  --output runs/laya-alcohol-001/benchmark.json
```

The JSON report has full predictions, accuracy, macro F1, precision/recall/F1 per class, confusion matrix, direct and indirect miss counts, multiclass Brier score, 10-bin expected calibration error, confidence/coverage at fixed thresholds, scenario accuracy, elapsed time, and a paired bootstrap interval for the accuracy change over base Laya. Accuracy on the deliberately balanced synthetic set is **not** an estimate of deployment performance. For a real test set, run the same scorer on a locked real `test.jsonl` and report class prevalence and confidence intervals alongside these metrics. Do not select thresholds using test results; select them on a separate real calibration/validation set and then evaluate once on test.

A full RTX 5090 run completed on September 27, 2026. See [measured results](RUN_RESULTS.md) and the local report at `runs/laya-alcohol-001/benchmark.json`. The benchmark script fails rather than reporting CPU or silently truncated results.

### Counterfactual generalization check

The separate [generalization set](../eval/alcohol/generalization.jsonl) contains 30 independently authored groups of three visit reports. Each group holds the presentation mostly constant while changing the patient's own drinking, another person's causal drinking, or an incidental/negative mention. The strict score requires **all three** labels in a group to be correct. It was built after the four-epoch checkpoint was frozen and has no exact report overlap with training, calibration, the synthetic test, or the first challenge set.

```bash
.venv/bin/python -m eval.alcohol.generalization_eval \
  --finetuned runs/laya-alcohol-001/final \
  --output runs/laya-alcohol-001/generalization.json
```

The measured result is in [RUN_RESULTS.md](RUN_RESULTS.md). This authored synthetic test probes attribution and negation shortcuts; it cannot prove generalization to real hospital documentation. Do not train or select hyperparameters on it and then continue to call it a locked test.

## Inference

Pass one JSON case, a JSON array of cases, or JSONL cases with `encounters`. This prints one JSON result per visit with class probabilities and a provisional review flag. Choose the review threshold from real calibration data before operational use.

```bash
.venv/bin/python -m alcohol_ft.predict \
  --model runs/laya-alcohol-001/final \
  --input data/clinical/new_visits.jsonl \
  --review-below 0.8
```

The command rejects reports over the model's input window. Route those to review or build and validate a visit-level long-document strategy with real data. Probabilities from synthetic calibration should not be treated as clinical confidence estimates.
