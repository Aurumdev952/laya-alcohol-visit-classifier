# RTX 5090 training run: September 27, 2026

The full four-epoch run completed on one NVIDIA GeForce RTX 5090 Laptop GPU (24,463 MiB reported by `nvidia-smi`). The installed stack was PyTorch 2.14.0+cu130, Laya 0.3.20, and Transformers 5.17.0. Training and held-out calibration took about **1 hour 47 minutes** after the base checkpoint download. The final model is local at `runs/laya-alcohol-001/final`; the machine-readable results are at `runs/laya-alcohol-001/benchmark.json`. The checkpoint and detailed report are ignored by Git.

The run used 24,000 synthetic training visits, four epochs, micro-batch 8, gradient accumulation 4, BF16 autocast, and encoder/head activation checkpointing. The 3,000-case calibration split did not enter training. Its fitted choice temperature was **4.9069**; the exported config removes inherited per-option temperature overrides. The 3,000-case synthetic test and 90 authored challenge cases were not used for training or calibration.

| Set | Model | Accuracy | Macro F1 | Direct recall | Indirect recall | Negative recall | Brier | ECE (10 bins) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Synthetic test, n=3,000 | Base | 0.371 | 0.340 | 0.482 | 0.537 | 0.094 | 0.696 | 0.106 |
| Synthetic test, n=3,000 | Fine-tuned | **0.870** | **0.867** | 0.936 | 1.000 | 0.673 | 0.228 | 0.110 |
| Authored challenge, n=90 | Base | 0.344 | 0.350 | 0.300 | 0.367 | 0.367 | 0.706 | 0.174 |
| Authored challenge, n=90 | Fine-tuned | **0.867** | **0.867** | 0.933 | 0.933 | 0.733 | 0.233 | 0.111 |

The paired accuracy gain over base was +0.499 on the synthetic test (bootstrap 95% interval 0.481–0.516) and +0.522 on the challenge set (0.400–0.644). These intervals describe variation within these fixed synthetic sets; they do not measure generalization to real hospital records.

The main error is **false direct positives**. On the 3,000-case test, 327 of 1,000 negative cases were predicted `direct`; 64 of 1,000 direct cases were predicted `negative`. The challenge set had 8 negative-to-direct errors out of 30 negatives, including alcohol wipes, isopropyl alcohol, hand sanitizer, unrelated family drinking, and a prevention leaflet. Several were assigned more than 0.99 probability, so a simple confidence threshold does not reliably catch them. At a 0.9 threshold on the synthetic test, the model retained 96.2% of cases but accuracy was only 89.4% on those retained. Two indirect challenge cases were missed: a police report of the other driver's alcohol test and a prenatal exposure note.

## Counterfactual generalization test

After training was complete, we authored a second locked test of **30 matched triplets (90 reports)**. Each triplet changes the actor or causal role of alcohol while holding the presentation similar. No exact rendered report overlaps any earlier split. The dataset SHA-256 is `1cce8f497c8b35d4b0241ef3acc262ed4b83814f35958ff735d5ccbfbb622c89`. Full predictions and per-group results are in `runs/laya-alcohol-001/generalization.json`.

| Model | Case accuracy | Macro F1 | Complete triplets | Negative recall | Negative → direct | Errors at ≥0.9 probability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 0.367 | 0.328 | 2/30 | 0.133 | 10/30 | 0 |
| Fine-tuned | **0.889** | **0.884** | **21/30** | 0.700 | 5/30 | 8 |

The fine-tuned model labeled all 30 indirect variants correctly and 29 of 30 direct variants correctly, but missed **9 of 30 negative variants**. It was highly confident on eight errors. Failures included incidental alcohol hand rub or wipes, unrelated family alcohol history, generic medication warnings, and explicit patient denial. This test shows transfer beyond the generator's exact templates while revealing a persistent “alcohol word implies positive” shortcut. It does **not** ensure the model has avoided overfitting to synthetic clinical language.

**Interpretation:** fine-tuning substantially improved this synthetic task over the base checkpoint, but the false-positive pattern and residual calibration error make this checkpoint unsuitable as an autonomous clinical decision. The dataset is templated and class-balanced, while real notes and prevalence differ. A deployment decision needs de-identified, independently adjudicated real visits with patient-level and time-based separation, a real calibration split, and an untouched real test set. Reviewers should examine denial, product, family-history, counseling, prenatal, and mixed-encounter slices separately. The current checkpoint is suitable for further development and error analysis.

Commands used:

```bash
.venv/bin/python -m alcohol_ft.train --data-dir data/alcohol_synthetic --output-dir runs/laya-alcohol-001 --epochs 4 --micro-batch 8 --grad-accum 4
.venv/bin/python -m alcohol_ft.benchmark --finetuned runs/laya-alcohol-001/final --data-dir data/alcohol_synthetic --curated eval/alcohol/curated.jsonl --output runs/laya-alcohol-001/benchmark.json --batch-size 8
.venv/bin/python -m eval.alcohol.generalization_eval --finetuned runs/laya-alcohol-001/final --output runs/laya-alcohol-001/generalization.json
.venv/bin/python -m pytest -q tests/test_alcohol_ft.py tests/test_alcohol_generalization.py
```

The eight alcohol-specific tests passed. Before public publication, the full repository suite also passed: **29 tests**, with one dependency deprecation warning.
