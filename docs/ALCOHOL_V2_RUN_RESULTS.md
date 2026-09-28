# Alcohol attribution v2 run results

## Protocol

The v2 corpus contains 30,000 fully synthetic visit reports: 21,000 training, 3,000 validation, 3,000 calibration, and 3,000 locked test reports. Each split is balanced across direct, indirect, and negative labels. Cases in a matched three-label group stay in the same split. The test includes presentation families withheld from training. The corpus manifest and token audit are in `data/alcohol_synthetic_v2/manifest.json` and `runs/alcohol-v2/data_audit.json`.

Validation macro F1 selects an epoch. A separate calibration split fits the exported temperature. The locked test and independently authored cases are evaluated after selection. No real hospital records are included, so these results do not establish clinical accuracy.

## Baseline and training status

The earlier v1 Laya checkpoint scored **0.8072 macro F1** on v2 validation, with 0.607 negative recall and 360 negative-to-direct errors among 1,000 negative cases. Its report is `runs/alcohol-v2/v1_on_v2_validation.json`.

The v2 `laya_rlcd` run uses pinned `convaiinnovations/laya` revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, seed 42, BF16, micro-batch 8, accumulation 4, and a 1,024-token input limit. All three completed epochs reached **1.000 macro F1** on the synthetic validation split. The preset earliest-best rule selected epoch 1; the separate calibration split set the export temperature to 1.0. Its checkpoint and reports are in `runs/alcohol-v2/laya_rlcd-seed42/`.

The computer rebooted during epoch 2. Training resumed from the saved epoch-1 model, optimizer, scheduler, and random states. The trainer now converts saved CUDA random-state tensors back to CPU byte tensors during restore. The restarted epochs 2 and 3 completed normally.

| Evaluation | Cases | Macro F1 | Negative recall | Complete triplets | Negative → direct |
| --- | ---: | ---: | ---: | ---: | ---: |
| V1 checkpoint on v2 validation | 3,000 | 0.8072 | 0.607 | 505/1,000 | 360 |
| V2 checkpoint on v2 validation | 3,000 | 1.0000 | 1.000 | 1,000/1,000 | 0 |
| V2 checkpoint on locked synthetic test | 3,000 | 0.9993 | 0.998 | 998/1,000 | 0 |
| V2 checkpoint on authored curated set | 90 | 0.9555 | 0.900 | — | 0 |
| V2 checkpoint on authored counterfactual set | 90 | 0.9211 | 0.767 | 23/30 | 0 |

The locked synthetic test's 1,500-case reserved-family slice was 100% accurate. Both test errors were familiar-family negative cases called indirect. The authored counterfactual set exposed seven negative-to-indirect errors, five at prediction confidence at least 0.9. These included alcohol products, patient denials and negative screens, and unrelated family or partner drinking. The curated set had four errors: three negative-to-indirect and one indirect-to-negative. There were no negative-to-direct errors in either authored set.

The v1 checkpoint previously scored 0.867 macro F1 on the authored curated set and 0.884 on the authored counterfactual set, with 21/30 complete triplets. These authored sets are development diagnostics: the earlier run's errors were already inspected. They are not untouched evidence for model selection.

On the shared v2 validation groups, the v2 checkpoint improved accuracy by 0.1887 (group-bootstrap 95% interval 0.1763–0.2020), macro F1 by 0.1928 (0.1799–0.2069), and negative recall by 0.393 (0.363–0.424) over the v1 checkpoint. These intervals describe sampling of the fixed synthetic groups, not uncertainty about hospital performance. The full 2,000-draw comparison is in `runs/alcohol-v2/laya_rlcd-seed42/paired_comparison.json`.

The synthetic notes use recurring evidence patterns. The near-perfect synthetic results should therefore be read as performance on this generated corpus. The authored cases show that causal attribution still fails on some unrelated mentions, including highly confident errors. None of these sets substitutes for independently labeled, de-identified hospital visits with patient-level separation and a real calibration and test split.

This report covers one v2 Laya training recipe at seed 42. The prepared CE, decomposed-question, learning-rate, alternative-starting-weight, and classifier experiments have not been trained or compared yet. The locked synthetic test is now opened for this selected checkpoint, so future variants should be compared on development data or a newly reserved test rather than tuned to this result. Seed variability is also unmeasured.

## Reproduce the run

From the project root, with the Python environment installed and the GPU available:

```bash
.venv/bin/python experiments/alcohol_v2/run.py laya_rlcd --execute
.venv/bin/python experiments/alcohol_v2/run.py laya_rlcd --test --execute
.venv/bin/python -m alcohol_ft.benchmark_v2 --model runs/alcohol-v2/laya_rlcd-seed42/final --family laya --split custom --cases eval/alcohol/generalization.jsonl --output runs/alcohol-v2/laya_rlcd-seed42/generalization_diagnostic.json
```

The first command requires a fresh run directory. The test command refuses to overwrite an existing test report. In the recorded run, a reboot interrupted epoch 2, so the trainer was resumed from `resume.pt` with `python -m alcohol_ft.train_v2 --config runs/alcohol-v2/configs/laya_rlcd-seed42.json --output-dir runs/alcohol-v2/laya_rlcd-seed42 --resume`. The standalone validation benchmark was then run manually. The focused alcohol test suites passed: 17 tests.
