# Alcohol attribution v3 run results

Analysis of the direct-class regression: [DIRECT_RECALL_INVESTIGATION.md](DIRECT_RECALL_INVESTIGATION.md).

## Run and protocol

The v3 run used 30,000 fully synthetic case reports: 21,000 train, 3,000 validation, 3,000 calibration, and 3,000 locked test. Each split contains complete direct/indirect/negative triplets. Half of the groups combine a non-alcohol visit mechanism with an incidental alcohol mention or link another person's impairment to the visit. The base checkpoint, `rlcd_ce` objective, training defaults, seed 42, and 1,024-token limit match v2. The selected epoch was 4 of 4 by validation macro F1. Its export temperature was 2.481, fit on the calibration split. The final checkpoint is `runs/alcohol-v3/laya_rlcd-seed42/final` (local, gitignored).

The v3 corpus manifest and split audit passed. The 72-case authored synthetic challenge was committed as `5709725` with SHA-256 `369c57f97db99f0d15232b1a71a717be6bfb20526d3be235479384111d59ede2` before the v3 checkpoint was scored on it. Neither the challenge nor the older authored diagnostics entered training. The older diagnostics informed the v3 data design and are development checks; the new challenge is a small, separate check. The v3 locked test was opened once after checkpoint selection.

## Model comparison

| Set | Model | Macro F1 | Negative recall | Complete triplets |
| --- | --- | ---: | ---: | ---: |
| v3 validation, 3,000 cases | v2 | 0.8500 | 0.571 | 571/1,000 |
| v3 validation, 3,000 cases | v3 | 0.9997 | 0.999 | 999/1,000 |
| v3 locked test, 3,000 cases | v3 | 0.9997 | 0.999 | 999/1,000 |
| Fresh authored challenge, 72 cases | v2 | 0.8857 | 0.667 | 16/24 |
| Fresh authored challenge, 72 cases | v3 | 0.9443 | 0.875 | 20/24 |
| Prior counterfactual diagnostic, 90 cases | v2 | 0.9211 | 0.767 | 23/30 |
| Prior counterfactual diagnostic, 90 cases | v3 | 0.9778 | 1.000 | 28/30 |
| Prior curated diagnostic, 90 cases | v2 | 0.9555 | 0.900 | — |
| Prior curated diagnostic, 90 cases | v3 | 0.9442 | 1.000 | — |

On v3 validation, hard-negative accuracy rose from 71/500 (v2) to 499/500 (v3), and negative-to-indirect errors fell from 427 to 1. The paired bootstrap over 1,000 matched validation groups estimated a macro F1 increase of 0.1497 (95% interval 0.1382 to 0.1620) and negative recall increase of 0.428 (0.398 to 0.459). These intervals describe this synthetic validation distribution, not clinical performance.

The locked test had one error in 3,000 cases: a 21-note trolley report with incidental alcohol-based cleaner use was labeled indirect instead of negative, with 0.995 predicted probability. Among the four test-only hard-negative mechanisms, elevator, ladder, and playground were 51/51, 51/51, and 50/50 correct; trolley was 48/49. The fresh challenge had three negative-to-indirect errors (printed aftercare warning; unrelated neighbor drinking in both note orders) and one direct-to-negative error (patient daily alcohol intake in a social-history note). These four errors yielded 20/24 complete triplets. Three of four fresh-challenge errors had predicted probability at least 0.9.

The prior curated diagnostic reveals a tradeoff: negative recall increased from 0.900 to 1.000, while direct recall fell from 1.000 to 0.867, lowering overall macro F1. The prior counterfactual diagnostic improved, but two direct cases remain mislabeled indirect. These are reasons to retain the full three-class diagnostics rather than judge the model on negative recall alone.

## Interpretation

The dataset update and retraining substantially reduced the targeted negative-to-indirect failure on matched synthetic cases, including held-out generator mechanisms and a separately authored challenge. It did not remove every causal attribution error. All sets here are synthetic and relatively small or templated outside the 30,000-case corpus. High-confidence mistakes persist, and no clinician-adjudicated hospital notes were evaluated. Clinical deployment would require a de-identified, patient-separated, clinician-labeled holdout with adjudicated label rules and safety review.

Reproduce the commands and inspect the local JSON reports using [ALCOHOL_V3_RUNBOOK.md](ALCOHOL_V3_RUNBOOK.md). The paired comparison is saved at `runs/alcohol-v3/laya_rlcd-seed42/validation_paired_comparison.json`; the benchmark reports are in the same run directory. The local model and reports are ignored by git because the public repository contains code and synthetic examples, not weights.
