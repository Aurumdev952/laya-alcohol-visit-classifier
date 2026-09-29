# Direct recall investigation after v3 retraining

The v3 checkpoint fixes most negative-to-indirect errors, but its direct recall drops on authored synthetic cases. The task rule counts the patient's own past use, disorder in remission, intoxication, and alcohol-attributed condition as **direct**, even when current use is denied or the visit has another immediate cause.

## Where recall changed

| Set | Direct cases | v2 correct | v3 correct |
| --- | ---: | ---: | ---: |
| v3 validation | 1,000 | 1,000 | 1,000 |
| v3 locked test | 1,000 | — | 1,000 |
| Older v2 validation | 1,000 | 1,000 | 1,000 |
| Curated authored diagnostic | 30 | 30 | 26 |
| Counterfactual authored diagnostic | 30 | 30 | 28 |
| Fresh authored challenge | 24 | 24 | 23 |

The seven v3 direct misses are `curated-direct-20` (alcohol-attributed neuropathy, predicted indirect), `curated-direct-21` (disorder in remission, negative), `curated-direct-25` (denies today plus remote dependence, negative), `curated-direct-30` (`AUD` in remission, negative), `gen-14-direct` (patient intoxicated before aspiration, indirect), `gen-15-direct` (patient drinking before a bus stumble, indirect), and `v3-adversarial-08-0-direct` (daily intake in social history, negative). The v2 checkpoint labeled all seven direct. The v3 checkpoint assigns direct probabilities below 0.001 to six; `gen-15-direct` receives 0.277. The failures are mostly confident, not threshold-edge cases.

## Controlled checks

The v2 and v3 experiment files have the same base revision, question, objective, seed, and epoch limit; they point to different datasets. Calibration temperature cannot change the winning class. Benchmarking rejects overlength inputs, and these short authored reports pass. The same seven direct errors appear in each saved v3 epoch (1 through 4), so selecting epoch 4 did not create the regression.

For each of the seven failed reports, we tested the original, a semantically similar rewrite that names the patient more explicitly, and the original with an additional sentence drawn from the training-style direct phrasing. V2 chose direct for all 21 probes. V3 chose direct for **0/7 originals**, **4/7 explicit rewrites**, and **7/7 with an added training-style sentence**. Adding evidence is a stronger intervention than rewriting, so this is evidence of wording sensitivity rather than a clean causal estimate of any one token. In `curated-direct-25`, remote dependence alone was negative; explicitly naming the patient made it direct; pairing that explicit remote history with a current-use denial made it negative again. A familiar “prior alcohol use disorder ... abstinent” sentence restored direct despite the denial. The probe inputs are [direct_recall_probes.jsonl](../eval/alcohol/direct_recall_probes.jsonl) and [direct_25_ablation.jsonl](../eval/alcohol/direct_25_ablation.jsonl); the scored reports are stored locally under `runs/alcohol-v3/`.

## Training-data explanation

The 7,000 v3 direct examples use the same 24 direct signal templates as v2. V3 replaces half the v2-style groups with hard causal groups, yet uses those same direct templates inside a fixed “Additional history” wrapper. This creates many examples without adding much direct wording variety. In v3 train, `AUD` as an abbreviation and patient intoxication after drinking are absent as direct signals. `neuropathy` appears in 101 direct evidence snippets versus 216 in v2, and `daily` in 316 versus 627. The hard groups restrict direct condition names and time phrases.

The new decoys also strengthen competing word associations. V3 has 275 negative reports mentioning a family member's *alcohol dependence*, versus none with that phrase in v2 negatives. Its indirect evidence mentions `intoxicated` 817 times versus 36 direct evidence snippets; v2 had 622 versus 65. Neither dataset has a direct report with a standalone current-use denial in one note and affirmative remote patient history in another. These counts and the probes support a narrow synthetic-template explanation, but model internals cannot establish a unique causal mechanism.

## Next data revision

Add patient-specific paraphrases for remote dependence, `AUD`, remission, neuropathy, own intoxication, and unquantified daily intake. Include multi-note direct examples where a current-use denial coexists with documented past use. Pair those with negative family-history and indirect actor-swapped counterparts across varying note orders and lengths. Use a separate authored development slice that tracks recall of all three classes for checkpoint selection, then create a new untouched challenge after revisions. Do not train on the existing authored diagnostic texts or the locked v3 test.

The authored sets are synthetic and small. This investigation identifies a reproducible regression on those sets; it does not estimate direct recall on real hospital notes.

To rescore a probe file, run `python -m alcohol_ft.benchmark_v2 --model runs/alcohol-v3/laya_rlcd-seed42/final --family laya --split custom --cases eval/alcohol/direct_recall_probes.jsonl --output runs/alcohol-v3/laya_rlcd-seed42/direct_probe_repeat.json`. Use the v2 final checkpoint as `--model` for comparison, and a new output path each time.
