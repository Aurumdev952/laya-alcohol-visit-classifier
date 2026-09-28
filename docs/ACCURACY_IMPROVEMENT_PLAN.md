# Alcohol attribution: improvement and experiment plan

Status: proposed implementation plan; no new dataset, training run, or measured improvement is implied. Prepared September 28, 2026.

## Objective and current evidence

Improve visit-level classification into `direct`, `indirect`, and `negative`, especially attribution to the correct person and false positives on incidental alcohol mentions. Complete all five workstreams: data, validation-driven training, question decomposition, objective/configuration experiments, and starting-weight comparisons.

The existing run scored 0.870 accuracy and 0.867 macro F1 on 3,000 synthetic cases, with negative recall 0.673. The counterfactual set scored 0.889 accuracy but only 21/30 complete triplets. These results are development evidence, not real-hospital performance estimates. The generator has 22 core training signal templates for 24,000 training visits. See `alcohol_ft/RUN_RESULTS.md`.

## 1. Dataset v2: 30,000 visits

Keep the v1 dataset, checkpoint, and reports immutable. Generate v2 into a new directory with a versioned manifest.

| Split | Visits | Purpose |
| --- | ---: | --- |
| Training | 21,000 | Fit weights; initially 7,000 visits per class |
| Validation | 3,000 | Choose recipe, checkpoint, and any decision thresholds |
| Calibration | 3,000 | Fit probabilities after recipe/checkpoint selection |
| Locked test | 3,000 | Final comparison after candidate selection |
| Total | 30,000 | Synthetic bootstrap dataset |

The new allocation adds validation while preserving the requested 30k total. Count original visits separately from training-time augmentation and question-level examples. Balanced synthetic splits are for controlled development; they do not estimate hospital prevalence.

### Label policy and annotations

- Keep existing direct-first precedence and the current treatment of documented past personal use, prenatal exposure, and unrelated family history unless the task owner changes the policy.
- Explicitly document correction/retraction, conflicting encounters, uncertain allegations, nonbeverage exposure, and historical versus current use. Obtain adjudication for unresolved policy cases before including them as hard-label examples.
- Store the final label, evidence spans with encounter references, actor, assertion status, temporal status, causal relationship, scenario family, source scenario ID, counterfactual group ID, and generation provenance.
- Store patient-use and other-person-causal-involvement labels independently. A direct visit can also contain genuine indirect evidence; the second label must not be inferred as false merely because the final class is direct.
- Keep all annotation and generation metadata out of model input. Provide only the case report and the versioned production question.

### Coverage and generation

Build at least 60 reviewed scenario families across personal use/history, alcohol-attributed illness, other-person causal involvement, and negative cases. Include denial, negative tests, wipes/sanitizer, routine medication warnings, unrelated family history, quoted statements, abbreviations, and screening-only notes. Also include visits with no alcohol language at all.

Create matched counterfactual groups that preserve the clinical background while changing actor, assertion, or causality. Include mixed encounters, late decisive evidence, conflicting notes, and combinations requiring direct-first precedence. Balance nuisance features such as note count, style, section, and length across labels.

Generate a structured scenario with an explicit evidence contract first, then render it into varied clinical prose. Prototype a local language-model renderer and benchmark its throughput and fidelity before selecting it; record model revision, prompts, seeds, and license. Do not assume a larger teacher produces correct labels. Independent consistency checks must reject altered actors, missing evidence, accidental extra positive evidence, and unsupported causal links. Audit a stratified sample manually; clinical validity still requires clinical review.

Split source scenarios and counterfactual groups before rendering. Never put paraphrases or variants from the same source into different splits. Reserve scenario families and rendering styles for generalization evaluation. Tag the locked test's familiar-family and unseen-family slices separately. Exact and approximate duplicate detection must ignore neutral differences such as vitals and identifiers and flag suspicious semantic overlap for review.

Profile token lengths with each candidate tokenizer. Include both short and long visits. Long cases must retain decisive evidence and its context; unsupported lengths must fail explicitly rather than silently truncate. Length/encounter coverage and rendering acceptance rates belong in the dataset report.

**Ready when:** split counts, label coverage, group isolation, duplicate checks, evidence consistency, token coverage, and a documented sample review pass. Freeze hashes and provenance before the training sweep.

## 2. Training and validation infrastructure

Extend `alcohol_ft/train.py` and validation utilities to support four splits, versioned configuration files, and a distinct validation loop.

- Expose encoder/head learning rates, objective, RL coefficient, noise schedule/group size, weight decay, warmup, seed, context length, and checkpoint frequency.
- Retain per-epoch checkpoints and optionally validate more frequently. Record validation loss, macro F1, class recalls, and negative-to-direct error rate. Support best-checkpoint selection and early stopping with explicit patience.
- Implement resume with model, optimizer, scheduler, random states, and data-order progress. Save actual run metadata rather than inherited base-checkpoint training metadata.
- Record code commit, dependency versions, checkpoint revision, dataset hashes, question schema, tokenizer, precision, effective batch size, elapsed time, and peak GPU memory.
- Keep calibration out of weight updates and recipe selection. Fit it only for selected finalists; clear inherited temperature overrides and verify training/export/runtime temperature bounds agree.
- Round-trip the exported checkpoint through the production prediction path and compare outputs with the evaluation path.

Start from the proven 24 GB setup: BF16, gradient checkpointing, micro-batch 8, accumulation 4, context 1024. Profile the longest batches for every candidate; reduce micro-batch and increase accumulation if needed. Profile 2048-token training separately before choosing a long-visit strategy. An encoder advertising 8192 tokens does not establish that full-length training fits this GPU or that accuracy transfers to that length.

**Ready when:** a short training/resume/export smoke run succeeds, validation selects the intended checkpoint, and no test or calibration rows enter the optimization/selection path.

## 3. Question decomposition

Add a versioned alternative to the existing three-way question:

1. Is qualifying personal alcohol involvement documented for the patient?
2. Did another person's alcohol use cause or contribute to the patient's presentation?

Use explicit semantic `choice` labels and descriptions. Derive final hard decisions by the declared precedence: personal evidence -> direct; otherwise other-person causal evidence -> indirect; otherwise negative. Compare at visit level on exactly the same data.

Do not multiply the two marginal probabilities and describe the result as a validated joint distribution. Retain component probabilities and report component calibration separately; any derived three-class probability model requires its own calibration and validation. Measure actual inference cost for two questions.

Add option-order and meaning-preserving wording tests. Any training-time option shuffle must remap labels and probabilities correctly. Keep counterfactual label-changing tests separate from transformations that should preserve the answer.

**Ready when:** target construction, precedence, mixed direct/indirect cases, output mapping, and reload behavior are covered by focused tests, with no gold metadata in the input.

## 4. Objective and configuration experiments

First compare the existing RLCD + cross-entropy recipe with cross-entropy alone using the same data, seed, initialization, effective batch size, and validation schedule. Treat superiority of either objective as an open question.

Save checkpoints at epochs 1, 2, and 4 so epoch comparisons can reuse one run. Treat early stopping as checkpoint selection, not a separate claimed objective improvement.

After the objective comparison, run a small learning-rate sweep: encoder `{1e-5, 2.5e-5}` and head `{5e-5, 1e-4}`. Keep weight decay 0.01 initially. Test a short warmup only after establishing the baseline. Avoid changing all parameters at once. Consider a frozen-encoder control if full fine-tuning still has a large training/validation gap.

Hard-negative sampling is a separate data ablation, with its exposure counts recorded. Do not combine arbitrary class weights with oversampling without measuring their effects; current class counts are already balanced.

**Ready when:** the runner produces comparable configurations, selected checkpoints, validation reports, and an experiment ledger identifying exactly which factor changed.

## 5. Starting weights and clinical baseline

| Candidate | Role | Required work |
| --- | --- | --- |
| `convaiinnovations/laya` | Current English initialization and main control | Pin revision; retrain on v2 |
| `convaiinnovations/laya-typed-decisions` | Alternative Laya initialization | Validate checkpoint loading; use the same v2 recipe and question |
| `thomas-sounack/BioClinical-ModernBERT-large` | Clinical encoder with a standard three-class classifier | Add training/inference adapter and evaluate through the same visit-level harness |
| Clinical encoder inside Laya | Follow-up architecture experiment | Audit hidden size, tokenizer/token IDs, state-dict mapping, positions, and head compatibility; retrain and calibrate the decision head |

The clinical classifier is a separate model baseline, not a drop-in Laya checkpoint. Changing the encoder name in a config does not replace the loaded encoder weights. Compare clinical and general ModernBERT classifiers under the same classifier recipe if attributing a gain specifically to clinical pretraining.

The clinical-Laya integration gets a compatibility report and prototype as part of readiness. A full run depends on passing its loading and training checks. The multilingual checkpoint becomes relevant if the actual notes require additional languages; it is not assumed to improve English attribution.

**Ready when:** each primary candidate loads, completes a training smoke run, exports, and returns the common prediction schema with recorded provenance and GPU memory.

## Experiment order

| Stage | Comparison | Decision |
| --- | --- | --- |
| A | Existing checkpoint on v2 validation | Establish baseline on the new distribution |
| B | v2 + current Laya initialization + current four-epoch recipe | Measure the effect of rebuilding data (training allocation changes too) |
| C | Best saved epoch from B; CE-only counterpart | Choose checkpoint policy and objective |
| D | Small learning-rate sweep using the chosen objective | Select a stable training recipe |
| E | Single question versus two questions | Test attribution decomposition |
| F | English versus typed-decisions initialization; general versus clinical encoder classifiers | Test weight and architecture choices |
| G | Combine compatible winning changes; clinical-Laya prototype if viable | Confirm interactions rather than assuming gains add together |
| H | Repeat the strongest two recipes with three seeds each | Measure training variability before final selection |
| I | Freeze finalists, fit calibration, run locked test once per finalist | Produce the final comparison |

This is a staged search rather than a full cross-product. All five workstreams are implemented, but expensive runs follow evidence from validation. A clinical classifier winning does not itself establish an improved Laya model; report those results separately.

## Evaluation and success criteria

Use new validation data for decisions. Existing challenge and counterfactual sets remain useful regression diagnostics, but their inspected errors must not serve as the only evidence of generalization.

Report macro F1; precision/recall per class; negative-to-direct and all negative-to-positive error rates; complete-counterfactual-group accuracy; high-confidence mistakes; NLL/Brier/ECE where probabilities are comparable; and latency/memory. Slice by negation, products, family history, causal actor, mixed encounters, historical use, abbreviations, length, and unseen scenario family.

Bootstrap comparisons by source scenario/counterfactual group (and patient for real data), not by treating correlated variants as independent visits. Show seed variability alongside sampling intervals.

Proposed development targets, to freeze before the sweep:

- Negative recall at least 0.90 on validation, with direct and indirect recall each no more than 0.02 below the existing checkpoint on that same validation set.
- Among candidates meeting those constraints, select by macro F1; use NLL and inference cost as documented tie-breakers where comparable. If no candidate qualifies, report the unmet target instead of silently relaxing it.
- Confirm the selected recipe against the existing checkpoint on the new locked test with group-aware confidence intervals. Targets are research objectives, not promised results or clinical deployment criteria.

Real-hospital validation requires separately authorized, de-identified notes with independent annotation/adjudication, patient isolation, and a later-time or external-site test. Public clinical corpora can overlap a clinical encoder's pretraining sources; account for that provenance when claiming external validation. Synthetic work can proceed while access and annotation are arranged.

## Implementation deliverables

- Dataset v2 schema, scenario bank, renderer interface, provenance manifest, split builder, and audit report.
- Configurable/resumable trainer with validation selection, objective controls, and reliable exports.
- Three-way and decomposed question adapters, plus the clinical-classifier adapter and clinical-Laya compatibility prototype.
- Experiment configurations and a sequential single-GPU runner with an explicit final-test stage.
- Shared benchmark/report generation and focused tests for leakage, targets, precedence, checkpoint selection/resume, calibration isolation, and inference parity.
- Detailed usage/results documentation outside the short root README. Publish code, configurations, and approved synthetic artifacts; keep hospital records and local weights out of Git.

Implementation order: data/schema and benchmark contracts -> trainer/validation -> model adapters -> smoke checks -> dataset freeze -> staged experiments -> final report. No accuracy improvement is claimed until measured.

## Research references

- [Requested Laya fine-tuning guide](https://laya-ai.com/guides/fine-tune-laya): overview and dataset recommendations; follow its links to upstream evidence.
- [Official Laya fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb): RLCD recipe and calibration workflow.
- [Laya browser fine-tuning case study](https://github.com/NandhaKishorM/laya/blob/main/docs/finetune_browser_agent.md): template shortcuts, counterexamples, and input-format effects on a different task.
- [Laya benchmarks](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md): calibration and option-order sensitivity.
- [Typed-decisions model card](https://huggingface.co/convaiinnovations/laya-typed-decisions): specialized training domains and objective.
- [BioClinical ModernBERT model card](https://huggingface.co/thomas-sounack/BioClinical-ModernBERT-large): clinical pretraining, downstream use, and reported benchmarks; no alcohol-attribution result.
- [Hugging Face Trainer documentation](https://huggingface.co/docs/transformers/main/main_classes/trainer): validation and best-checkpoint selection concepts.
- [Laya evaluation documentation](https://github.com/NandhaKishorM/laya/blob/main/docs/evals.md): slices, revision tracking, and regression comparison.
