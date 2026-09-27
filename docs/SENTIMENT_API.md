# Sentiment and decision API

A local English sentiment and decision-recipe API powered by [Laya](https://github.com/NandhaKishorM/laya). It uses the English checkpoint on an NVIDIA GPU. The sentiment endpoint returns positive, neutral, or negative with Laya's per-label probabilities; six additional endpoints implement the published decision recipes. It requires CUDA; startup fails if the model cannot run on the GPU.

## Set up on an RTX 5090

Use Linux with a working NVIDIA driver, Python 3.10–3.13, and internet access for the first checkpoint download. Check that `nvidia-smi` lists the RTX 5090. PyTorch's RTX 50-series support requires a CUDA 12.8 or newer build; the commands below select its CUDA 13.0 wheel. Ensure your installed driver supports that CUDA runtime.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.version.cuda)'
```

The first server start downloads the `convaiinnovations/laya` English checkpoint from Hugging Face into the standard cache. Later starts can use the cached weights.

```bash
.venv/bin/python -m uvicorn sentiment_api.app:app --host 127.0.0.1 --port 8000 --workers 1
```

## Use

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/sentiment \
  -H 'Content-Type: application/json' \
  -d '{"text":"The service was excellent and I would recommend it."}'
```

Example response shape (numbers depend on the model output):

```json
{
  "sentiment": "positive",
  "probabilities": {"positive": 0.8, "neutral": 0.15, "negative": 0.05}
}
```

`POST /sentiment` accepts one nonempty `text` string. Empty text or text exceeding the checkpoint's available token window returns HTTP 422. If Laya loses CUDA during inference, the request returns HTTP 503 and `/health` reports unavailable. The API is limited to English and is intended for local use. The probabilities are raw Laya outputs; evaluate accuracy and calibration on your own examples before using them as decision thresholds.

## Decision recipes

The same CUDA model also serves the six [Laya AI recipes](https://laya-ai.com/recipes). Send JSON to these endpoints; every field shown is required and must contain nonblank text. Each response has an `answers` object keyed by the recipe's typed questions. Choice answers include `choice` and per-option `probabilities`; score answers include the expected numeric `score`, a `legend`, and per-level `probabilities`; `noul` answers include `noul`, the model's P(true). The service preserves the other metadata returned by Laya for each answer.

| Endpoint | Request JSON fields | Answer fields |
| --- | --- | --- |
| `POST /recipes/support-ticket-triage` | `body` | `department`, `urgency`, `refund_requested` |
| `POST /recipes/model-routing` | `request` | `model_tier`, `needs_reasoning` |
| `POST /recipes/llm-guardrails` | `prompt` | `jailbreak`, `escalate` |
| `POST /recipes/rag-filtering` | `question`, `passage` | `relevance`, `keep` |
| `POST /recipes/content-moderation` | `message` | `moderation_action`, `threat` |
| `POST /recipes/phishing-detection` | `subject`, `body` | `email_type`, `credential_risk`, `payment_risk` |

For example:

```bash
curl -X POST http://127.0.0.1:8000/recipes/support-ticket-triage \
  -H 'Content-Type: application/json' \
  -d '{"body":"I was charged twice. Please refund the duplicate."}'
```

The fixed question definitions are in [sentiment_api/recipes.py](../sentiment_api/recipes.py). All questions for one recipe run in one Laya call. States that would exceed the shortest question's token window return HTTP 422 instead of being truncated. CUDA loss returns HTTP 503. These endpoints return decision signals only; your application chooses and validates any routing or review thresholds. In particular, do not use the guardrails, moderation, or phishing outputs as a sole safety or security control without task-specific evaluation and calibration. This service uses the existing English checkpoint, while the site's examples use Laya's automatic Router; the two may differ on typed tasks. The published recipe index was used for the guardrails and phishing workflows because their detail pages were unavailable during implementation.

Run all tests after installing the `test` extra:

```bash
.venv/bin/python -m pytest -q
```

## Run the 50-case evaluation

With the API running on the RTX 5090, run:

```bash
.venv/bin/python -m eval.run_eval
```

The runner verifies that `/health` reports CUDA and a GPU name containing `5090`, then sends all 50 hand-labeled English examples from [eval/cases.jsonl](../eval/cases.jsonl). It prints accuracy and macro F1, and saves predictions, per-label metrics, the confusion matrix, and mistakes to `eval/results.json`. These are curated smoke-test examples, not a representative accuracy benchmark.

On September 25, 2026, the pinned stack ran on an NVIDIA GeForce RTX 5090 Laptop GPU and scored **46/50 (92.0% accuracy, 0.9185 macro F1)**. All 17 negative examples were correct; the four mistakes were one positive classified as neutral and three neutral classified as positive. The full output is in [eval/results.json](../eval/results.json). If port 8000 is occupied, start Uvicorn on another port and pass it to the runner, for example `--base-url http://127.0.0.1:8001`.

To repeat the same evaluation 10 times and calculate mean accuracy and macro F1:

```bash
.venv/bin/python -m eval.repeat_eval --base-url http://127.0.0.1:8001 --runs 10
```

The report is saved to [eval/repeated_results.json](../eval/repeated_results.json). On September 25, 2026, all 10 runs scored 46/50: mean accuracy **92.0%** and mean macro F1 **0.9185**, with zero observed standard deviation and no changed predictions. Repeated runs measure prediction stability on these fixed cases; they do not estimate accuracy on new data.

The service tests also run without third-party packages:

```bash
python3 -m unittest discover -s tests -p 'test_service.py' -v
python3 -m unittest discover -s tests -p 'test_eval.py' -v
```

## Sector classification evaluations

Five additional balanced datasets test routing messages to the right team. Each has 50 synthetic English questions, with 10 examples for each of five categories. The fixed task descriptions and cases are in [eval/sectors](../eval/sectors). These use the same unquantized English Laya checkpoint directly on CUDA, in batches of eight; they do not call the sentiment endpoint.

| Sector | Categories | Correct | Macro F1 |
| --- | --- | ---: | ---: |
| Education | admissions, coursework, fees and aid, technology, student support | 45/50 (90%) | 0.9062 |
| Healthcare operations | appointments, billing and insurance, records, prescriptions, care team | 47/50 (94%) | 0.9424 |
| Mining | safety, equipment, geology, logistics, environment | 47/50 (94%) | 0.9370 |
| Agriculture | crops, irrigation, machinery, logistics, farm finance | 47/50 (94%) | 0.9412 |
| Banking support | accounts, payments, cards, loans, security | 41/50 (82%) | 0.8222 |

Overall: **227/250 (90.8%)**, with mean sector macro F1 **0.9098** on the NVIDIA GeForce RTX 5090 Laptop GPU. Banking account messages were frequently routed to security; environmental mining reports were sometimes routed to safety or geology. The complete [sector report](../eval/sectors/results.json) contains all predictions, probabilities, per-category metrics, confusion matrices, and errors.

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m eval.sector_eval
```

Omit `HF_HUB_OFFLINE=1` on the first run so the checkpoint can download. These hand-labeled examples are a smoke test, not a representative sector benchmark. The healthcare task evaluates message routing only, never urgency or diagnosis; banking labels are support topics, not financial decisions.

## Structured extraction evaluations

The [structured evaluation datasets](../eval/structured) contain 50 hand-labeled messages each for support tickets, education requests, and mining maintenance notes. Each task extracts three bounded JSON fields: a category, a small integer, and an explicit yes/no request. The runner uses [Laya's JSON-schema question mapping](https://nandhakishorm.github.io/laya/structured/) directly with the unquantized checkpoint. This tests constrained-field extraction, not arbitrary names, dates, or free-text spans.

```bash
HF_HOME="$PWD/.hf-cache" HF_HUB_OFFLINE=1 .venv/bin/python -m eval.structured_eval --require-gpu 5090
```

The runner refuses to run if CUDA or the required GPU is unavailable. It writes exact-record accuracy, per-field accuracy, confusion matrices, and all predictions to [eval/structured/results.json](../eval/structured/results.json). Omit `HF_HUB_OFFLINE=1` if the checkpoint has not been downloaded yet. The examples are synthetic smoke tests, not representative production data.

On September 27, 2026, the unquantized checkpoint ran on the RTX 5090 Laptop GPU in bfloat16. Exact-record accuracy was **94/150 (62.7%)**; field accuracy was **379/450 (84.2%)**:

| Task | Exact records | Field accuracy |
| --- | ---: | ---: |
| Support tickets | 24/50 (48%) | 78.0% |
| Education requests | 29/50 (58%) | 80.7% |
| Mining maintenance | 41/50 (82%) | 94.0% |

The weakest field was support-ticket urgency (54%); mining asset identification was 100%. Do not use these synthetic-case results as production accuracy estimates.

## GPU API concurrency benchmark

Run a bounded sweep against one local API worker on the 5090. The launcher starts and stops the server automatically:

```bash
HF_HOME="$PWD/.hf-cache" HF_HUB_OFFLINE=1 .venv/bin/python -m eval.run_concurrency_local \
  --concurrency 1,2,4,8,16,32,64 --requests 200 --p95-target-ms 200
```

The benchmark requires `/health` to identify CUDA and a 5090. It records successful requests per second, error counts, and p50/p95/p99 latency at every load level in [the initial report](../eval/concurrency_results.json). “Practical concurrency” is the highest tested load with no errors and p95 latency at or below the chosen 200 ms target; it is not an intrinsic GPU maximum. The API serializes model inference with a lock, so this measures capacity of this deployment, including HTTP overhead and queueing.

On September 27, 2026, **6 concurrent clients** met the 200 ms p95 target in every one of three boundary runs (p95 154–184 ms), with 200 or 400 requests per level. Seven clients passed in two of three runs; eight passed in one of four. At 64 concurrent clients, all 200 requests completed without errors, but p95 latency was 1.80 seconds. The broad sweep and [three](../eval/concurrency_refined_results.json) [boundary](../eval/concurrency_confirm_1.json) [reports](../eval/concurrency_confirm_2.json) preserve the raw metrics. Treat six as a conservative limit for this deployment and workload, not a guaranteed capacity or a general hardware specification. Longer texts, other endpoints, background GPU work, and thermal/clock changes will alter it.
