# Laya hospital visit alcohol classifier

Fine-tune [Laya](https://github.com/NandhaKishorM/laya) to classify a hospital visit from all of its encounter notes as **direct alcohol positive**, **indirect alcohol positive** (another person's drinking caused the presentation), or **negative**. This repository includes a 30,000-visit synthetic dataset, training and calibration scripts, held-out benchmarks, and a separate Laya sentiment/decision API.

The included checkpoint must be trained locally; model weights are not stored in Git. The published datasets are synthetic and do not establish clinical accuracy.

## Run on an RTX 5090

Requires Linux, Python 3.10–3.13, a working NVIDIA driver, and an RTX 5090 with a compatible CUDA runtime.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pip install -e '.[train,test]'
.venv/bin/python -c 'import torch; assert torch.cuda.is_available()'
```

Train and evaluate the alcohol classifier:

```bash
.venv/bin/python -m alcohol_ft.train --output-dir runs/laya-alcohol-001
.venv/bin/python -m alcohol_ft.benchmark --finetuned runs/laya-alcohol-001/final --output runs/laya-alcohol-001/benchmark.json
.venv/bin/python -m eval.alcohol.generalization_eval --finetuned runs/laya-alcohol-001/final --output runs/laya-alcohol-001/generalization.json
.venv/bin/python -m alcohol_ft.predict --model runs/laya-alcohol-001/final --input eval/alcohol/curated.jsonl
```

Run the included sentiment and decision API:

```bash
.venv/bin/python -m uvicorn sentiment_api.app:app --host 127.0.0.1 --port 8000 --workers 1
```

## Documentation

- [Alcohol task, data format, training, and benchmark](alcohol_ft/README.md)
- [Measured alcohol run and limitations](alcohol_ft/RUN_RESULTS.md)
- [Sentiment and decision API](docs/SENTIMENT_API.md)

Run tests with `.venv/bin/python -m pytest -q tests/test_alcohol_ft.py tests/test_alcohol_generalization.py`.
