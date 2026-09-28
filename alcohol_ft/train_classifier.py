"""Train a standard ModernBERT visit classifier as a controlled baseline.

This is a separate Transformers classifier, not a Laya checkpoint. No model is
loaded until the CLI is run for a training experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from alcohol_ft.benchmark import metrics
from alcohol_ft.data import read_cases, validate_splits
from alcohol_ft.task import LABELS
from alcohol_ft.train import fit_temperature

DEFAULTS = {
    "data_dir": "data/alcohol_synthetic_v2",
    "model_id": "thomas-sounack/BioClinical-ModernBERT-large",
    "model_revision": None, "epochs": 4, "micro_batch": 4,
    "grad_accum": 8, "eval_batch": 8, "max_len": 1024,
    "learning_rate": 2e-5, "weight_decay": 0.01,
    "warmup_fraction": 0.05, "patience": 2, "seed": 42,
}


def load_config(path: Path) -> dict:
    supplied = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError("invalid classifier experiment configuration")
    cfg = {**DEFAULTS, **supplied}
    for key in ("epochs", "micro_batch", "grad_accum", "eval_batch", "max_len", "patience"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be positive")
    if not 0 < cfg["learning_rate"] or not 0 <= cfg["weight_decay"] or not 0 <= cfg["warmup_fraction"] < 1:
        raise ValueError("invalid optimizer settings")
    return cfg


def encode(rows, tokenizer, max_len):
    items = []
    for row in rows:
        ids = tokenizer(row["state"], add_special_tokens=True)["input_ids"]
        if len(ids) > max_len:
            raise ValueError(f"{row['case_id']}: {len(ids)} tokens exceed {max_len}")
        items.append({"ids": ids, "label": LABELS.index(row["label"]),
                      "case_id": row["case_id"], "scenario": row.get("scenario", "other")})
    return items


def collate(items, pad_id, torch):
    width = max(len(item["ids"]) for item in items)
    ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = torch.tensor([item["label"] for item in items], dtype=torch.long)
    for i, item in enumerate(items):
        ids[i, :len(item["ids"])] = torch.tensor(item["ids"], dtype=torch.long)
        mask[i, :len(item["ids"])] = 1
    return ids, mask, labels


def collect(model, items, pad_id, batch_size, torch, device):
    logits = []
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for offset in range(0, len(items), batch_size):
            ids, mask, _ = (v.to(device) for v in collate(items[offset:offset + batch_size], pad_id, torch))
            logits.extend(model(input_ids=ids, attention_mask=mask).logits.float().cpu().tolist())
    return logits


def validation_report(model, items, pad_id, batch_size, torch, device):
    logits = collect(model, items, pad_id, batch_size, torch, device)
    rows = []
    for item, values in zip(items, logits):
        probs = torch.softmax(torch.tensor(values, dtype=torch.float64), -1).tolist()
        rows.append({"case_id": item["case_id"], "scenario": item["scenario"],
                     "expected": LABELS[item["label"]], "predicted": LABELS[max(range(3), key=probs.__getitem__)],
                     "probabilities": dict(zip(LABELS, probs))})
    report = metrics(rows)
    report["negative_to_direct"] = sum(r["expected"] == "negative" and r["predicted"] == "direct"
                                       for r in rows)
    return report


def run(config_path: Path, output_dir: Path, *, resume: bool = False):
    cfg = load_config(config_path)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise ValueError("output directory is nonempty")
    data_dir = Path(cfg["data_dir"])
    paths = {part: data_dir / f"{part}.jsonl" for part in ("train", "validation", "calibration")}
    validate_splits(paths)
    train_rows, val_rows, cal_rows = (read_cases(paths[part]) for part in paths)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 20 * 1024**3:
        raise RuntimeError("classifier training needs a CUDA GPU with at least 20 GiB")
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["model_revision"])
    train_items = encode(train_rows, tokenizer, cfg["max_len"])
    val_items = encode(val_rows, tokenizer, cfg["max_len"])
    cal_items = encode(cal_rows, tokenizer, cfg["max_len"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"], num_labels=3,
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id={label: i for i, label in enumerate(LABELS)},
        ignore_mismatched_sizes=True,
    )
    source_revision = getattr(model.config, "_commit_hash", None)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    device = torch.device("cuda:0")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    updates_per_epoch = math.ceil(math.ceil(len(train_items) / cfg["micro_batch"]) / cfg["grad_accum"])
    total_updates = updates_per_epoch * cfg["epochs"]
    warmup = int(total_updates * cfg["warmup_fraction"])
    def schedule(step):
        if step < warmup:
            return max(1e-8, (step + 1) / max(1, warmup))
        progress = (step - warmup) / max(1, total_updates - warmup)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * min(progress, 1))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = output_dir / "experiment_config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != cfg:
        raise ValueError("resume config differs from original experiment")
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    start_epoch, best_epoch, best_score, stalls = 0, 0, -1.0, 0
    resume_path = output_dir / "resume.pt"
    if resume:
        if not resume_path.exists():
            raise ValueError("resume state missing")
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        start_epoch, best_epoch, best_score, stalls = (state[k] for k in
                                                      ("epoch", "best_epoch", "best_score", "stalls"))
    history_path = output_dir / "validation_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        order = list(range(len(train_items)))
        random.Random(cfg["seed"] + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        for offset in range(0, len(order), cfg["micro_batch"]):
            batch_items = [train_items[i] for i in order[offset:offset + cfg["micro_batch"]]]
            ids, mask, labels = (v.to(device) for v in collate(batch_items, tokenizer.pad_token_id, torch))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            (loss / cfg["grad_accum"]).backward()
            final = offset + cfg["micro_batch"] >= len(order)
            if ((offset // cfg["micro_batch"] + 1) % cfg["grad_accum"] == 0) or final:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        report = validation_report(model, val_items, tokenizer.pad_token_id,
                                   cfg["eval_batch"], torch, device)
        report.update(epoch=epoch + 1, peak_gpu_bytes=torch.cuda.max_memory_allocated())
        history.append(report)
        history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        epoch_dir = output_dir / f"epoch_{epoch + 1}"
        model.save_pretrained(epoch_dir, safe_serialization=True)
        tokenizer.save_pretrained(epoch_dir)
        if report["macro_f1"] > best_score:
            best_epoch, best_score, stalls = epoch + 1, report["macro_f1"], 0
        else:
            stalls += 1
        state = {"epoch": epoch + 1, "best_epoch": best_epoch, "best_score": best_score,
                 "stalls": stalls, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all(), "python_rng": random.getstate()}
        temp = resume_path.with_suffix(".tmp")
        torch.save(state, temp)
        temp.replace(resume_path)
        print(f"epoch {epoch+1}: macro_f1={report['macro_f1']:.4f}, best={best_epoch}", flush=True)
        if stalls >= cfg["patience"]:
            break
    if not best_epoch:
        raise RuntimeError("no validation checkpoint selected")
    best_dir = output_dir / f"epoch_{best_epoch}"
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    logits = collect(model, cal_items, tokenizer.pad_token_id, cfg["eval_batch"], torch, device)
    raw_temp = fit_temperature(logits, [x["label"] for x in cal_items], torch)
    temperature = min(5.0, max(0.5, raw_temp))
    final_dir = output_dir / "final"
    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    (final_dir / "alcohol_config.json").write_text(json.dumps({
        "model_family": "transformers_sequence_classifier", "temperature": temperature,
        "max_len": cfg["max_len"], "question_mode": "three_way"}, indent=2) + "\n", encoding="utf-8")
    (output_dir / "selection.json").write_text(json.dumps({
        "selected_epoch": best_epoch, "validation_macro_f1": best_score,
        "calibration_cases": len(cal_items), "raw_temperature": raw_temp,
        "export_temperature": temperature, "model_id": cfg["model_id"],
        "model_revision_requested": cfg["model_revision"],
        "model_revision_resolved": source_revision}, indent=2) + "\n", encoding="utf-8")
    print(f"prepared final classifier: {final_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.config, args.output_dir, resume=args.resume)


if __name__ == "__main__":
    main()
