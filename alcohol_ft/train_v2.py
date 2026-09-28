"""Configurable single-GPU Laya trainer for the v2 alcohol-attribution data.

This entry point has no import-time model download or GPU side effect. Run it
only when ready for an actual experiment. Calibration follows validation-based
checkpoint selection; the locked test split is never opened here.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

from alcohol_ft.benchmark import metrics
from alcohol_ft.data import read_cases, validate_splits
from alcohol_ft.task import LABELS, combine_components, combine_probabilities, questions_for, target_for
from alcohol_ft.train import collate, collect_logits, fit_temperature, save_checkpoint

DEFAULTS = {
    "data_dir": "data/alcohol_synthetic_v2", "model_id": "convaiinnovations/laya",
    "model_revision": None, "question_mode": "three_way", "objective": "rlcd_ce",
    "epochs": 4, "micro_batch": 8, "grad_accum": 4, "calibration_batch": 16,
    "max_len": 1024, "head_max_len": 256, "encoder_lr": 2.5e-5,
    "head_lr": 1e-4, "weight_decay": 0.01, "warmup_fraction": 0.0,
    "rl_weight": 1.0, "noise_samples": 4, "noise_start": 0.4,
    "noise_end": 0.1, "patience": 2, "seed": 42,
}


def load_config(path: Path) -> dict:
    supplied = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(supplied, dict):
        raise ValueError("experiment config must be a JSON object")
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    cfg = {**DEFAULTS, **supplied}
    if cfg["question_mode"] not in ("three_way", "decomposed"):
        raise ValueError("question_mode must be three_way or decomposed")
    if cfg["objective"] not in ("ce", "rlcd_ce"):
        raise ValueError("objective must be ce or rlcd_ce")
    for key in ("epochs", "micro_batch", "grad_accum", "calibration_batch",
                "max_len", "head_max_len", "noise_samples", "seed"):
        if type(cfg[key]) is not int or cfg[key] < (0 if key == "seed" else 1):
            raise ValueError(f"{key} must be a positive integer")
    for key in ("encoder_lr", "head_lr", "noise_start", "noise_end"):
        if not isinstance(cfg[key], (int, float)) or cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 <= cfg["warmup_fraction"] < 1 or cfg["weight_decay"] < 0 or cfg["rl_weight"] < 0:
        raise ValueError("invalid warmup, weight decay, or RL weight")
    if type(cfg["patience"]) is not int or cfg["patience"] < 1:
        raise ValueError("patience must be positive")
    return cfg


def prepare_items(rows: list[dict], tokenizer, model_cfg: dict, mode: str) -> list[dict]:
    from laya.common import QTYPES, build_sequence, render_options

    questions = questions_for(mode)
    definitions = {}
    for qid, question in questions.items():
        internal = {"t": "choice", "ins": question["instructions"],
                    "crit": question["criteria"]}
        empty, _ = build_sequence(tokenizer, "", internal,
                                  model_cfg["max_len"], model_cfg["head_max_len"])
        definitions[qid] = (internal, list(question["criteria"]), len(empty),
                            len(render_options(internal)))
    prepared = []
    for row in rows:
        state = row["state"]
        state_safe = state.replace(tokenizer.mask_token, " ") if tokenizer.mask_token else state
        state_ids = tokenizer(state_safe, add_special_tokens=False)["input_ids"]
        for qid, (internal, options, empty_len, marker_count) in definitions.items():
            budget = model_cfg["max_len"] - empty_len
            if len(state_ids) > budget:
                raise ValueError(f"{row['case_id']} / {qid}: {len(state_ids)} tokens exceed {budget}")
            ids, markers = build_sequence(tokenizer, state, internal,
                                          model_cfg["max_len"], model_cfg["head_max_len"],
                                          state_ids=state_ids)
            if len(ids) != empty_len + len(state_ids) or len(markers) != marker_count:
                raise ValueError(f"{row['case_id']} / {qid}: truncated sequence or options")
            index = options.index(target_for(row, qid))
            prepared.append({"ids": ids, "markers": markers, "qtype": QTYPES["choice"],
                             "target": [float(i == index) for i in range(len(options))],
                             "label": index, "case_id": row["case_id"], "qid": qid,
                             "visit_label": row["label"], "scenario": row.get("scenario", "other")})
    return prepared


def visit_predictions(items: list[dict], logits: list[list[float]], mode: str) -> list[dict]:
    import torch

    if len(items) != len(logits):
        raise ValueError("prediction count mismatch")
    by_case = {}
    for item, vector in zip(items, logits):
        case = by_case.setdefault(item["case_id"], {"expected": item["visit_label"],
                                                    "scenario": item["scenario"]})
        case[item["qid"]] = torch.softmax(torch.tensor(vector, dtype=torch.float64), -1).tolist()
    rows = []
    for case_id, case in by_case.items():
        if mode == "three_way":
            probabilities = dict(zip(LABELS, case["alcohol_attribution"]))
            predicted = max(probabilities, key=probabilities.get)
        else:
            p_patient = case["patient_alcohol"][0]
            p_other = case["other_person_causal"][0]
            probabilities = combine_probabilities(p_patient, p_other)
            predicted = combine_components(p_patient >= 0.5, p_other >= 0.5)
        rows.append({"case_id": case_id, "scenario": case["scenario"],
                     "expected": case["expected"], "predicted": predicted,
                     "probabilities": probabilities})
    return rows


def validation_report(model, items, tokenizer, batch_size, torch, device, mode):
    logits, _ = collect_logits(model, items, tokenizer.pad_token_id, batch_size, torch, device)
    rows = visit_predictions(items, logits, mode)
    report = metrics(rows)
    report["negative_to_direct"] = sum(r["expected"] == "negative" and r["predicted"] == "direct"
                                       for r in rows)
    report["question_mode"] = mode
    if mode == "decomposed":
        report["brier"] = None
        report["ece_10bin"] = None
        report["calibration_bins"] = []
        report["confidence_coverage"] = {}
        report["probability_note"] = "Joint visit probabilities are a heuristic; visit-level calibration is withheld."
    return report


def _training_loss(model, batch, torch, cfg, sigma):
    ids, att, pos, mask, target, qtype = batch
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, act = model(ids, att, pos, mask, qtype)
    logits = logits.float()
    mask = mask.bool()
    ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    if cfg["objective"] == "ce":
        return ce + 0.0 * act.sum()
    from laya.common import proper_reward

    count = mask.sum(-1, keepdim=True).float()
    eps = torch.randn((cfg["noise_samples"],) + logits.shape, device=logits.device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / count) * mask
    noisy = logits.detach().unsqueeze(0) + eps
    probabilities = torch.softmax(noisy.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        reward = proper_reward(probabilities, target.unsqueeze(0), qtype, mask,
                               w_sph=0.75, w_rps=1.0)
        advantage = reward - reward.mean(0, keepdim=True)
        advantage = advantage / (advantage.std() + 1e-6)
    log_probability = -(((noisy - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
    rl = -(advantage * log_probability).mean()
    return ce + cfg["rl_weight"] * rl + 0.0 * act.sum()


def _save_resume(path, model, optimizer, scheduler, epoch, best_epoch, best_score, stalls, torch):
    payload = {"epoch": epoch, "best_epoch": best_epoch, "best_score": best_score,
               "stalls": stalls, "model": model.state_dict(),
               "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
               "cuda_rng": torch.cuda.get_rng_state_all()}
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def run(config_path: Path, output_dir: Path, *, resume: bool = False) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise ValueError("output directory is nonempty; use a fresh path or --resume")
    cfg = load_config(config_path)
    data_dir = Path(cfg["data_dir"])
    paths = {key: data_dir / f"{key}.jsonl" for key in ("train", "validation", "calibration")}
    validate_splits(paths)
    train_rows, val_rows, cal_rows = (read_cases(paths[key]) for key in paths)

    import torch
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_model
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 20 * 1024**3:
        raise RuntimeError("Laya v2 training needs a CUDA GPU with at least 20 GiB")
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    local_model = Path(cfg["model_id"])
    model_dir = local_model if local_model.is_dir() else Path(snapshot_download(
        cfg["model_id"], revision=cfg["model_revision"],
        allow_patterns=["model.safetensors", "rl_agent_config.json",
                        "encoder/*", "tokenizer/*"], max_workers=2))
    _fix_tokenizer_config(str(model_dir))
    model_cfg = json.loads((model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    model_cfg.update(gradient_checkpointing=True, max_tokens_per_batch=4096,
                     max_len=cfg["max_len"], head_max_len=cfg["head_max_len"],
                     alcohol_question_mode=cfg["question_mode"])
    model_cfg["source_revision"] = model_dir.name if len(model_dir.name) == 40 else cfg["model_revision"]
    model_cfg.pop("training", None)
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    train_items = prepare_items(train_rows, tokenizer, model_cfg, cfg["question_mode"])
    val_items = prepare_items(val_rows, tokenizer, model_cfg, cfg["question_mode"])
    cal_items = prepare_items(cal_rows, tokenizer, model_cfg, cfg["question_mode"])
    model = build_model(model_cfg, encoder_dir=str(model_dir / "encoder"))
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    device = torch.device("cuda:0")
    model.to(device)
    encoder = [p for name, p in model.named_parameters() if name.startswith("encoder.")]
    head = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": cfg["encoder_lr"]},
                                   {"params": head, "lr": cfg["head_lr"]}],
                                  weight_decay=cfg["weight_decay"])
    updates_per_epoch = math.ceil(math.ceil(len(train_items) / cfg["micro_batch"]) / cfg["grad_accum"])
    total_updates = updates_per_epoch * cfg["epochs"]
    warmup = int(total_updates * cfg["warmup_fraction"])
    def schedule(step):
        if step < warmup:
            return max(1e-8, (step + 1) / max(1, warmup))
        progress = (step - warmup) / max(1, total_updates - warmup)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * min(1, progress))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = json.dumps(cfg, sort_keys=True)
    config_copy = output_dir / "experiment_config.json"
    if config_copy.exists() and json.dumps(json.loads(config_copy.read_text()), sort_keys=True) != canonical:
        raise ValueError("resume config differs from original experiment")
    config_copy.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    resume_file = output_dir / "resume.pt"
    start_epoch, best_epoch, best_score, stalls = 0, 0, -1.0, 0
    if resume:
        if not resume_file.exists():
            raise ValueError("resume state missing")
        state = torch.load(resume_file, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng"]])
        random.setstate(state["python_rng"])
        start_epoch, best_epoch, best_score, stalls = (state[k] for k in
                                                      ("epoch", "best_epoch", "best_score", "stalls"))
    reports = []
    report_path = output_dir / "validation_history.json"
    if report_path.exists():
        reports = json.loads(report_path.read_text(encoding="utf-8"))
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        epoch_started = time.perf_counter()
        order = list(range(len(train_items)))
        random.Random(cfg["seed"] + epoch).shuffle(order)
        sigma = cfg["noise_start"] + (cfg["noise_end"] - cfg["noise_start"]) * epoch / max(1, cfg["epochs"] - 1)
        optimizer.zero_grad(set_to_none=True)
        for offset in range(0, len(order), cfg["micro_batch"]):
            chunk = [train_items[i] for i in order[offset:offset + cfg["micro_batch"]]]
            batch = tuple(v.to(device) for v in collate(chunk, tokenizer.pad_token_id, torch))
            loss = _training_loss(model, batch, torch, cfg, sigma) / cfg["grad_accum"]
            loss.backward()
            final = offset + cfg["micro_batch"] >= len(order)
            if ((offset // cfg["micro_batch"] + 1) % cfg["grad_accum"] == 0) or final:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            batches_done = offset // cfg["micro_batch"] + 1
            if batches_done % 250 == 0:
                print(f"epoch {epoch+1}: {batches_done}/{math.ceil(len(order) / cfg['micro_batch'])} "
                      f"batches, elapsed={time.perf_counter() - epoch_started:.0f}s", flush=True)
        report = validation_report(model, val_items, tokenizer, cfg["calibration_batch"],
                                   torch, device, cfg["question_mode"])
        report.update(epoch=epoch + 1, peak_gpu_bytes=torch.cuda.max_memory_allocated())
        reports.append(report)
        report_path.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
        model_cfg["training"] = {"epochs_completed": epoch + 1, "objective": cfg["objective"],
                                  "seed": cfg["seed"]}
        epoch_dir = output_dir / f"epoch_{epoch + 1}"
        save_checkpoint(model, tokenizer, model_cfg, epoch_dir, torch, save_file, epoch=epoch + 1)
        score = report["macro_f1"]
        if score > best_score:
            best_epoch, best_score, stalls = epoch + 1, score, 0
        else:
            stalls += 1
        _save_resume(resume_file, model, optimizer, scheduler, epoch + 1,
                     best_epoch, best_score, stalls, torch)
        print(f"epoch {epoch+1}: macro_f1={score:.4f}, negative_recall="
              f"{report['per_label']['negative']['recall']:.4f}, best={best_epoch}", flush=True)
        if stalls >= cfg["patience"]:
            break
    if not best_epoch:
        raise RuntimeError("no validation checkpoint selected")
    best_dir = output_dir / f"epoch_{best_epoch}"
    model.load_state_dict(load_file(str(best_dir / "model.safetensors")), strict=True)
    logits, labels = collect_logits(model, cal_items, tokenizer.pad_token_id,
                                    cfg["calibration_batch"], torch, device)
    raw_temperature = fit_temperature(logits, labels, torch)
    temperature = min(5.0, max(0.5, raw_temperature))
    model_cfg["training"] = {"epochs_completed": best_epoch, "objective": cfg["objective"],
                              "seed": cfg["seed"], "selected_by": "validation_macro_f1"}
    final_dir = output_dir / "final"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    save_checkpoint(model, tokenizer, model_cfg, final_dir, torch, save_file,
                    temperature=temperature)
    (output_dir / "selection.json").write_text(json.dumps({
        "selected_epoch": best_epoch, "validation_macro_f1": best_score,
        "calibration_cases": len(cal_rows), "calibration_questions": len(cal_items),
        "raw_temperature": raw_temperature, "export_temperature": temperature,
        "question_mode": cfg["question_mode"], "model_id": cfg["model_id"],
        "model_revision_requested": cfg["model_revision"],
        "model_revision_resolved": model_cfg["source_revision"]}, indent=2) + "\n", encoding="utf-8")
    print(f"prepared final checkpoint: {final_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.config, args.output_dir, resume=args.resume)


if __name__ == "__main__":
    main()
