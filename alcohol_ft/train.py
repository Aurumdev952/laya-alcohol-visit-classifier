"""Single-GPU Laya RLCD training, adapted from the upstream typed-decisions notebook."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from alcohol_ft.data import read_cases, validate_splits
from alcohol_ft.task import LABELS, QUESTION


def prepare_items(rows, tokenizer, cfg):
    from laya.common import QTYPES, build_sequence, render_options

    question = QUESTION["alcohol_attribution"]
    internal = {"t": "choice", "ins": question["instructions"], "crit": question["criteria"]}
    empty, _ = build_sequence(tokenizer, "", internal, cfg["max_len"], cfg["head_max_len"])
    budget = cfg["max_len"] - len(empty)
    if budget <= 0:
        raise ValueError("question exceeds model input window")
    items = []
    for row in rows:
        state = row["state"]
        safe_state = state.replace(tokenizer.mask_token, " ") if tokenizer.mask_token else state
        state_tokens = len(tokenizer(safe_state, add_special_tokens=False)["input_ids"])
        if state_tokens > budget:
            raise ValueError(f"{row['case_id']}: {state_tokens} state tokens exceed {budget}; refusing truncation")
        seq, markers = build_sequence(tokenizer, state, internal, cfg["max_len"], cfg["head_max_len"])
        if len(seq) != len(empty) + state_tokens:
            raise ValueError(f"{row['case_id']}: state was truncated during sequence construction")
        if len(markers) != len(render_options(internal)):
            raise ValueError(f"{row['case_id']}: incomplete option markers")
        label = LABELS.index(row["label"])
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES["choice"],
                      "target": [float(i == label) for i in range(len(LABELS))],
                      "label": label})
    return items


def collate(items, pad_id, torch):
    n = len(items)
    width = max(len(x["ids"]) for x in items)
    ids = torch.full((n, width), pad_id, dtype=torch.long)
    att = torch.zeros((n, width), dtype=torch.long)
    markers = torch.tensor([x["markers"] for x in items], dtype=torch.long)
    target = torch.tensor([x["target"] for x in items], dtype=torch.float32)
    for i, item in enumerate(items):
        ids[i, :len(item["ids"])] = torch.tensor(item["ids"], dtype=torch.long)
        att[i, :len(item["ids"])] = 1
    return ids, att, markers, torch.ones_like(markers, dtype=torch.bool), target, torch.full((n,), items[0]["qtype"], dtype=torch.long)


def fit_temperature(logits, labels, torch):
    z = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.long)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().clamp(0.1, 10.0).item())


def collect_logits(model, items, pad_id, batch_size, torch, device):
    result, labels = [], []
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for offset in range(0, len(items), batch_size):
            batch = collate(items[offset:offset + batch_size], pad_id, torch)
            ids, att, pos, mask, _, qtype = [value.to(device) for value in batch]
            logits, _ = model(ids, att, pos, mask, qtype)
            result.extend(logits.float().cpu().tolist())
            labels.extend(item["label"] for item in items[offset:offset + batch_size])
    return result, labels


def save_checkpoint(model, tokenizer, cfg, output_dir, torch, save_file, *, temperature=None, epoch=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = {name: value.detach().half().contiguous().cpu()
               for name, value in model.state_dict().items()}
    save_file(weights, str(output_dir / "model.safetensors"))
    model.encoder.config.save_pretrained(output_dir / "encoder")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    export = dict(cfg)
    export["fine_tuned"] = True
    export["model_name"] = "laya-alcohol-visit"
    if temperature is not None:
        temps = list(export.get("temperature", [1.0, 1.0, 1.0]))
        while len(temps) < 3:
            temps.append(1.0)
        temps[0] = temperature
        export["temperature"] = temps
        export.pop("temperature_by_options", None)
    (output_dir / "rl_agent_config.json").write_text(json.dumps(export, indent=2) + "\n", encoding="utf-8")
    if epoch is not None:
        (output_dir / "training_meta.json").write_text(json.dumps({"epoch": epoch}, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/alcohol_synthetic"))
    parser.add_argument("--model-id", default="convaiinnovations/laya")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/laya-alcohol"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--calibration-batch", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--head-max-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.micro_batch, args.grad_accum, args.calibration_batch) < 1:
        parser.error("epochs and batch sizes must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory is not empty; choose a fresh run directory")
    paths = {split: args.data_dir / f"{split}.jsonl" for split in ("train", "calibration", "test")}
    print("Data validation:", validate_splits(paths), flush=True)

    import torch
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_model, proper_reward
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; training requires the RTX 5090 driver/runtime")
    device = torch.device("cuda:0")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    if torch.cuda.get_device_properties(0).total_memory < 20 * 1024**3:
        raise RuntimeError("This preset expects at least 20 GiB of GPU memory")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    model_dir = Path(snapshot_download(
        args.model_id,
        allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*"],
        max_workers=2,
    ))
    _fix_tokenizer_config(str(model_dir))
    cfg = json.loads((model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    cfg.update(gradient_checkpointing=True, max_tokens_per_batch=4096,
               max_len=args.max_len, head_max_len=args.head_max_len)
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    train = prepare_items(read_cases(paths["train"]), tokenizer, cfg)
    calibration = prepare_items(read_cases(paths["calibration"]), tokenizer, cfg)
    model = build_model(cfg, encoder_dir=str(model_dir / "encoder"))
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device)
    model.train()
    enc = [p for name, p in model.named_parameters() if name.startswith("encoder.")]
    head = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW([{"params": enc, "lr": 2.5e-5},
                                   {"params": head, "lr": 1e-4}], weight_decay=0.01)
    updates_per_epoch = math.ceil(math.ceil(len(train) / args.micro_batch) / args.grad_accum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=updates_per_epoch * args.epochs, eta_min=1e-6)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8")
    step = 0
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(train)
        sigma = 0.4 + (0.1 - 0.4) * epoch / max(1, args.epochs - 1)
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(train), args.micro_batch):
            chunk = train[start:start + args.micro_batch]
            batch = collate(chunk, tokenizer.pad_token_id, torch)
            ids, att, pos, mask, target, qtype = [value.to(device) for value in batch]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, act = model(ids, att, pos, mask, qtype)
            logits = logits.float()
            mask = mask.bool()
            k = mask.sum(-1, keepdim=True).float()
            eps = torch.randn((4,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            noisy = logits.detach().unsqueeze(0) + eps
            probs = torch.softmax(noisy.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                reward = proper_reward(probs, target.unsqueeze(0), qtype, mask,
                                       w_sph=0.75, w_rps=1.0)
                advantage = reward - reward.mean(0, keepdim=True)
                advantage = advantage / (advantage.std() + 1e-6)
            logp = -(((noisy - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
            rl = -(advantage * logp).mean()
            ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (rl + ce + 0.0 * act.sum()) / args.grad_accum
            loss.backward()
            end = start + args.micro_batch >= len(train)
            if ((start // args.micro_batch + 1) % args.grad_accum == 0) or end:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 100 == 0:
                    print(f"epoch={epoch+1} update={step} loss={(loss.item()*args.grad_accum):.4f}", flush=True)
        save_checkpoint(model, tokenizer, cfg, args.output_dir / "checkpoint_latest", torch, save_file, epoch=epoch + 1)
        print(f"completed epoch {epoch+1}/{args.epochs}", flush=True)
    del optimizer, scheduler
    torch.cuda.empty_cache()
    logits, labels = collect_logits(model, calibration, tokenizer.pad_token_id,
                                    args.calibration_batch, torch, device)
    temperature = fit_temperature(logits, labels, torch)
    save_checkpoint(model, tokenizer, cfg, args.output_dir / "final", torch, save_file,
                    temperature=temperature)
    (args.output_dir / "calibration.json").write_text(
        json.dumps({"cases": len(labels), "choice_temperature": temperature,
                    "source": str(paths["calibration"])}, indent=2) + "\n", encoding="utf-8")
    print(f"Final checkpoint: {args.output_dir / 'final'}; temperature={temperature:.4f}", flush=True)


if __name__ == "__main__":
    main()
