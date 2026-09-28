"""Check and optionally build a Laya initialization with a clinical encoder.

The encoder transplant is experimental. It requires identical token IDs and
compatible encoder tensors. The decision head remains from the chosen Laya
checkpoint and must be fine-tuned and recalibrated for the target task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def probe(base_id: str, clinical_id: str) -> tuple[dict, Path]:
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    from transformers import AutoConfig, AutoTokenizer

    local_base = Path(base_id)
    base = local_base if local_base.is_dir() else Path(snapshot_download(
        base_id, allow_patterns=["model.safetensors", "rl_agent_config.json",
                                 "encoder/*", "tokenizer/*"], max_workers=2))
    _fix_tokenizer_config(str(base))
    base_cfg = AutoConfig.from_pretrained(base / "encoder")
    clinical_cfg = AutoConfig.from_pretrained(clinical_id)
    base_tok = AutoTokenizer.from_pretrained(base / "tokenizer")
    clinical_tok = AutoTokenizer.from_pretrained(clinical_id)
    checks = {
        "model_type": [base_cfg.model_type, clinical_cfg.model_type],
        "hidden_size": [base_cfg.hidden_size, clinical_cfg.hidden_size],
        "vocab_size": [base_cfg.vocab_size, clinical_cfg.vocab_size],
        "tokenizer_vocab_identical": base_tok.get_vocab() == clinical_tok.get_vocab(),
        "special_token_ids_identical": base_tok.all_special_ids == clinical_tok.all_special_ids,
    }
    checks["compatible_metadata"] = (all(pair[0] == pair[1] for key, pair in checks.items()
                                         if key in ("model_type", "hidden_size", "vocab_size"))
                                      and checks["tokenizer_vocab_identical"]
                                      and checks["special_token_ids_identical"])
    return checks, base


def build(base_id: str, clinical_id: str, output_dir: Path):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory must be empty")
    checks, base = probe(base_id, clinical_id)
    if not checks["compatible_metadata"]:
        raise ValueError(f"clinical encoder metadata/tokenizer incompatible: {checks}")
    from laya.common import build_model
    from safetensors.torch import load_file, save_file
    from transformers import AutoModel, AutoTokenizer
    from alcohol_ft.train import save_checkpoint
    import torch

    cfg = json.loads((base / "rl_agent_config.json").read_text(encoding="utf-8"))
    model = build_model(cfg, encoder_dir=str(base / "encoder"))
    model.load_state_dict(load_file(str(base / "model.safetensors")), strict=True)
    clinical = AutoModel.from_pretrained(clinical_id, attn_implementation="sdpa")
    base_state = model.encoder.state_dict()
    clinical_state = clinical.state_dict()
    if base_state.keys() != clinical_state.keys():
        missing = sorted(base_state.keys() - clinical_state.keys())[:8]
        extra = sorted(clinical_state.keys() - base_state.keys())[:8]
        raise ValueError(f"encoder keys differ; missing={missing}, extra={extra}")
    shape_differences = [key for key in base_state if base_state[key].shape != clinical_state[key].shape]
    if shape_differences:
        raise ValueError(f"encoder tensor shapes differ: {shape_differences[:8]}")
    model.encoder.load_state_dict(clinical_state, strict=True)
    cfg["encoder"] = clinical_id
    cfg["initialization"] = {"decision_head": base_id, "encoder": clinical_id,
                             "note": "experimental; requires downstream training and calibration"}
    cfg.pop("training", None)
    cfg.pop("temperature_by_options", None)
    cfg["temperature"] = [1.0, 1.0, 1.0]
    tokenizer = AutoTokenizer.from_pretrained(base / "tokenizer")
    save_checkpoint(model, tokenizer, cfg, output_dir, torch, save_file)
    (output_dir / "compatibility.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="convaiinnovations/laya")
    parser.add_argument("--clinical", default="thomas-sounack/BioClinical-ModernBERT-large")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checks = build(args.base, args.clinical, args.output) if args.output else probe(args.base, args.clinical)[0]
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
