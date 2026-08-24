#!/usr/bin/env python3
"""HF deployment preflight for ADTC v5 AfriqueQwen candidates.

Checks config/tokenizer, optional 4-bit load + PEFT attach + short generate.
Writes docs/artifacts/v5/model_preflight.json (merge-friendly).

Example:
  python preflight_v5_model.py --alias qwen35_extcm --config-only
  python preflight_v5_model.py --alias qwen35_extcm --load-4bit --generate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_loader import inspect_config, load_pretrained_model, load_tokenizer, suggest_lora_targets  # noqa: E402

MANIFEST = Path(__file__).resolve().parent / "configs" / "v5_models.yaml"
OUT_DEFAULT = ROOT / "docs" / "artifacts" / "v5" / "model_preflight.json"


def load_manifest() -> dict:
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alias", required=True, help="Key in v5_models.yaml")
    ap.add_argument("--config-only", action="store_true")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    manifest = load_manifest()
    if args.alias not in manifest["models"]:
        raise SystemExit(f"unknown alias {args.alias}; have {list(manifest['models'])}")
    spec = manifest["models"][args.alias]
    hf_id = spec["hf_id"]
    model_class = spec.get("model_class", "auto")

    entry: dict = {
        "alias": args.alias,
        "hf_id": hf_id,
        "model_class": model_class,
        "role": spec.get("role"),
        "status": "pending",
        "architecture": None,
        "parameter_count": None,
        "tokenizer_vocab": None,
        "context_length": None,
        "chat_template_present": None,
        "has_vision_config": None,
        "bnb_4bit_load": None,
        "peft_qlora_possible": None,
        "generate_ok": None,
        "errors": [],
        "warnings": [],
    }

    try:
        meta = inspect_config(hf_id)
        entry["architecture"] = meta.get("architectures") or meta.get("model_type")
        entry["tokenizer_vocab"] = meta.get("vocab_size")
        entry["context_length"] = meta.get("max_position_embeddings")
        entry["chat_template_present"] = bool(meta.get("chat_template_present"))
        entry["has_vision_config"] = bool(meta.get("has_vision_config"))
        entry["config"] = meta
        if entry["has_vision_config"]:
            entry["warnings"].append(
                "HF checkpoint has vision_config; competition requires single text GGUF without mandatory mmproj"
            )
    except Exception as e:  # noqa: BLE001
        entry["errors"].append(f"inspect_config: {type(e).__name__}: {e}")
        entry["status"] = "error"
        _write_merge(args.out, entry)
        raise SystemExit(1) from e

    if args.config_only:
        entry["status"] = "config_ok"
        _write_merge(args.out, entry)
        print(json.dumps(entry, indent=2))
        return

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import BitsAndBytesConfig

        tok = load_tokenizer(hf_id)
        kwargs = {"model_class": model_class, "device_map": "auto"}
        if args.load_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16

        t0 = time.time()
        model = load_pretrained_model(hf_id, **kwargs)
        entry["load_seconds"] = round(time.time() - t0, 2)
        entry["bnb_4bit_load"] = bool(args.load_4bit)
        try:
            entry["parameter_count"] = count_parameters(model)
        except Exception:  # noqa: BLE001
            entry["parameter_count"] = None

        targets = suggest_lora_targets(model, text_only=True)
        entry["lora_target_modules"] = targets
        try:
            peft_cfg = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=targets,
            )
            model = get_peft_model(model, peft_cfg)
            entry["peft_qlora_possible"] = True
        except Exception as e:  # noqa: BLE001
            entry["peft_qlora_possible"] = False
            entry["errors"].append(f"peft: {type(e).__name__}: {e}")

        if args.generate:
            prompt = "Hello. Explain what photosynthesis is in one sentence."
            messages = [{"role": "user", "content": prompt}]
            try:
                text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:  # noqa: BLE001
                text = prompt
            inputs = tok(text, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            decoded = tok.decode(out[0], skip_special_tokens=True)
            entry["generate_ok"] = len(decoded) > 0
            entry["generate_sample"] = decoded[-400:]

        entry["status"] = "ok" if not entry["errors"] else "error"
    except Exception as e:  # noqa: BLE001
        entry["errors"].append(f"load: {type(e).__name__}: {e}")
        entry["status"] = "error"

    _write_merge(args.out, entry)
    print(json.dumps(entry, indent=2))
    if entry["status"] == "error":
        raise SystemExit(1)


def _write_merge(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"models": {}, "updated": None}
    if path.exists():
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            blob = {"models": {}}
    blob.setdefault("models", {})
    blob["models"][entry["alias"]] = entry
    from datetime import datetime, timezone

    blob["updated"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
