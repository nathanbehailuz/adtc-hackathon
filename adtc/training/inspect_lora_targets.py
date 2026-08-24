#!/usr/bin/env python3
"""Inspect LoRA target modules for a HF checkpoint (text-only by default).

Example:
  python inspect_lora_targets.py --model McGill-NLP/AfriqueQwen3.5-4B-ExtendedCM \\
    --out ../docs/artifacts/v5/lora_targets_qwen35_extcm.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_loader import (  # noqa: E402
    inspect_config,
    is_vision_module,
    list_linear_module_names,
    load_pretrained_model,
    suggest_lora_targets,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF id or local path")
    ap.add_argument("--model-class", default="auto", choices=("auto", "causal", "qwen3_5"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--load-weights",
        action="store_true",
        help="Load weights (slow/memory heavy); default is config-only",
    )
    args = ap.parse_args()

    meta = inspect_config(args.model)
    result: dict = {"config": meta, "lora_target_modules": [], "linear_modules_sample": []}

    if args.load_weights:
        import torch

        model = load_pretrained_model(
            args.model,
            model_class=args.model_class,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        all_lin = list_linear_module_names(model)
        vision = [n for n in all_lin if is_vision_module(n)]
        text = [n for n in all_lin if not is_vision_module(n)]
        targets = suggest_lora_targets(model, text_only=True)
        result.update(
            {
                "n_linear": len(all_lin),
                "n_vision_linear": len(vision),
                "n_text_linear": len(text),
                "linear_modules_sample": text[:40],
                "vision_modules_sample": vision[:20],
                "lora_target_modules": targets,
            }
        )
        del model
    else:
        result["lora_target_modules"] = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        result["note"] = "config-only; standard text proj names assumed"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
