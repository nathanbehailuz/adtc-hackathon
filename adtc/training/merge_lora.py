#!/usr/bin/env python3
"""Merge LoRA adapters into the base HF model for GGUF conversion.

Example:
  python merge_lora.py \\
    --base Qwen/Qwen3-1.7B \\
    --adapter runs/qwen3_1_7b_qlora_v0/adapter \\
    --out runs/qwen3_1_7b_merged_v0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="HF base model id or path")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"loading base {args.base} …")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )
    print(f"loading adapter {args.adapter} …")
    model = PeftModel.from_pretrained(base, str(args.adapter))
    print("merging …")
    model = model.merge_and_unload()

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out), safe_serialization=True)
    tokenizer.save_pretrained(str(args.out))
    print(f"merged model saved to {args.out}")
    print("Next: convert to GGUF with llama.cpp convert script, then quantize Q8→Q4.")


if __name__ == "__main__":
    main()
