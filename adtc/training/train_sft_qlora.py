#!/usr/bin/env python3
"""QLoRA supervised fine-tuning with TRL + PEFT.

Example (cloud GPU):
  pip install -r requirements.txt
  python train_sft_qlora.py --config configs/qlora_qwen3_1_7b.yaml

QLoRA 4-bit ≠ final GGUF Q4. After training, merge adapters then convert to GGUF.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None, help="Override data_path from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    here = Path(__file__).resolve().parent
    data_path = Path(args.data) if args.data else (here / cfg["data_path"]).resolve()
    output_dir = (here / cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    compute_dtype = getattr(torch, cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    bnb = BitsAndBytesConfig(
        load_in_4bit=bool(cfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
    )

    model_id = cfg["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.get("lora_target_modules", ["q_proj", "v_proj"])),
    )

    ds = load_dataset("json", data_files=str(data_path), split="train")

    def to_text(example: dict) -> dict:
        # Prefer chat template when available
        messages = example["messages"]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            parts = []
            for m in messages:
                parts.append(f"{m['role'].upper()}: {m['content']}")
            text = "\n".join(parts) + "\n"
        return {"text": text}

    ds = ds.map(to_text, remove_columns=[c for c in ds.column_names if c != "text"])

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(cfg.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        save_steps=int(cfg.get("save_steps", 200)),
        bf16=bool(cfg.get("bf16", True)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        seed=int(cfg.get("seed", 42)),
        report_to=cfg.get("report_to", "none"),
        max_length=int(cfg.get("max_seq_length", 2048)),
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))

    metrics = dict(train_result.metrics)
    metrics["model_name_or_path"] = model_id
    metrics["data_path"] = str(data_path)
    metrics["note"] = "QLoRA training complete. Merge adapters before GGUF conversion."
    (output_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"adapter saved to {output_dir / 'adapter'}")


if __name__ == "__main__":
    main()
