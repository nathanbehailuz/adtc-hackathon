#!/usr/bin/env python3
"""Optional continued pretraining (CPT) with TRL — run only if Phase 4 diagnostics require it.

Expects CPT JSONL with a ``text`` field (from ``data/normalize_cpt_sources.py``).

Example (cloud GPU, after SFT diagnostics say CPT is needed)::

  python train_cpt_qlora.py --config configs/cpt_qwen3_1_7b.yaml

Until a CPT config exists, this script still records a structured run log when invoked
so HPC failures are inspectable under ``logs/train_cpt/``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.run_log import RunLogger  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None, help="Override data_path from config")
    args = parser.parse_args()

    if not args.config.exists():
        log = RunLogger("train_cpt", meta={"config": str(args.config)})
        log.item_error(
            "config",
            f"missing config {args.config} — create configs/cpt_*.yaml only if Gate 4 triggers CPT",
        )
        log.finish(status="error")
        raise SystemExit(1)

    cfg = load_config(args.config)
    here = Path(__file__).resolve().parent
    data_path = Path(args.data) if args.data else (here / cfg["data_path"]).resolve()
    output_dir = (here / cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_id = cfg["model_name_or_path"]

    log = RunLogger(
        "train_cpt",
        meta={
            "config": str(args.config),
            "model_name_or_path": model_id,
            "data_path": str(data_path),
            "output_dir": str(output_dir),
        },
    )

    try:
        log.item_start("load_model", hf_id=model_id)
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
        log.item_ok("load_model", hf_id=model_id)

        peft_config = LoraConfig(
            r=int(cfg.get("lora_r", 8)),
            lora_alpha=int(cfg.get("lora_alpha", 16)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(cfg.get("lora_target_modules", ["q_proj", "v_proj"])),
        )

        log.item_start("load_data", path=str(data_path))
        ds = load_dataset("json", data_files=str(data_path), split="train")
        if "text" not in ds.column_names:
            raise ValueError("CPT data must have a 'text' field")
        log.item_ok("load_data", path=str(data_path), n_rows=len(ds))

        # Transformers removed warmup_ratio; float in [0,1) for warmup_steps = ratio of total steps.
        warmup = float(cfg.get("warmup_steps", cfg.get("warmup_ratio", 0.03)))
        sft_args = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=float(cfg.get("num_train_epochs", 1)),
            per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
            learning_rate=float(cfg.get("learning_rate", 1e-4)),
            lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
            warmup_steps=warmup,
            logging_steps=int(cfg.get("logging_steps", 10)),
            save_steps=int(cfg.get("save_steps", 200)),
            bf16=bool(cfg.get("bf16", True)),
            gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
            seed=int(cfg.get("seed", 42)),
            report_to=cfg.get("report_to", "none"),
            max_length=int(cfg.get("max_seq_length", 2048)),
            dataset_text_field="text",
            packing=bool(cfg.get("packing", True)),
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

        log.item_start("train", output_dir=str(output_dir))
        train_result = trainer.train()
        trainer.save_model(str(output_dir / "adapter"))
        tokenizer.save_pretrained(str(output_dir / "adapter"))

        metrics = dict(train_result.metrics)
        metrics["model_name_or_path"] = model_id
        metrics["data_path"] = str(data_path)
        metrics["run_id"] = log.run_id
        metrics["note"] = "CPT complete. Usually follow with SFT on the CPT adapter/base."
        (output_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        log.item_ok("train", metrics=metrics, adapter=str(output_dir / "adapter"))
        print(json.dumps(metrics, indent=2))
    except KeyboardInterrupt:
        log.warn("KeyboardInterrupt during CPT")
        log.finish(status="interrupted", message="interrupted by user")
        raise SystemExit(130) from None
    except Exception as e:  # noqa: BLE001
        log.item_error("train_cpt", e)
        log.finish(status="error")
        raise

    summary = log.finish()
    print(f"run log -> {log.log_path}")
    print(f"summary -> {summary}")


if __name__ == "__main__":
    main()
