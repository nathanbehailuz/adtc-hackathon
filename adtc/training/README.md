# Training (cloud GPU)

QLoRA with **Transformers + PEFT + TRL**.  
Training uses **Hugging Face** checkpoints. ADTC submission needs a **GGUF** after merge + convert.

`QLoRA 4-bit during training ≠ GGUF Q4 at deployment.`

## Setup (on the GPU machine)

```bash
cd adtc/training
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data (before train)

From repo root / `adtc/`:

```bash
# Frozen eval (once)
python eval/prepare_iroko_am.py --limit 50   # drop --limit for full freeze
python eval/prepare_en_stem.py --limit 100

# Train pools
python data/build_en_stem_sft.py --limit 500
python data/build_translate_am.py --in data/train/en_stem_sft_v0.jsonl --backend stub
# Later: --backend nllb  or  --backend file --map translations.json

python data/mix_sft.py \
  --en-stem data/train/en_stem_sft_v0.jsonl \
  --am-stem data/train/am_stem_sft_v0.jsonl \
  --eval data/eval/custom_tutoring_v0.jsonl data/eval/en_stem_holdout_v0.jsonl \
  --out data/train/sft_mix_v0.jsonl \
  --total 2000
```

Replace stub MT with real Amharic translations before serious runs. Nathan reviews Amharic samples.

## Train

```bash
cd training
python train_sft_qlora.py --config configs/qlora_qwen3_1_7b.yaml
# after Phase 2, maybe also:
python train_sft_qlora.py --config configs/qlora_qwen3_4b.yaml
```

Adapters land in `runs/*/adapter/` (gitignored).

## Merge → GGUF

```bash
python merge_lora.py \
  --base Qwen/Qwen3-1.7B \
  --adapter runs/qwen3_1_7b_qlora_v0/adapter \
  --out runs/qwen3_1_7b_merged_v0
```

Then on a machine with llama.cpp:

1. Convert HF folder → high-precision GGUF  
2. Quantize to Q8 / Q6 / Q5 / Q4  
3. Profile with pinned `adtc-profiler`  
4. Host public GGUF and point `download_model.sh` at it  

## Fertility (optional, after tokenizer available)

```bash
python eval/fertility.py --tokenizer Qwen/Qwen3-1.7B
```
