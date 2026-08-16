# Training (cloud GPU)

QLoRA with **Transformers + PEFT + TRL**.  
Training uses **Hugging Face** checkpoints. ADTC submission needs a **GGUF** after merge + convert.

`QLoRA 4-bit during training ≠ GGUF Q4 at deployment.`

**Run logs:** every stage below writes OK/FAIL under `adtc/logs/<stage>/`. See [`../docs/RUNLOGS.md`](../docs/RUNLOGS.md).

## Jubail (Slurm)

On NYUAD Jubail, do **not** train or download on login nodes. From `$SCRATCH`:

```bash
cd /scratch/nz2212/adtc-hackathon/adtc/hpc
bash submit_chain.sh    # setup → data → models → SFT 1.7B → merge
```

Details, partitions (A100), and monitor commands: [`../hpc/README.md`](../hpc/README.md).

## Setup (on the GPU machine)

Local / interactive (non-Slurm):

```bash
cd adtc/training
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# huggingface_hub is pulled in by transformers; needed for download_base_models.py
```

On Jubail the preferred env is the scratch conda prefix created by `hpc/setup_env.sbatch`
(`adtc/training/.conda-env`).

## Data (before train)

From repo root / `adtc/`:

```bash
# Train corpora (logged → logs/download_train/)
python data/download_train_sources.py --profile first_experiment
# resume subset after interrupt / gated failures:
# python data/download_train_sources.py --only wikipedia_amharic walia finetome_am

python data/normalize_cpt_sources.py      # logs/normalize_cpt/
python data/normalize_sft_sources.py      # logs/normalize_sft/

python data/build_en_stem_sft.py --limit 500
python data/build_translate_am.py --in data/train/en_stem_sft_v0.jsonl --backend stub
# Later: --backend nllb  or  --backend file --map translations.json

python data/mix_sft.py \
  --en-stem data/train/en_stem_sft_v0.jsonl \
  --sft data/train/sources/walia_sft_v0.jsonl data/train/sources/finetome_am_sft_v0.jsonl \
  --eval data/eval/custom_tutoring_v0.jsonl data/eval/en_stem_holdout_v0.jsonl \
  --out data/train/sft_mix_v0.jsonl \
  --total 2000
```

Replace stub MT with real Amharic translations before serious runs. Nathan reviews Amharic samples.

## Base models (Phase 2)

```bash
cd adtc
python training/download_base_models.py
# or: python training/download_base_models.py --only qwen3_1_7b
# inspect: cat logs/download_models/latest.summary.json
```

## Train (SFT)

```bash
cd training
python train_sft_qlora.py --config configs/qlora_qwen3_1_7b.yaml
# after Phase 2, maybe also:
python train_sft_qlora.py --config configs/qlora_qwen3_4b.yaml
```

Adapters land in `runs/*/adapter/` (gitignored). Log: `logs/train_sft/`.

### CPT (conditional only)

If Gate 4 diagnostics say direct-Amharic is still weak:

```bash
# add configs/cpt_qwen3_1_7b.yaml pointing at a CPT text mix, then:
python train_cpt_qlora.py --config configs/cpt_qwen3_1_7b.yaml
```

Log: `logs/train_cpt/`. Prefer SFT-first; do not burn GPU on CPT by default.

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
