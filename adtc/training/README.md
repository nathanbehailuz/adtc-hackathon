# Training (v7 English-only)

QLoRA with **Transformers + PEFT + TRL** on `Qwen/Qwen3-1.7B`.  
Training uses **Hugging Face** checkpoints. ADTC submission needs a **GGUF** after merge + convert.

`QLoRA 4-bit during training ≠ GGUF Q4 at deployment.`

**Run logs:** every stage writes OK/FAIL under `adtc/logs/<stage>/`. See [`../docs/RUNLOGS.md`](../docs/RUNLOGS.md).

## Africa GPU Hub (preferred)

```bash
cd adtc/agh
source env.sh
bash setup_env.sh
bash download_models.sh
bash run_chain.sh   # inside tmux
```

Details: [`../agh/README.md`](../agh/README.md).

## Jubail (Slurm, legacy)

On NYUAD Jubail, do **not** train or download on login nodes:

```bash
cd /scratch/nz2212/adtc-hackathon/adtc/hpc
sbatch setup_env.sbatch
sbatch download_models.sbatch
bash submit_chain.sh
```

Details: [`../hpc/README.md`](../hpc/README.md).

## Setup (local / interactive)

```bash
cd adtc/training
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On AGH, prefer `agh/setup_env.sh` → `adtc/training/.conda-env`.

## Data (before train)

From `adtc/`:

```bash
python data/build_authored_tutoring_v7.py
python data/mix_sft_v7.py \
  --sciq-limit 3000 \
  --out data/train/sft_mix_v7.jsonl \
  --counts-out docs/artifacts/v7/sft_mix_v7_counts.json \
  --report-out docs/artifacts/v7/sft_mix_v7_report.md
```

Mix: cleaned GSM8K train + SciQ (~3k) + authored tutoring bank. Dedup vs frozen EN eval sets. Assistant targets must not contain `####` or `<<>>`.

## Base model

```bash
cd adtc
python training/download_base_models.py --only qwen3_1_7b
```

## Train (SFT)

```bash
cd adtc/training
python train_sft_qlora.py --config configs/qlora_qwen3_1_7b_v7.yaml
```

Adapter: `runs/qwen3_1_7b_qlora_v7/adapter/`. Log: `logs/train_sft/`.

## Merge → GGUF

```bash
python merge_lora.py \
  --base Qwen/Qwen3-1.7B \
  --adapter runs/qwen3_1_7b_qlora_v7/adapter \
  --out runs/qwen3_1_7b_merged_v7
```

On AGH, `agh/convert_gguf.sh` converts merged HF → GGUF quants.  
Candidate deploy: `artifacts/gguf/adapted/qwen3_1_7b_merged_v7-Q4_K_M.gguf` or `Q5_K_M` after profiler pick.

## Eval

```bash
python eval/run_hf_eval.py --model runs/qwen3_1_7b_merged_v7
python eval/run_gguf_eval.py --gguf artifacts/gguf/adapted/qwen3_1_7b_merged_v7-Q5_K_M.gguf
python eval/run_judge_smoke.py --gguf artifacts/gguf/adapted/qwen3_1_7b_merged_v7-Q5_K_M.gguf \
  --out docs/artifacts/v7/judge_smoke_Q5_K_M.json
```

Results: [`../docs/artifacts/v7/`](../docs/artifacts/v7/).
