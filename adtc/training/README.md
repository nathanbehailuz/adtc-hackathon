# Training (v6 English-only)

QLoRA with **Transformers + PEFT + TRL** on `Qwen/Qwen3-1.7B`.  
Training uses **Hugging Face** checkpoints. ADTC submission needs a **GGUF** after merge + convert.

`QLoRA 4-bit during training ≠ GGUF Q4 at deployment.`

**Run logs:** every stage writes OK/FAIL under `adtc/logs/<stage>/`. See [`../docs/RUNLOGS.md`](../docs/RUNLOGS.md).

## Jubail (Slurm)

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

On Jubail the preferred env is the scratch conda prefix from `hpc/setup_env.sbatch` (`adtc/training/.conda-env`).

## Data (before train)

From `adtc/`:

```bash
python data/mix_sft_v6.py \
  --sciq-limit 3000 \
  --out data/train/sft_mix_v6.jsonl \
  --counts-out docs/artifacts/v6/sft_mix_v6_counts.json \
  --report-out docs/artifacts/v6/sft_mix_v6_report.md
```

Mix: full GSM8K train (7473 tutoring rows) + SciQ (3000). Dedup vs frozen EN eval sets.

## Base model

```bash
cd adtc
python training/download_base_models.py --only qwen3_1_7b
```

## Train (SFT)

```bash
cd adtc/training
python train_sft_qlora.py --config configs/qlora_qwen3_1_7b_v6.yaml
```

Adapter: `runs/qwen3_1_7b_qlora_v6/adapter/`. Log: `logs/train_sft/`.

## Merge → GGUF

```bash
python merge_lora.py \
  --base Qwen/Qwen3-1.7B \
  --adapter runs/qwen3_1_7b_qlora_v6/adapter \
  --out runs/qwen3_1_7b_merged_v6
```

On Jubail, `convert_gguf.sbatch` converts merged HF → GGUF quants.  
Deploy pick: `artifacts/gguf/adapted/qwen3_1_7b_merged_v6-Q5_K_M.gguf`.

## Eval

```bash
python eval/run_hf_eval.py --model runs/qwen3_1_7b_merged_v6
python eval/run_gguf_eval.py --gguf artifacts/gguf/adapted/qwen3_1_7b_merged_v6-Q5_K_M.gguf
```

Results: [`../docs/artifacts/v6/`](../docs/artifacts/v6/).
