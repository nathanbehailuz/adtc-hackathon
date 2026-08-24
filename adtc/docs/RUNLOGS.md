# Run logs (HPC / interrupted jobs)

Every pipeline stage writes a **timestamped run log** under `adtc/logs/<stage>/` so a Ctrl+C or node kill still leaves OK/FAIL status.

## Layout (v6)

```
adtc/logs/
  download_models/    # training/download_base_models.py
  train_sft/          # training/train_sft_qlora.py
  merge_lora/         # training/merge_lora.py
  try_prompt/         # eval/try_prompt.py (Slurm try_prompt.sbatch)
```

Per run (example `20260824T104500Z_a1b2c3d4`):

| File | Purpose |
|------|---------|
| `*.log` | Human-readable `START` / `OK` / `FAIL` / `DONE` lines |
| `*.jsonl` | One JSON event per line (incremental; survives interrupt) |
| `*.summary.json` | Rollup with `counts` + per-item status |
| `latest.summary.json` | Copy of the most recent summary for that stage |

Shared helper: [`../lib/run_log.py`](../lib/run_log.py).

## Quick check after a run

```bash
cd adtc
cat logs/download_models/latest.summary.json | python -m json.tool | head -40
cat logs/train_sft/latest.summary.json | python -m json.tool | head -40
ls -t logs/train_sft/*.log | head -1 | xargs cat
```

## HPC / Jubail order (v6)

```bash
cd /scratch/nz2212/adtc-hackathon/adtc/hpc
sbatch setup_env.sbatch
sbatch download_models.sbatch
bash submit_chain.sh
# Slurm stdout: adtc/hpc/logs/; stage logs: adtc/logs/<stage>/
```

Equivalent plain Python (inside an allocated job — **not** on login):

```bash
cd adtc
python data/mix_sft_v6.py
python training/download_base_models.py --only qwen3_1_7b
cd training && python train_sft_qlora.py --config configs/qlora_qwen3_1_7b_v6.yaml
python merge_lora.py --base Qwen/Qwen3-1.7B \
  --adapter runs/qwen3_1_7b_qlora_v6/adapter \
  --out runs/qwen3_1_7b_merged_v6
```

`logs/` is gitignored — eval/profiler JSON under `docs/artifacts/v6/` is the committed record.
