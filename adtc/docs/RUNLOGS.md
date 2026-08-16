# Run logs (HPC / interrupted downloads)

Every pipeline stage writes a **timestamped run log** under `adtc/logs/<stage>/` so a Ctrl+C or node kill still leaves OK/FAIL status.

## Layout

```
adtc/logs/
  download_train/     # data/download_train_sources.py
  normalize_cpt/      # data/normalize_cpt_sources.py
  normalize_sft/      # data/normalize_sft_sources.py
  mix_sft/            # data/mix_sft.py
  download_models/    # training/download_base_models.py
  train_sft/          # training/train_sft_qlora.py
  train_cpt/          # training/train_cpt_qlora.py (only if Gate 4 triggers CPT)
  merge_lora/         # training/merge_lora.py
```

Per run (example `20260816T104500Z_a1b2c3d4`):

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
# last download attempt
cat logs/download_train/latest.summary.json | python -m json.tool | head -80

# human log
ls -t logs/download_train/*.log | head -1 | xargs cat
```

`counts.ok` / `counts.error` / `counts.skipped` tell you what finished. Interrupted items show `status: interrupted` or `error` with reason.

## Also written (stage-specific)

| Stage | Extra artifact |
|-------|----------------|
| download_train | `data/raw/download_manifest_v0.json` (rewritten after **each** source) |
| normalize_cpt | `data/train/cpt/cpt_normalize_summary_v0.json` |
| download_models | `training/model_download_manifest_v0.json` |
| train_sft / train_cpt | `training/runs/.../train_metrics.json` |

## HPC tomorrow order (with logs)

```bash
cd adtc
# 1) finish / resume train corpora (partial OK is fine; re-run or --only …)
python data/download_train_sources.py --profile first_experiment

# 2) normalize
python data/normalize_cpt_sources.py
python data/normalize_sft_sources.py
python data/build_en_stem_sft.py --limit 2000
python data/mix_sft.py --sft data/train/sources/*.jsonl \
  --en-stem data/train/en_stem_sft_v0.jsonl \
  --eval data/eval/custom_tutoring_v0.jsonl data/eval/en_stem_holdout_v0.jsonl

# 3) base models (Phase 2)
python training/download_base_models.py

# 4) SFT (after mix exists)
cd training && python train_sft_qlora.py --config configs/qlora_qwen3_1_7b.yaml

# 5) CPT only if diagnostics require it
# python train_cpt_qlora.py --config configs/cpt_qwen3_1_7b.yaml
```

`logs/` is gitignored — copy `*.summary.json` into `docs/artifacts/` only when you want a committed record of a run.
