# ADTC v6 — English-only Qwen3-1.7B (profiler track)

**Date:** 2026-08-24  
**Status:** **Complete.** Train → merge → GGUF → eval → profiler ran on Jubail. Gate 5 winner: **Q5_K_M**.

## Decisions

| Item | Choice |
|------|--------|
| Base | `Qwen/Qwen3-1.7B` |
| Language | English-only SFT (GSM8K + SciQ) |
| Shape | SFT → merge → GGUF Q4/Q5/Q6 → HF+GGUF eval → profiler |
| Deploy | `qwen3_1_7b_merged_v6-Q5_K_M.gguf` |
| Thinking | HF: `enable_thinking=False`; GGUF: `/no_think` prefix |

## Mix

| File | n |
|------|--:|
| `data/train/sft_mix_v6.jsonl` | **10473** |

| source | n |
|--------|--:|
| gsm8k_train_v6 | 7473 |
| sciq_train_v6 | 3000 |

Dedup vs `en_stem_holdout_v0` + `afrimgsm_eng_test_v0`: **0 dropped**.

## Results (HF full frozen)

| Suite | acc |
|-------|----:|
| AfriMGSM EN | 0.392 |
| EN STEM holdout | 0.370 |
| Custom tutoring | 0.980 |

Profiler (Q5_K_M): TPS 2.46, peak RSS 1402 MB, composite 21.01 — see `phase5_gate5_winner_v6.json`.

## Files

| Path | Role |
|------|------|
| `data/mix_sft_v6.py` | Mix builder |
| `hpc/submit_chain.sh` | Full Slurm chain |
| `hpc/prepare_mix.sbatch` | Mix prep |
| `hpc/train_sft.sbatch` | QLoRA SFT |
| `hpc/merge_lora.sbatch` | Merge |
| `hpc/convert_gguf.sbatch` | GGUF quants |
| `hpc/eval_hf.sbatch` / `eval_gguf.sbatch` | Frozen eval |
| `hpc/profile_gguf.sbatch` | Gate 5 pick |
| `training/configs/qlora_qwen3_1_7b_v6.yaml` | QLoRA |
| `eval/try_prompt.py` model **1** | `qwen3_1_7b_merged_v6-Q5_K_M.gguf` |
| `adtc-2026-submission-template/` | Staged submission (metadata + model.gguf) |

## Launch (reproduce)

```bash
cd adtc/hpc
bash submit_chain.sh
```

Artifacts: `docs/artifacts/v6/`.
