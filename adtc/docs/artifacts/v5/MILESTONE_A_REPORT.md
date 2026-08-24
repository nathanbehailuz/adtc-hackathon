# ADTC v5 — Milestone A stop report

**Date:** 2026-08-23  
**Status:** Infrastructure + data + HF config preflight complete. **CPT/SFT training not launched** (awaiting your approval after GGUF preflight jobs finish).

## 1. Files created / modified

### New
| Path | Role |
|------|------|
| `training/configs/v5_models.yaml` | Model manifesto (aliases + gates) |
| `training/model_loader.py` | CausalLM / Qwen3.5-safe loader; text-only LoRA targets |
| `training/inspect_lora_targets.py` | Dump LoRA targets |
| `training/preflight_v5_model.py` | HF config / 4bit / PEFT / generate preflight |
| `data/mix_cpt_v5.py` | CPT mix builder |
| `data/mix_sft_v5.py` | SFT mix builder (simonbutt-capped, cross-lingual) |
| `data/build_xling_sft_v5.py` | `am_en` / `en_am` tutoring pairs |
| `data/train/cpt_mix_v5.jsonl` | **21500** CPT docs |
| `data/train/sft_mix_v5.jsonl` | **8086** SFT rows |
| `data/train/xling_sft_v5.jsonl` | **1600** cross-lingual rows |
| `training/configs/cpt_qwen35_*_v5.yaml`, `qlora_qwen35_*_v5.yaml`, `cpt_qwen3_afrique_v5.yaml`, `qlora_qwen3_afrique_v5.yaml`, `qlora_qwen35_extcm_m0sft_v5.yaml` | Train configs |
| `hpc/02g_prepare_mix_v5.sbatch` | Mix prep |
| `hpc/03d_train_cpt_v5.sbatch`, `03e_merge_cpt_v5.sbatch` | CPT + merge |
| `hpc/04f_train_sft_v5.sbatch`, `05e_merge_lora_v5.sbatch` | SFT + merge |
| `hpc/08e_preflight_hf_v5.sbatch`, `08f_preflight_gguf_v5.sbatch`, `08g_convert_gguf_v5.sbatch` | Preflight / GGUF |
| `hpc/12g_eval_v5.sbatch`, `12h_eval_v5_hf.sbatch` | Eval |
| `hpc/submit_v5_preflight.sh` | Milestone A submitter |
| `hpc/submit_v5_chain.sh` | Milestone B submitter (**do not run yet**) |
| `docs/artifacts/v5/*` | Audits, counts, LoRA targets, model_preflight.json |

### Modified (backward-compatible)
| Path | Change |
|------|--------|
| `training/train_cpt_qlora.py` | Use `model_loader`; optional `model_class` |
| `training/train_sft_qlora.py` | Same |
| `training/merge_lora.py` | `--model-class` + `model_loader` |
| `training/download_base_models.py` | Added v5 AfriqueQwen aliases |

**Untouched:** all v0–v4 mixes, Gemma runs, GGUFs, frozen eval.

## 2. Architecture findings (HF config preflight)

| Alias | HF ID | Class | Vision? | Vocab | Chat template | Status |
|-------|-------|-------|---------|------:|:-------------:|--------|
| `qwen35_extcm` | `McGill-NLP/AfriqueQwen3.5-4B-ExtendedCM` | `Qwen3_5ForConditionalGeneration` | **yes** | 248044 | yes | config_ok |
| `qwen35_afrique` | `McGill-NLP/AfriqueQwen3.5-4B` | `Qwen3_5ForConditionalGeneration` | **yes** | 248044 | yes | config_ok |
| `qwen3_afrique` | `McGill-NLP/AfriqueQwen-4B` | `Qwen3ForCausalLM` | no | (text) | yes | config_ok |

**LoRA targets (text-only assumption):** `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`

**Hard risk:** Qwen3.5 Afrique checkpoints are multimodal. Competition packaging needs **one text GGUF** without a mandatory mmproj. GGUF convert smoke jobs are queued to prove this; if they fail, fail-fast to `qwen3_afrique`.

## 3. GGUF preflight

Slurm jobs submitted for all three aliases via `08f_preflight_gguf_v5.sbatch` (+ GPU HF 4bit via `08e`).  
**RSS / tok/s / convert OK not yet filled** — check:

```bash
squeue -u $USER
ls -la adtc/docs/artifacts/v5/gguf_preflight_*.json
```

Gate: peak RSS ≤ 7 GB on Q4; text gen without mmproj.

## 4. Mix counts

### CPT `cpt_mix_v5.jsonl` — n=21500 (target 25000; pool-limited)

| source | n |
|--------|--:|
| fineweb2_amh | 7993 |
| afrinllb | 5172 |
| en_stem_sft_v4 | 4000 |
| wikipedia_amharic | 3257 |
| am_stem_nllb_filtered | 1078 |

### SFT `sft_mix_v5.jsonl` — n=8086

| direction | n | | behavior | n |
|-----------|--:|-|----------|--:|
| am_am | 3524 | | explain | 3732 |
| en_am | 2142 | | solve | 2204 |
| en_en | 1620 | | first_error | 850 |
| am_en | 800 | | hint | 842 |
| | | | instruct | 450 |
| | | | code_switch | 8 |

**simonbutt_frac = 0.0** (cap respected). Cross-lingual hundreds present (`am_en` 800, `en_am` 2142).

## 5. Exact commands for Milestone B (not executed)

```bash
cd /scratch/nz2212/adtc-hackathon/adtc/hpc

# After GGUF preflight JSON shows convert+smoke OK for a candidate:
V5_ALIAS=qwen35_extcm bash submit_v5_chain.sh
# and/or
V5_ALIAS=qwen35_afrique bash submit_v5_chain.sh

# If Qwen3.5 GGUF fails text-only packaging:
V5_ALIAS=qwen3_afrique bash submit_v5_chain.sh

# Optional ablation (M0→SFT, skip CPT) after editing chain / using:
#   configs/qlora_qwen35_extcm_m0sft_v5.yaml
```

Manual single-stage example:

```bash
export V5_ALIAS=qwen35_extcm
sbatch --export=ALL,V5_ALIAS 03d_train_cpt_v5.sbatch
sbatch --export=ALL,V5_ALIAS --dependency=afterok:<cpt_job> 03e_merge_cpt_v5.sbatch
sbatch --export=ALL,V5_ALIAS --dependency=afterok:<merge_cpt> 04f_train_sft_v5.sbatch
sbatch --export=ALL,V5_ALIAS --dependency=afterok:<sft> 05e_merge_lora_v5.sbatch
sbatch --export=ALL,V5_ALIAS --dependency=afterok:<merge_sft> 08g_convert_gguf_v5.sbatch
sbatch --export=ALL,V5_ALIAS,V5_QUANT=Q4_K_M --dependency=afterok:<gguf> 12g_eval_v5.sbatch
sbatch --export=ALL,V5_ALIAS,V5_STAGE=m2 --dependency=afterok:<merge_sft> 12h_eval_v5_hf.sbatch
```

## 6. Decision needed from you

1. Wait for GGUF preflight JSON (convert + Q4 smoke + whether mmproj required).  
2. Pick primary train alias (`qwen35_extcm` preferred if text GGUF works).  
3. Approve `submit_v5_chain.sh` (and optional M0→SFT ablation).  

**Do not start long CPT/SFT until (1)+(2)+(3).**
