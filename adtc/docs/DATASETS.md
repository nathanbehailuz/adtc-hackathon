# Datasets — v6 English-only STEM tutor

**Measured eval / profiler:** [`RESULTS_REPORT.md`](./RESULTS_REPORT.md).  
**Mix report:** [`artifacts/v6/sft_mix_v6_report.md`](./artifacts/v6/sft_mix_v6_report.md).

**Rule:** frozen eval never enters training. Dedup train against `adtc/data/eval/` before every mix (`eval/dedup_against_eval.py`).

---

## Train (v6)

Built by [`data/mix_sft_v6.py`](../data/mix_sft_v6.py) → `data/train/sft_mix_v6.jsonl`.

| Source | HF id | Rows (v6) | Behaviors |
|--------|-------|----------:|-----------|
| GSM8K train | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) `main/train` | 7473 | solve, explain, hint, first_error |
| SciQ train | [`allenai/sciq`](https://huggingface.co/datasets/allenai/sciq) train | 3000 | solve, explain |

**Total:** 10473 rows. Dedup vs `en_stem_holdout_v0` + `afrimgsm_eng_test_v0`: 0 dropped.

Row schema (JSONL):

```json
{
  "id": "en_en_solve_gsm8k_v6_00000",
  "direction": "en_en",
  "behavior": "solve",
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "source": "gsm8k_train_v6"
}
```

---

## Eval (frozen — DO NOT TRAIN)

Under `adtc/data/eval/` (see [`FREEZE.md`](../data/eval/FREEZE.md)).

| Suite | Path | Use (v6) |
|-------|------|----------|
| AfriMGSM EN | `afrimgsm_eng_test_v0.jsonl` | Primary EN math |
| EN STEM holdout | `en_stem_holdout_v0.jsonl` | GSM8K test holdout |
| Custom tutoring | `custom_tutoring_v0.jsonl` | Product behaviors |
| AfriMGSM AM | `afrimgsm_amh_test_v0.jsonl` | Secondary (not trained) |
| AfriMMLU AM | `afrimmlu_amh_test_v0.jsonl` | Secondary (not trained) |

Dedup at mix time uses EN holdout + AfriMGSM EN only.

---

## Base model cache

- HF weights: `data/raw/hf_home/models--Qwen--Qwen3-1.7B/`
- Dataset cache: `data/raw/hf/` (GSM8K, SciQ via `datasets`)

Download: `python training/download_base_models.py --only qwen3_1_7b`

---

## HPC prep

```bash
cd adtc/hpc
sbatch prepare_mix.sbatch   # builds sft_mix_v6.jsonl on compute node
```

No separate Amharic corpus download job — v6 mix pulls GSM8K/SciQ via `datasets` inside the mix script.
