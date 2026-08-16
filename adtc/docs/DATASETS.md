# Datasets catalog — EN + Amharic STEM tutor

**Rule:** frozen eval never enters training. Dedup train against `adtc/data/eval/` before every mix.

Language codes: BCP-47 `am` / Iroko HF configs often `amh`; English `en` / `eng`.

---

## Eval (freeze first)

| Suite | HF id / path | Config / split | Use |
|-------|--------------|----------------|-----|
| AfriMGSM (Amharic) | [`masakhane/afrimgsm`](https://huggingface.co/datasets/masakhane/afrimgsm) | `amh` test | Amharic math |
| AfriMGSM (English) | same | `eng` test | EN math control / translate-test |
| AfriMMLU | [`masakhane/afrimmlu`](https://huggingface.co/datasets/masakhane/afrimmlu) | Amharic config if present | Knowledge / STEM |
| AfriXNLI | [`masakhane/afrixnli`](https://huggingface.co/datasets/masakhane/afrixnli) | Amharic config if present | Language understanding |
| Custom tutoring | `adtc/data/eval/custom_tutoring_v0.jsonl` | authored | Product behaviors |
| EN STEM holdout | `adtc/data/eval/en_stem_holdout_v0.jsonl` | from GSM8K **test** only | Forgetting check |
| Fertility parallels | `adtc/data/eval/fertility_parallel_v0.jsonl` | authored EN‖am | Tokenizer diagnostics |

Scripts: `adtc/eval/prepare_iroko_am.py`, `prepare_en_stem.py`, `fertility.py`.

**License note:** Iroko / Masakhane sets are typically CC BY-SA — keep attribution in REPORT.md; do not train on test splits.

---

## Train pools

| Pool | Initial sources | Role | Builder |
|------|-----------------|------|---------|
| Native Amharic | Open Amharic text (e.g. Wikipedia `am`, curated open dumps listed when added) | grammar / register | manual / future CPT |
| EN STEM tutoring | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) **train** (+ tutoring templates) | reasoning + pedagogy | `adtc/data/build_en_stem_sft.py` |
| EN science QA | [`allenai/sciq`](https://huggingface.co/datasets/allenai/sciq) train (subsample) | science explain | same |
| Translated am STEM | MT of EN STEM pool → Amharic | STEM into Amharic | `adtc/data/build_translate_am.py` |
| EN replay | subsample of EN STEM SFT | reduce forgetting | `adtc/data/mix_sft.py` |
| Code / structured | small open math-word slice (optional) | compositional | mix ratios |

### SFT row schema (JSONL)

```json
{
  "id": "en_en_solve_0001",
  "direction": "en_en",
  "behavior": "solve",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "source": "gsm8k_train"
}
```

`direction`: `en_en` | `am_am` | `en_am` | `am_en`  
`behavior`: `solve` | `explain` | `hint` | `first_error` | `simplify` | `related_exercise` | `code_switch`

### Suggested mix (SFT starting point)

Aligned with methodology: native / translated STEM / EN STEM / replay / code ≈ **35 / 25 / 20 / 10 / 10** when CPT runs; for first bilingual SFT emphasize tutoring directions evenly across `en_en`, `am_am`, `en_am`, `am_en`.

---

## Layout

```
adtc/data/eval/   # frozen holdouts (commit small JSONL)
adtc/data/train/  # generated mixes (large files may be gitignored)
adtc/data/raw/    # HF downloads / caches (gitignored)
```

---

## Never

- AfriMGSM / AfriMMLU / AfriXNLI **test** in training
- Custom tutoring eval IDs in training (dedup by normalized text hash)
- Submitting teacher / MT models in the ADTC GGUF
