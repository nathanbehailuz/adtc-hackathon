# Datasets catalog — EN + Amharic STEM tutor

**On-disk status (what was downloaded, splits, row counts):** [`DATASETS_REPORT.md`](./DATASETS_REPORT.md).  
**Measured eval / profiler / leaderboard:** [`RESULTS_REPORT.md`](./RESULTS_REPORT.md).

**Rule:** frozen eval never enters training. Dedup train against `adtc/data/eval/` before every mix.

Language codes: BCP-47 `am` / Iroko HF configs often `amh`; English `en` / `eng`. FineWeb Amharic: `amh_Ethi`.

---

## Eval (freeze first) — DO NOT TRAIN

| Suite | HF id / path | Config / split | Use |
|-------|--------------|----------------|-----|
| AfriMGSM (Amharic) | [`masakhane/afrimgsm`](https://huggingface.co/datasets/masakhane/afrimgsm) | `amh` test | Amharic math |
| AfriMGSM (English) | same | `eng` test | EN math control / translate-test |
| AfriMMLU | [`masakhane/afrimmlu`](https://huggingface.co/datasets/masakhane/afrimmlu) | `amh` test | Knowledge / STEM |
| AfriXNLI | [`masakhane/afrixnli`](https://huggingface.co/datasets/masakhane/afrixnli) | `amh` test | Language understanding |
| Custom tutoring | `adtc/data/eval/custom_tutoring_v0.jsonl` | authored | Product behaviors |
| EN STEM holdout | `adtc/data/eval/en_stem_holdout_v0.jsonl` | GSM8K **test** | Forgetting check |
| Fertility parallels | `adtc/data/eval/fertility_parallel_v0.jsonl` | authored EN‖am | Tokenizer diagnostics |

Also keep out of training: Belebele, Global-MMLU, Global-MGSM (if used later).

**Freeze:** [`../data/eval/FREEZE.md`](../data/eval/FREEZE.md). Language: [`LANGUAGE.md`](./LANGUAGE.md).

---

## First-experiment recipe

```
CPT (only if Phase 4 diagnostics require it):
  FineWeb2-amh_Ethi-100M + Amharic News + AddisAI Wikipedia + AfriNLLB
    →
SFT (default first adaptation):
  Walia + FineTome + AfriqueLLM Amharic GSM8K
  + EN STEM (GSM8K train / SciQ) + verified MT STEM
  (+ optional R1 / Dolly / TACO after dedup)
```

Scripts: `data/download_train_sources.py`, `normalize_sft_sources.py`, `normalize_cpt_sources.py`, `mix_sft.py`.

**Run logs:** each of those scripts (plus model download / SFT / CPT train) writes OK/FAIL under `adtc/logs/<stage>/`. See [`RUNLOGS.md`](./RUNLOGS.md). Mid-run Ctrl+C still flushes a partial summary + `download_manifest_v0.json`.

---

## CPT pools (prepare now; run only if needed)

| Role | HF id | Notes |
|------|-------|--------|
| Native Amharic (primary) | [`MultilingualUnigramLM/FineWeb2-amh_Ethi-100M`](https://huggingface.co/datasets/MultilingualUnigramLM/FineWeb2-amh_Ethi-100M) | ~100M tokens; start here |
| Native Amharic (scale-up) | [`HuggingFaceFW/fineweb-2`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) `amh_Ethi` | **Not downloaded** in first pass |
| Native news | [`dagn/expanded-amharic-news-dataset`](https://huggingface.co/datasets/dagn/expanded-amharic-news-dataset) | Fluency / discourse |
| Educational / encyclopedic | [`addisai/wikipedia-amharic`](https://huggingface.co/datasets/addisai/wikipedia-amharic) | Prefer STEM filter later |
| Parallel EN↔AM | [`AfriNLP/AfriNLLB-train`](https://huggingface.co/datasets/AfriNLP/AfriNLLB-train) | CPT alignment **or** transform to translate SFT |
| Optional (filter hard) | [`a3xrfgb/amharic-sentences-corpus`](https://huggingface.co/datasets/a3xrfgb/amharic-sentences-corpus) | Social-media origin; secondary |

Suggested CPT mix (when triggered): native ~35% / translated-edu+parallel ~25% / EN STEM ~20% / EN replay ~10% / code ~10%.

---

## SFT pools

| Role | HF id | Notes |
|------|-------|--------|
| General Amharic instruct | [`EthioNLP/Amharic_Instruction_dataset`](https://huggingface.co/datasets/EthioNLP/Amharic_Instruction_dataset) (Walia) | Core |
| Conversational | [`addisai/FineTome-single-turn-dedup-amharic`](https://huggingface.co/datasets/addisai/FineTome-single-turn-dedup-amharic) | ~83k |
| STEM / math Amharic | [`peterlu02/afriquellm-coldstart-gsm8k-11lang`](https://huggingface.co/datasets/peterlu02/afriquellm-coldstart-gsm8k-11lang) | Filter Amharic |
| Reasoning aug | [`lightblue/reasoning-multilingual-R1-Llama-70B-train`](https://huggingface.co/datasets/lightblue/reasoning-multilingual-R1-Llama-70B-train) | Filter + quality gate |
| EN STEM tutoring | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) **train** | `build_en_stem_sft.py` |
| EN science QA | [`allenai/sciq`](https://huggingface.co/datasets/allenai/sciq) train | same builder |
| Own translated STEM | MT via `build_translate_am.py` | Review before heavy use |
| Supplementary | [`iocuydi/amharic-dolly-15k`](https://huggingface.co/datasets/iocuydi/amharic-dolly-15k), [`CRLannister/Amharic`](https://huggingface.co/datasets/CRLannister/Amharic) | After Walia/FineTome dedup |

**Train mixes on disk (16 Aug 2026):** `data/train/sft_mix_v0.jsonl` (stub MT); `data/train/sft_mix_v1.jsonl` (NLLB-200 real MT — see `docs/artifacts/sft_mix_v1_direction_counts.json`). **AfriqueLLM GSM8K** Hub snapshot still missing from downloads — known gap (not in mix_v1).

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
`behavior`: `solve` | `explain` | `hint` | `first_error` | `simplify` | `related_exercise` | `code_switch` | `instruct` | `translate`

---

## Excluded / cautious

| Dataset | Why |
|---------|-----|
| `YoseAli/amharic-llm-training-data` | Mixed provenance; possible benchmark contamination — **do not use wholesale** |
| All `adtc/data/eval/*` | Frozen evaluation |
| Full FineWeb2 (this pass) | Too large; use 100M slice first |

---

## Layout

```
adtc/data/eval/              # frozen holdouts (commit)
adtc/data/raw/hf/            # HF cache (gitignored)
adtc/data/raw/download_manifest_v0.json   # latest download status (updated per source)
adtc/data/train/sources/     # normalized SFT JSONL (gitignored if large)
adtc/data/train/cpt/         # normalized CPT text JSONL (gitignored)
adtc/data/train/sft_mix_*.jsonl
adtc/logs/<stage>/           # per-run OK/FAIL logs (gitignored) — see RUNLOGS.md
```

Download: `python data/download_train_sources.py --profile first_experiment`

After any interrupt, inspect:

```bash
cat data/raw/download_manifest_v0.json
cat logs/download_train/latest.summary.json
```
