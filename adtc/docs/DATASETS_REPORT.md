# Datasets report — what was downloaded, from where, train vs test

**As of:** 18 Aug 2026 (Jubail `$SCRATCH`, download run `f94ed8c6`).  
**Catalog / recipe:** [`DATASETS.md`](./DATASETS.md). Freeze rules: [`../data/eval/FREEZE.md`](../data/eval/FREEZE.md).  
**Manifest:** `adtc/data/raw/download_manifest_v0.json`. Logs: `adtc/logs/download_train/`.

This is a **status report of files on disk**, not the planning catalog. Counts are JSON Lines (JSONL) rows unless noted.

---

## Acronyms

Terms used in this report. Language codes are **not** interchangeable: BCP-47 `am` / `en` appear in our JSONL; Hugging Face (HF) / Iroko configs often use `amh` / `eng`.

| Acronym | Expansion | In this project |
|---------|-----------|-----------------|
| **ADTC** | this 2026 offline multilingual STEM-tutor hackathon | Repo / product name |
| **AM / am** | Amharic | Target language (Geʽez / Ethiopic script) |
| **amh** | Amharic (ISO 639-3) | HF / Iroko config name |
| **amh_Ethi** | Amharic in Ethiopic script | FineWeb-2 language id |
| **AfriMGSM** | African Multilingual Grade School Math | Frozen math eval (`masakhane/afrimgsm`) |
| **AfriMMLU** | African Massive Multitask Language Understanding | Frozen knowledge eval (`masakhane/afrimmlu`) |
| **AfriNLLB** | African No Language Left Behind (train set) | Parallel EN↔AM corpus (`AfriNLP/AfriNLLB-train`) |
| **AfriXNLI** | African Cross-lingual Natural Language Inference | Frozen NLI eval (`masakhane/afrixnli`) |
| **ARC** | AI2 Reasoning Challenge | `allenai/ai2_arc` in cache from lm-eval only |
| **BCP-47** | IETF Best Current Practice 47 (language tags) | `am`, `en`, `amh_Ethi` |
| **CPT** | continued pre-training | Next-token training on raw text; prepared, **not run** |
| **EN / en / eng** | English | Control language / HF `eng` config |
| **eval** | evaluation | Frozen holdouts under `data/eval/` — never train |
| **GGUF** | GPT-Generated Unified Format | llama.cpp weight file (eval scripts, not this mix) |
| **GSM8K** | Grade School Math 8K | OpenAI math word problems |
| **HF** | Hugging Face | Hub + `datasets` library; cache under `data/raw/hf/` |
| **Iroko / IrokoBench** | Masakhane African LLM eval suite | Source of AfriMGSM / AfriMMLU / AfriXNLI exports |
| **ISO 639-3** | three-letter language codes | `amh`, `eng` |
| **JSONL** | JSON Lines | One JSON object per newline |
| **lm-eval** | EleutherAI Language Model Evaluation Harness | May pull extra suites (e.g. ARC) into cache |
| **LoRA** | Low-Rank Adaptation | Parameter-efficient adapters |
| **MT** | machine translation | NLLB (or stub) EN→AM tutoring text |
| **NLI** | natural language inference | Entailment / contradiction / neutral (AfriXNLI) |
| **NLLB** | No Language Left Behind | Meta translation model `facebook/nllb-200-distilled-600M` |
| **QLoRA** | Quantized Low-Rank Adaptation | 4-bit base + LoRA SFT (`training/train_sft_qlora.py`) |
| **R1** | DeepSeek-R1-style reasoning traces | `lightblue/reasoning-multilingual-R1-Llama-70B-train` |
| **SciQ** | Science Questions (AllenAI) | Optional EN science SFT — **not downloaded** |
| **SFT** | supervised fine-tuning | Chat JSONL instruction tuning |
| **SHA-256** | 256-bit Secure Hash Algorithm | Checksums in `eval_manifest_v0.json` |
| **STEM** | science, technology, engineering, mathematics | Tutoring domain |
| **TACO** | Amharic instruction dump keyed `taco_am` | `CRLannister/Amharic` (downloaded, not in mix) |

Dataset nicknames (not acronyms, used as `source` tags): **Walia** = `EthioNLP/Amharic_Instruction_dataset`; **FineTome** = `addisai/FineTome-single-turn-dedup-amharic`; **Dolly** = `iocuydi/amharic-dolly-15k` (Amharic Dolly-15k).

---

## Split policy (not a random 80/20)

Train and test come from **different Hugging Face splits and authored holdouts**. Frozen evaluation (eval) never enters training. Mixes are hashed against `adtc/data/eval/` via `eval/dedup_against_eval.py`.

| Bucket | What goes in | HF split used |
|--------|----------------|---------------|
| **Train (SFT)** | Amharic instruct + English GSM8K tutoring (+ optional MT) | Hub **`train`** only |
| **Train (CPT, prepared, unused)** | FineWeb2-am / Wikipedia-am / AfriNLLB | Hub **`train`** (capped) |
| **Test / eval (frozen)** | AfriMGSM, AfriMMLU, AfriXNLI, GSM8K holdout | Hub **`test`** + authored sets |

There is **no validation split** in the current recipe. Quantized Low-Rank Adaptation (QLoRA) SFT loads the mix JSONL as `split="train"` only (`training/train_sft_qlora.py`).

**SFT actually used for Phase 4 adapters:** `data/train/sft_mix_v0.jsonl` (2 000 rows). Configs still point here (`training/configs/qlora_*.yaml`). `sft_mix_v1.jsonl` exists (No Language Left Behind / NLLB machine translation) but was **not** the training file for those runs.

---

## Test / eval (frozen — do not train)

Built by `eval/prepare_iroko_am.py`, `eval/prepare_en_stem.py`, plus authored JSONL. SHA-256 checksums for IrokoBench exports: `data/eval/eval_manifest_v0.json`.

| File | Rows | Downloaded from | HF config / split | Role |
|------|------|-----------------|-------------------|------|
| `data/eval/afrimgsm_amh_test_v0.jsonl` | 250 | [`masakhane/afrimgsm`](https://huggingface.co/datasets/masakhane/afrimgsm) | `amh` / **test** | Amharic math |
| `data/eval/afrimgsm_eng_test_v0.jsonl` | 250 | same | `eng` / **test** | EN math control + translate-test |
| `data/eval/afrimmlu_amh_test_v0.jsonl` | 500 | [`masakhane/afrimmlu`](https://huggingface.co/datasets/masakhane/afrimmlu) | `amh` / **test** | Amharic knowledge / STEM |
| `data/eval/afrixnli_amh_test_v0.jsonl` | 600 | [`masakhane/afrixnli`](https://huggingface.co/datasets/masakhane/afrixnli) | `amh` / **test** | Amharic NLI (entailment; frozen, not scored in current runners) |
| `data/eval/en_stem_holdout_v0.jsonl` | 100 | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) `main` | **test**, first 100 | EN STEM forgetting check |
| `data/eval/custom_tutoring_v0.jsonl` | 101 | **authored** (not HF) | `split: eval` | Product tutoring EN↔am |
| `data/eval/fertility_parallel_v0.jsonl` | 10 | **authored** EN‖am pairs | — | Tokenizer fertility only |

**Custom tutoring mix (101):** EN `en_en` 49, AM `am_am` 40, plus code-switch / bilingual (`en_am` / `am_en`) 12. Behaviors: solve, explain, hint, first_error, simplify, related_exercise, code_switch.

**GSM8K official sizes** (`main`): train 7 473 / test 1 319. We use **train[:2000]** for SFT and **test[:100]** for holdout — disjoint by construction.

Current accuracy scripts (`eval/run_hf_eval.py`, `eval/run_gguf_eval.py`) score **all** frozen AfriMGSM am+en, AfriMMLU am, English STEM holdout, and custom tutoring rows unless `--limit N` is passed. AfriXNLI is frozen on disk but not scored in those runners. GGUF here is the llama.cpp weight format used at eval time, not a training source.

---

## Train sources downloaded from Hugging Face

Script: `python data/download_train_sources.py --profile first_experiment`  
Job: `adtc/hpc/01_download_train.sbatch` (16 Aug 2026).  
Cache: `data/raw/hf/` and `data/raw/hf_home/`. Snapshots: `data/raw/snapshots/<key>.jsonl`.

Loader tries **`train`**, then `validation` / `test` / `dev` if needed. Every successful source below used **`train`**.

### OK (8)

| Key | Hugging Face id | Role | Requested split | Cap | Rows on disk | Snapshot |
|-----|-----------------|------|-----------------|-----|--------------|----------|
| `fineweb2_amh_100m` | [`MultilingualUnigramLM/FineWeb2-amh_Ethi-100M`](https://huggingface.co/datasets/MultilingualUnigramLM/FineWeb2-amh_Ethi-100M) | CPT | train | 50 000 | 50 000 | `snapshots/fineweb2_amh_100m.jsonl` |
| `wikipedia_amharic` | [`addisai/wikipedia-amharic`](https://huggingface.co/datasets/addisai/wikipedia-amharic) | CPT | train | 50 000 | 50 000 | `snapshots/wikipedia_amharic.jsonl` |
| `afrinllb` | [`AfriNLP/AfriNLLB-train`](https://huggingface.co/datasets/AfriNLP/AfriNLLB-train) | CPT / parallel | train | 50 000 | 50 000 | `snapshots/afrinllb.jsonl` |
| `walia` | [`EthioNLP/Amharic_Instruction_dataset`](https://huggingface.co/datasets/EthioNLP/Amharic_Instruction_dataset) | SFT | train | full | 122 425 | `snapshots/walia.jsonl` |
| `finetome_am` | [`addisai/FineTome-single-turn-dedup-amharic`](https://huggingface.co/datasets/addisai/FineTome-single-turn-dedup-amharic) | SFT | train | full | 83 290 | `snapshots/finetome_am.jsonl` |
| `r1_multilingual` | [`lightblue/reasoning-multilingual-R1-Llama-70B-train`](https://huggingface.co/datasets/lightblue/reasoning-multilingual-R1-Llama-70B-train) | SFT | train | 10 000 | **2 477** (stream ended) | `snapshots/r1_multilingual.jsonl` |
| `dolly_am` | [`iocuydi/amharic-dolly-15k`](https://huggingface.co/datasets/iocuydi/amharic-dolly-15k) | SFT | train | full | 15 011 | `snapshots/dolly_am.jsonl` |
| `taco_am` | [`CRLannister/Amharic`](https://huggingface.co/datasets/CRLannister/Amharic) | SFT | train | 20 000 | 20 000 | `snapshots/taco_am.jsonl` |

### FAIL (2)

| Key | Hugging Face id | Why |
|-----|-----------------|-----|
| `amharic_news` | [`dagn/expanded-amharic-news-dataset`](https://huggingface.co/datasets/dagn/expanded-amharic-news-dataset) | Gated Hub dataset — needs `HF_TOKEN` |
| `afriquellm_gsm8k` | [`peterlu02/afriquellm-coldstart-gsm8k-11lang`](https://huggingface.co/datasets/peterlu02/afriquellm-coldstart-gsm8k-11lang) | Dataset missing / inaccessible on Hub |

### Intentionally skipped

| Key | Hugging Face id | Why |
|-----|-----------------|-----|
| `fineweb2_full` | `HuggingFaceFW/fineweb-2` (`amh_Ethi`) | Too large; use 100M slice |
| `yoseali` | `YoseAli/amharic-llm-training-data` | Contamination risk — do not use wholesale |

---

## Extra train builders (not in `download_train_sources.py`)

| Artifact | Rows | From | Split | Notes |
|----------|------|------|-------|-------|
| `data/train/en_stem_sft_v0.jsonl` | 2 000 | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) `main` | **train** | `build_en_stem_sft.py --limit 2000`. All rows `source=gsm8k_train`. SciQ **not** included (`--include-sciq` unused). |
| `data/train/am_stem_sft_v0.jsonl` | 6 000 | NLLB stub over EN STEM | derived | `[AM-STUB]` prefix — pipeline smoke only. |
| `data/train/am_stem_sft_nllb_v1.jsonl` | 6 000 | EN STEM via [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M) | derived | `en_am` / `am_am` / `am_en` × 2 000. Source tag `mt_nllb:gsm8k_train`. |

[`allenai/sciq`](https://huggingface.co/datasets/allenai/sciq) (Science Questions) is in the catalog for optional English science SFT. It was **not downloaded** and is **not** in any mix.

`allenai/ai2_arc` (AI2 Reasoning Challenge, ARC-Easy) appears in the Hugging Face cache (`data/raw/hf/allenai___ai2_arc/`) from the profiler / EleutherAI lm-eval harness, **not** from the train pipeline.

---

## Normalized SFT pools (`data/train/sources/`)

`normalize_sft_sources.py` (run `b29f1b50`) rewrites snapshots to chat JSONL and dedups against frozen eval.

| File | Rows after normalize | In mix v0 / v1? |
|------|----------------------|-----------------|
| `walia_sft_v0.jsonl` | 122 416 (9 dropped vs snapshot) | **yes** |
| `finetome_am_sft_v0.jsonl` | 83 290 | **yes** |
| `dolly_am_sft_v0.jsonl` | 15 011 | no (downloaded only) |
| `taco_am_sft_v0.jsonl` | 20 000 | no (downloaded only) |
| `r1_am_sft_v0.jsonl` | **0** | no — snapshot had no Amharic after filter |

---

## Normalized CPT pools (prepared; continued-pretraining skipped)

Gate 4 decision: **do not run continued pre-training (CPT)** for the Gate 1 deadline. Files exist for later.

| File | Rows | From snapshot | Notes |
|------|------|---------------|-------|
| `data/train/cpt/native_am/fineweb2_amh_100m_v0.jsonl` | 50 000 | FineWeb2-am 100M | full snapshot |
| `data/train/cpt/native_am/wikipedia_amharic_v0.jsonl` | 20 000 | Wikipedia-am | normalize cap 20 000 (snapshot has 50 000) |
| `data/train/cpt/parallel/afrinllb_v0.jsonl` | 20 000 | AfriNLLB | same cap |
| Amharic news CPT | — | — | skipped (no snapshot) |

---

## SFT mixes (what training actually saw)

`mix_sft.py` sample: `--total 2000 --en-ratio 0.30 --am-ratio 0.55` → 600 EN + 1 100 AM + 300 EN replay, seed 42. Dedup vs `custom_tutoring_v0.jsonl` + `en_stem_holdout_v0.jsonl` dropped **0** rows.

### `sft_mix_v0.jsonl` — **used for QLoRA SFT** (2 000 rows)

Stub MT was **not** mixed in. Sources:

| `source` | Rows | Origin |
|----------|------|--------|
| `gsm8k_train` | 900 | GSM8K train (600 + 300 replay) |
| `walia` | 642 | EthioNLP Amharic instruction |
| `finetome_am` | 458 | FineTome Amharic |

### `sft_mix_v1.jsonl` — NLLB (machine-translated) mix on disk, not the Phase 4 train file (2 000 rows)

| `source` | Rows | Origin |
|----------|------|--------|
| `gsm8k_train` | 900 | GSM8K train |
| `walia` | 633 | EthioNLP |
| `finetome_am` | 442 | FineTome |
| `mt_nllb:gsm8k_train` | 25 | NLLB-translated GSM8K |

Directions in v1: `en_en` 900, `am_am` 1 081, `en_am` 9, `am_en` 10 (`docs/artifacts/sft_mix_v1_direction_counts.json`).

---

## End-to-end picture

```
Hugging Face Hub
  │
  ├─ TEST splits ──► data/eval/*_test_v0.jsonl     (frozen)
  │                    openai/gsm8k test[:100] ──► en_stem_holdout
  │                    authored ──► custom_tutoring, fertility
  │
  └─ TRAIN splits ──► data/raw/snapshots/*.jsonl
                         │
                         ├─ normalize ──► data/train/sources/*.jsonl
                         ├─ GSM8K train[:2000] ──► en_stem_sft_v0.jsonl
                         └─ optional NLLB ──► am_stem_sft_nllb_v1.jsonl
                                │
                                ▼
                         mix + dedup vs eval
                                │
                    sft_mix_v0.jsonl  ──► QLoRA SFT (Phase 4)
                    sft_mix_v1.jsonl  ──► on disk only
```

---

## Gaps vs catalog

| Planned | Status |
|---------|--------|
| AfriqueLLM Amharic GSM8K | Download failed — **not in any mix** |
| SciQ EN science | Not downloaded |
| Amharic news CPT | Gated; skipped |
| Dolly / TACO / R1 in mix | Downloaded (R1 normalized to 0 AM rows); **not sampled into mix** |
| Full FineWeb2 `amh_Ethi` | Skipped by design |
| CPT training | Data ready; **not run** |
| Train on `sft_mix_v1` | Mix exists; adapters still trained on **v0** |
| **SFT mix v2 (Amharic STEM)** | In flight — `sft_mix_v2.jsonl` (5 000) + QLoRA `*_qlora_v2` / GGUF `*_merged_v2` via `submit_v2_chain.sh`. FineTome dropped (0 Ethiopic). AfriqueLLM + simonbutt download retried. |

---

## Reproduce / inspect

```bash
cd adtc
python data/download_train_sources.py --profile first_experiment
python eval/prepare_iroko_am.py
python eval/prepare_en_stem.py --limit 100
python data/normalize_sft_sources.py --sources walia finetome r1 dolly taco
python data/build_en_stem_sft.py --limit 2000
python data/mix_sft.py \
  --en-stem data/train/en_stem_sft_v0.jsonl \
  --sft data/train/sources/walia_sft_v0.jsonl data/train/sources/finetome_am_sft_v0.jsonl \
  --eval data/eval/custom_tutoring_v0.jsonl data/eval/en_stem_holdout_v0.jsonl \
  --out data/train/sft_mix_v0.jsonl --total 2000

cat data/raw/download_manifest_v0.json
cat logs/download_train/latest.summary.json
```
