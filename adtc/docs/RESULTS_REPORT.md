# Results report — measured decisions through Gate 5

**As of:** 18 Aug 2026 (Jubail `$SCRATCH`).  
**Plan / gates:** [`PRD.md`](./PRD.md). Day-to-day: [`DEVLOG.md`](./DEVLOG.md). Train data: [`DATASETS_REPORT.md`](./DATASETS_REPORT.md).  
**Tables:** [`artifacts/phase4_adapted_eval_v0.md`](./artifacts/phase4_adapted_eval_v0.md), [`artifacts/phase5_pareto_v0.md`](./artifacts/phase5_pareto_v0.md), [`artifacts/perf/leaderboard_v0.md`](./artifacts/perf/leaderboard_v0.md).

This is a **status report of numbers on disk**, not the short Gate 1 competition narrative (`REPORT.md` in the submission template). Dataset nicknames and Hub splits live in the datasets report.

**Read these caveats first**

| Caveat | What it means |
|--------|----------------|
| Frozen accuracy **in this report** is **n=50** | Numbers on disk used `--limit 50`. Runners now default to **all** frozen rows (AfriMGSM 250, AfriMMLU 500, holdout 100, tutoring 101). Pass `--limit N` only for a smoke. |
| Tutoring in this report is **n=20** and **not in S_acc** | Soft pedagogical pass; excluded from the ADTC-shaped total. Default is now the full 101. |
| Systems numbers are **Jubail compute nodes** | Not the 8 GB Standard Laptop. Phase 0 laptop numbers exist only for SmolLM2-135M. |
| Phase 5 Pareto used `--skip-accuracy` | Hardware-only. Accuracy on GGUFs is the later job `17261185` (three Q4 files only). |

---

## Headline

ADTC-shaped score on Jubail compute, formula `0.5*S_acc + 0.3*S_tps + 0.2*S_mem` ([`eval/aggregate_perf.py`](../eval/aggregate_perf.py)). Job `adtc_perf_eval-17261185`.

| Rank | Model (adapted Q4_K_M GGUF) | Total | S_acc | S_tps | S_mem | TPS | Peak RSS |
|------|-----------------------------|------:|------:|------:|------:|----:|---------:|
| 1 | `qwen3_1_7b_merged_v0-Q4_K_M` | 29.44 | 17.0 | 20.8 | 73.52 | 3.12 | 1.90 GB |
| 2 | `gemma3_4b_merged_v0-Q4_K_M` | 25.96 | 26.5 | 14.2 | 42.23 | 2.13 | 4.04 GB |
| 3 | `qwen3_4b_merged_v0-Q4_K_M` | 21.44 | 18.0 | 14.73 | 40.10 | 2.21 | 4.19 GB |

**Recommended deployment GGUF (hardware-weighted):** `adtc/artifacts/gguf/adapted/qwen3_1_7b_merged_v0-Q4_K_M.gguf`.

**Accuracy still favors Gemma 3 4B** (best Amharic MGSM / MMLU). The 1.7B Qwen wins the composite because memory and throughput outweigh its near-zero Amharic scores. Qwen2.5-3B has HF eval and GGUF hardware numbers but is **not** on this three-model GGUF leaderboard.

```
unadapted GGUF screen
        │
        ▼
QLoRA SFT on sft_mix_v0 (2 000 rows) → merge
        │
        ▼
HF eval n=50  ──►  PTQ Q8→Q4  ──►  hardware Pareto (skip-acc)
                                        │
                                        ▼
                          Q4 GGUF frozen + profiler + translate-test
                                        │
                                        ▼
                          leaderboard_v0  (this report)
```

---

## Acronyms

Dataset names (AfriMGSM, Walia, FineTome, NLLB, …) are in [`DATASETS_REPORT.md`](./DATASETS_REPORT.md). Scoring and systems terms used here:

| Acronym | Expansion | In this report |
|---------|-----------|----------------|
| **ADTC-shaped** | `0.5 S_acc + 0.3 S_tps + 0.2 S_mem` | Local proxy for the contest mix; **not** an official judge score |
| **S_acc** | Accuracy component (0–100) | 100 × mean of four frozen suite accuracies |
| **S_tps** | Throughput component (0–100) | 100 × min(1, gen TPS / 15) |
| **S_mem** | Memory component (0–100) | 100 × (7 − peak_GB) / 7, clipped to [0, 100] |
| **TPS** | tokens per second (generation) | `adtc-profiler` gen TPS |
| **TTFT** | time to first token | milliseconds |
| **RSS** | resident set size | Peak / steady process memory |
| **PTQ** | post-training quantization | llama.cpp Q8_0 / Q6_K / Q5_K_M / Q4_K_M |
| **Q4_K_M** | 4-bit K-quant (medium) | Deployment quant on the leaderboard |
| **QLoRA** | 4-bit base + LoRA adapters | Training memory only — not the GGUF quant |
| **HF eval** | Hugging Face merged checkpoint | `eval/run_hf_eval.py` on GPU |
| **GGUF eval** | llama.cpp weight file | `eval/run_gguf_eval.py` on CPU |
| **translate-test** | English AfriMGSM vs direct Amharic | Same items, index-aligned; gap = EN − AM |
| **fertility** | tokenizer fragmentation | `F_am`, `R_am/en` on 10 parallel pairs |
| **ARC-Easy** | AI2 Reasoning Challenge (easy) | Profiler `lm-eval` task, n=50; **not** frozen |

---

## Scoring formula

Implemented in [`eval/aggregate_perf.py`](../eval/aggregate_perf.py). Suites in S_acc: AfriMGSM am, AfriMGSM en, AfriMMLU am, English STEM holdout.

```
S_acc  = 100 × mean(acc_afrimgsm_amh, acc_afrimgsm_eng, acc_afrimmlu_amh, acc_en_stem_holdout)
S_tps  = 100 × min(1, TPS / 15)
S_mem  = 100 × (7 − peak_RSS_GB) / 7     # clipped to [0, 100]
total  = 0.5 S_acc + 0.3 S_tps + 0.2 S_mem
```

Example (rank-1 Qwen3-1.7B Q4): mean acc = (0.00 + 0.34 + 0.00 + 0.34) / 4 = 0.17 → S_acc = 17. Peak RSS 1897.76 MB ≈ 1.85 GB → S_mem ≈ 73.5. TPS 3.12 / 15 → S_tps = 20.8.

Thermal: none of the profiled runs set `throttled=true` or exceeded 85 °C peak (measured peaks ~65–69 °C on the Q4 leaderboard pass).

---

## Phase 0 — packaging smoke (laptop)

Gate 0: participant profiler on the template SmolLM2-135M Q4_K_M, **skip-accuracy**. Host was **not** Jubail. Artifacts: `docs/artifacts/phase0_submission_smoke.json`, `phase0_clean_clone_smoke.json`.

| Metric | Working-tree smoke | Clean-clone smoke |
|--------|-------------------:|------------------:|
| gen TPS | 86.45 | 81.7 |
| TTFT (ms) | 1312.71 | 707.34 |
| peak RSS (MB) | 270.58 | 272.39 |
| steady RSS (MB) | 220.73 | 221.52 |
| throttled | false | false |

Do **not** compare these TPS figures to the 1.7–4B HPC numbers below. Different model, different machine.

---

## Phase 2 — unadapted screen (Gate 2)

Jobs: fertility `17260639`; unadapted GGUF download + profiler `17260737`; translate-test `17260738`. Profiler: participant, `--skip-accuracy`, Jubail **compute**.

### Tokenizer fertility

`docs/artifacts/fertility_v0.json`. Ten authored EN‖am pairs (`data/eval/fertility_parallel_v0.jsonl`). High `R_am/en` = Amharic uses many more tokens than English for the same content.

| Tokenizer | mean F_am | mean R_am/en | Flag |
|-----------|----------:|-------------:|------|
| Qwen/Qwen3-1.7B | 5.78 | 4.31 | **severe** |
| Qwen/Qwen3-4B | 5.78 | 4.31 | **severe** |
| Qwen/Qwen2.5-3B-Instruct | 5.78 | 4.31 | **severe** |
| google/gemma-3-4b-it | 2.79 | 2.08 | **ok** |

Tokenizer extension was **not** started (Gate 4: defer unless post-Gate budget).

### Unadapted GGUF profiler (hardware only)

`docs/artifacts/phase2_unadapted_profile_summary_v0.json`. Qwen3.5-2B Q4 loaded (compat check) — kept as watchlist, not adapted.

| key | TPS | peak RSS MB | notes |
|-----|----:|------------:|-------|
| qwen3_1_7b_q4_k_m | 2.28 | 1897 | efficiency finalist |
| qwen3_1_7b_q6_k | 2.01 | 1553 | |
| qwen35_2b_q4_k_m | 1.71 | 1935 | compat only — loaded OK |
| qwen3_4b_q4_k_m | 1.55 | 4294 | |
| qwen25_3b_q4_k_m | 1.41 | 3466 | |
| gemma3_4b_q6_k | 1.35 | 3302 | |
| gemma3_4b_q4_k_m | 1.28 | 4090 | |
| qwen3_4b_q6_k | 1.26 | 3399 | |

### Unadapted translate-test

Only **Qwen3-1.7B** (HF, n=50 AfriMGSM). `docs/artifacts/phase2_translate_test_v0.md`.

| | Accuracy |
|--|----------:|
| Direct Amharic | 0.020 (1/50) |
| English translate-test | 0.040 (2/50) |
| Gap (EN − AM) | +0.020 |

Unadapted frozen AfriMGSM/MMLU for the other bases was **not** run.

### Gate 2

- **Efficiency finalist:** Qwen3-1.7B Q4_K_M (best TPS, lowest RSS).
- **Accuracy finalist:** Gemma 3 4B (only non-severe Amharic fertility; later adapted Amharic metrics also lead).

All four Phase 2 shortlist models were still QLoRA-adapted (plus Qwen2.5-3B as middle-size control). Qwen3.5 excluded from SFT.

---

## Phase 4 — QLoRA SFT + HF eval (Gate 4)

Train job `adtc_sft_all-17260420` (**failed=0 / total=4**). Merge `adtc_merge_all-17260554`. HF eval `adtc_eval_adapted-17260740`. Mix: `data/train/sft_mix_v0.jsonl` (2 000 rows: GSM8K 900 / Walia 642 / FineTome 458). **Not** `sft_mix_v1`.

Configs: `training/configs/qlora_*.yaml`. Shared: 1 epoch, max seq 2048, seed 42, 4-bit NF4, LoRA r=16 α=32. 1.7B: batch 1 × accum 8, lr 2e-4. Others: accum 16, lr 1.5e-4. A100 `bf16`.

### Train metrics

From `training/runs/*/qlora_v0/train_metrics.json`.

| Run | Base | Wall (s) | Train loss | Notes |
|-----|------|---------:|-----------:|-------|
| `qwen3_1_7b_qlora_v0` | Qwen/Qwen3-1.7B | 1159 | 1.332 | checkpoint-250 |
| `qwen3_4b_qlora_v0` | Qwen/Qwen3-4B | 1496 | 1.338 | mean token acc 0.745 |
| `qwen25_3b_instruct_qlora_v0` | Qwen/Qwen2.5-3B-Instruct | 1369 | 1.283 | lowest loss |
| `gemma3_4b_qlora_v0` | google/gemma-3-4b-it | 1597 | 1.562 | highest loss |

Merged HF folders: `training/runs/{qwen3_1_7b,qwen3_4b,qwen25_3b_instruct,gemma3_4b}_merged_v0`.

### Adapted HF eval (n=50 / suite, tutoring n=20)

`docs/artifacts/phase4_adapted_eval_v0.md`. GPU generation, not GGUF.

| model | am MGSM | en MGSM | am MMLU | en holdout | tutoring |
|-------|--------:|--------:|--------:|-----------:|---------:|
| gemma3_4b_merged_v0 | 0.18 | 0.32 | 0.26 | 0.32 | 0.70 |
| qwen25_3b_instruct_merged_v0 | 0.00 | 0.34 | 0.26 | 0.34 | 0.85 |
| qwen3_1_7b_merged_v0 | 0.00 | 0.36 | 0.00 | 0.36 | 1.00 |
| qwen3_4b_merged_v0 | 0.06 | 0.30 | 0.00 | 0.30 | 1.00 |

English STEM holdout ~0.30–0.36 (no catastrophic forget vs the 1.7B unadapted translate-test English 0.04 — different suite/size, so treat as a weak “still answers EN math” check, not a delta). Tutoring soft-pass is high. Amharic MGSM stays weak (best Gemma 0.18).

### Gate 4 / CPT

**Skip continued pre-training** for the Gate 1 deadline. CPT JSONL exists under `data/train/cpt/` (see datasets report). Revisit if Amharic remains the blocker after packaging.

`sft_mix_v1.jsonl` (NLLB machine translation, job `17260753`) is on disk and was **not** the train file.

---

## Phase 5 — GGUF PTQ + hardware Pareto (Gate 5)

Convert/quantize job `17260741` (**failed=0**): 20 GGUFs under `artifacts/gguf/adapted/` (f16 + Q8_0 / Q6_K / Q5_K_M / Q4_K_M × 4 models). Profiler job `17260742` (**COMPLETED**): 16 quantized candidates, participant, **`--skip-accuracy`**. Table: `docs/artifacts/phase5_pareto_v0.md`.

| key | TPS | peak RSS MB | steady RSS | throttled |
|-----|----:|------------:|-----------:|-----------|
| qwen3_1_7b_merged_v0-Q4_K_M | 3.10 | 1897 | 1802 | false |
| qwen3_1_7b_merged_v0-Q5_K_M | 2.99 | 1402 | 1305 | false |
| qwen3_1_7b_merged_v0-Q6_K | 2.96 | 1553 | 1456 | false |
| qwen3_1_7b_merged_v0-Q8_0 | 2.90 | 1952 | 1846 | false |
| qwen25_3b_instruct_merged_v0-Q4_K_M | 2.23 | 3287 | 3172 | false |
| qwen3_4b_merged_v0-Q4_K_M | 2.21 | 4294 | 4153 | false |
| qwen25_3b_instruct_merged_v0-Q5_K_M | 2.16 | 2314 | 2211 | false |
| qwen25_3b_instruct_merged_v0-Q6_K | 2.13 | 2614 | 2495 | false |
| gemma3_4b_merged_v0-Q4_K_M | 2.11 | 4141 | 3978 | false |
| qwen25_3b_instruct_merged_v0-Q8_0 | 2.06 | 3325 | 3202 | false |
| qwen3_4b_merged_v0-Q5_K_M | 2.03 | 3003 | 2879 | false |
| qwen3_4b_merged_v0-Q6_K | 2.02 | 3399 | 3264 | false |
| gemma3_4b_merged_v0-Q6_K | 1.99 | 3352 | 3191 | false |
| gemma3_4b_merged_v0-Q5_K_M | 1.93 | 3007 | 2841 | false |
| gemma3_4b_merged_v0-Q8_0 | 1.90 | 4248 | 4060 | false |
| qwen3_4b_merged_v0-Q8_0 | 1.88 | 4327 | 4181 | false |

**Gate 5 auto-heuristic winner:** `qwen3_1_7b_merged_v0-Q4_K_M` (best TPS / RSS). Accuracy arrays empty in this pass. Q8 vs Q4 **accuracy** drop was **not** measured — only hardware.

---

## GGUF accuracy leaderboard (job 17261185)

Later than Gate 5. Script: [`hpc/10_perf_eval.sbatch`](../hpc/10_perf_eval.sbatch). Per model: profiler **with** `arc_easy` n=50, frozen GGUF eval n=50, translate-test n=50 → `docs/artifacts/perf/<key>_v0.json`. Keys: the three Q4_K_M files above (Qwen2.5 omitted).

### Systems (profiler, accuracy mode)

| model | TPS | TTFT (ms) | peak RSS MB | steady RSS | T_peak °C | throttled |
|-------|----:|----------:|------------:|-----------:|----------:|-----------|
| qwen3_1_7b Q4_K_M | 3.12 | 4928 | 1898 | 1802 | 69.2 | false |
| gemma3_4b Q4_K_M | 2.13 | 10456 | 4141 | 3958 | 65.5 | false |
| qwen3_4b Q4_K_M | 2.21 | 11007 | 4294 | 4143 | 67.8 | false |

### Profiler ARC-Easy (n=50, acc_norm)

| model | ARC-Easy |
|-------|---------:|
| qwen3_1_7b Q4_K_M | 0.72 |
| gemma3_4b Q4_K_M | 0.78 |
| qwen3_4b Q4_K_M | 0.82 |

ARC-Easy is **not** in S_acc. English STEM holdout / AfriMGSM eng are.

### Frozen GGUF suites (n=50, tutoring n=20)

| model | am MGSM | en MGSM | am MMLU | en holdout | tutoring |
|-------|--------:|--------:|--------:|-----------:|---------:|
| qwen3_1_7b Q4_K_M | 0.00 (0/50) | 0.34 (17/50) | 0.00 (0/50) | 0.34 (17/50) | 1.00 (20/20) |
| gemma3_4b Q4_K_M | 0.12 (6/50) | 0.34 (17/50) | 0.26 (13/50) | 0.34 (17/50) | 0.85 (17/20) |
| qwen3_4b Q4_K_M | 0.08 (4/50) | 0.32 (16/50) | 0.00 (0/50) | 0.32 (16/50) | 1.00 (20/20) |

HF vs GGUF Q4 on the same n=50 slice (Amharic MGSM): Gemma 0.18 → 0.12; Qwen3-4B 0.06 → 0.08; Qwen3-1.7B 0.00 → 0.00. Small n; treat as noise except the Gemma drop.

### Adapted translate-test (n=50)

Direct Amharic AfriMGSM vs English AfriMGSM (same indices).

| model | Direct AM | Translate EN | Gap (EN − AM) |
|-------|----------:|-------------:|--------------:|
| qwen3_1_7b Q4_K_M | 0.00 (0/50) | 0.22 (11/50) | **+0.22** |
| gemma3_4b Q4_K_M | 0.08 (4/50) | 0.20 (10/50) | +0.12 |
| qwen3_4b Q4_K_M | 0.08 (4/50) | 0.12 (6/50) | +0.04 |

Unadapted 1.7B gap was +0.02 at ~floor accuracy. After SFT the English side rose (0.04 → 0.22) while direct Amharic stayed at 0.00 — language bottleneck, not “model cannot do grade-school math in English.”

### Human Amharic review

`human_amharic_review.status` in every perf JSON: **pending** (`n_samples: 0`). Review file on disk: `docs/artifacts/amharic_review_sample_v1.jsonl` (32 stratified Ethiopic rows). `african_alpha_claim` stays contingent on Nathan’s review ([`LANGUAGE.md`](./LANGUAGE.md)).

---

## Interpretation

1. **Composite score picks Qwen3-1.7B Q4** because S_mem and S_tps dominate. Peak RSS ~1.9 GB vs Gemma ~4.0 GB; TPS ~3.1 vs ~2.1. All three sit well under the 7 GB cap on this HPC node.
2. **Amharic accuracy still favors Gemma 3 4B**, consistent with fertility (`R_am/en` 2.08 vs 4.31). Qwen3-1.7B is 0.00 on both AfriMGSM am and AfriMMLU am after SFT.
3. **English STEM and tutoring look usable** (~0.32–0.36 holdout; tutoring 0.85–1.00). The product English path is much stronger than the Amharic path.
4. **Translate-test says the 1.7B failure is language, not reasoning capacity.** Direct AM 0.00 vs EN 0.22 on aligned items.
5. **SFT on mix_v0 (mostly EN GSM8K + generic Amharic instruct) did not fix Amharic math.** Mix_v1 (NLLB STEM) was not trained. CPT was skipped. AfriqueLLM Amharic GSM8K never downloaded.
6. **Qwen2.5-3B** is the odd HF row: 0.00 am MGSM but 0.26 am MMLU (same as Gemma). No GGUF frozen eval, so it cannot be ranked on the composite.

If judges weight accuracy much more than this local proxy, **reconsider Gemma Q4/Q5**. If they weight laptop RSS/TPS, **keep 1.7B Q4**.

---

## Gaps vs plan

Nothing below blocked writing this file. Full-set re-eval is now the default (`--limit` omitted).

| Planned / useful | Status |
|------------------|--------|
| Full AfriMGSM 250 / AfriMMLU 500 | **Artifacts still n=50**; scripts/jobs now score all rows unless `LIMIT=` is set |
| Qwen2.5-3B on GGUF leaderboard | HF eval yes; GGUF frozen **no** (omit from `10_perf_eval.sbatch` keys) |
| AfriXNLI scored | Frozen on disk; **not** in `run_hf_eval.py` / `run_gguf_eval.py` |
| Unadapted frozen AfriMGSM/MMLU (all bases) | Only 1.7B translate-test |
| Q8 vs Q4 **accuracy** drop | Hardware Pareto only |
| Standard Laptop profiler on adapted GGUFs | **Not run** (HPC compute only) |
| Human Amharic review | Sample exists; **pending** |
| Train on `sft_mix_v1` | Mix on disk; adapters still **v0** |
| CPT | Data ready; **skipped** (Gate 4) |
| Phase 6 ablations (mixture / distill / QAT) | **Not run** (optional) |
| Tokenizer extension | Fertility severe on Qwen; **not started** |
| Competition `REPORT.md` + packaged GGUF | Phase 7. Template dir on disk is **empty**. Separate from this file. |

---

## Reproduce / inspect (do not re-run unless needed)

```bash
cd /scratch/nz2212/adtc-hackathon/adtc

# HF adapted eval — default is now the full frozen files
# sbatch hpc/07_eval_adapted.sbatch

# Hardware Pareto (skip-accuracy) — phase5_pareto_v0.md
# sbatch hpc/09_profile_gguf.sbatch

# GGUF leaderboard (profiler + full frozen + translate-test)
# sbatch hpc/10_perf_eval.sbatch

python eval/aggregate_perf.py \
  --perf-dir docs/artifacts/perf \
  --keys qwen3_1_7b_merged_v0-Q4_K_M gemma3_4b_merged_v0-Q4_K_M qwen3_4b_merged_v0-Q4_K_M

cat docs/artifacts/perf/leaderboard_v0.md
cat docs/artifacts/phase4_adapted_eval_v0.md
cat docs/artifacts/phase5_pareto_v0.md
```

Selected job logs: `hpc/logs/adtc_sft_all-17260420.out`, `adtc_eval_adapted-17260740.out`, `adtc_prof_adapted-17260742.out`, `adtc_perf_eval-17261185.out`.

HPC acknowledgement: *This research was carried out on the High Performance Computing resources at New York University Abu Dhabi.*
