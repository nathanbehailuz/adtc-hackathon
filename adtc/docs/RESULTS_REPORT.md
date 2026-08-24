# Results report — v6 English-only Qwen3-1.7B

**As of:** 24 Aug 2026 (Jubail `$SCRATCH`).  
**Artifacts:** [`artifacts/v6/`](./artifacts/v6/). Day-to-day: [`DEVLOG.md`](./DEVLOG.md).

---

## Headline

**Deploy GGUF:** `adtc/artifacts/gguf/adapted/qwen3_1_7b_merged_v6-Q5_K_M.gguf`  
Gate 5 profiler winner: [`artifacts/v6/phase5_gate5_winner_v6.json`](./artifacts/v6/phase5_gate5_winner_v6.json)

| Metric | Q5_K_M GGUF |
|--------|------------:|
| Gen TPS | 2.46 |
| Peak RSS | 1402 MB |
| Composite (profiler) | 21.01 |
| S_tps | 16.4 |
| S_mem | 80.44 |

---

## HF frozen eval (merged checkpoint)

Source: [`artifacts/v6/qwen3_1_7b_merged_v6_hf_eval.json`](./artifacts/v6/qwen3_1_7b_merged_v6_hf_eval.json) — full frozen sets, no `--limit`.

| Suite | n | acc |
|-------|--:|----:|
| AfriMGSM EN | 250 | **0.392** |
| EN STEM holdout | 100 | **0.370** |
| Custom tutoring | 101 | **0.980** |
| AfriMGSM AM (secondary) | 250 | 0.024 |
| AfriMMLU AM (secondary) | 500 | 0.216 |

Primary EN KPIs improved vs v0 bilingual baseline on EN MGSM (0.34) and holdout (0.34) while keeping tutoring rubric near 1.0.

---

## GGUF frozen eval

| Quant | Artifact |
|-------|----------|
| Q4_K_M | [`artifacts/v6/qwen3_1_7b_merged_v6-Q4_K_M_eval.json`](./artifacts/v6/qwen3_1_7b_merged_v6-Q4_K_M_eval.json) |
| Q5_K_M | [`artifacts/v6/qwen3_1_7b_merged_v6-Q5_K_M_eval.json`](./artifacts/v6/qwen3_1_7b_merged_v6-Q5_K_M_eval.json) |

---

## Profiler / Gate 5

Pareto summary: [`artifacts/v6/phase5_pareto_v6.md`](./artifacts/v6/phase5_pareto_v6.md)  
Per-quant profiles: `phase5_profile_qwen3_1_7b_merged_v6-Q{4,5,6}_K*.json`

**Pick rationale:** Q5_K_M balances size (~1.2 GB) vs Q4 with acceptable TPS and lowest peak RSS among scored quants.

---

## Pipeline

```
GSM8K + SciQ → sft_mix_v6 (10473)
        │
        ▼
QLoRA SFT → merge → GGUF quants
        │
        ├── HF eval (full frozen)
        ├── GGUF eval (Q4, Q5)
        └── profiler → Q5_K_M winner
```

Reproduce: `cd adtc/hpc && bash submit_chain.sh`

---

## Caveats

| Caveat | Detail |
|--------|--------|
| Profiler numbers | Measured on Jubail compute nodes, not the 8 GB Standard Laptop |
| Amharic suites | Reported for completeness; v6 was not trained for Amharic |
| S_acc in profiler JSON | Hardware-only gate used `--skip-accuracy`; see HF/GGUF eval for accuracy |

---

## Scoring reference

ADTC-shaped composite (when accuracy included): `0.5*S_acc + 0.3*S_tps + 0.2*S_mem`.

Frozen accuracy suites for S_acc: AfriMGSM am/en, AfriMMLU am, EN STEM holdout (`eval/run_hf_eval.py`).
