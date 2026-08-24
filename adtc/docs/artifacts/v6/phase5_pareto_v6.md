# Phase 5 — v6 profiler Pareto (Qwen3-1.7B EN)

| key | TPS | peak RSS MB | ARC | composite | ok |
|-----|-----|-------------|-----|-----------|----|
| `qwen3_1_7b_merged_v6-Q4_K_M` | 2.54 | 1897.66 | None | 19.79 | True |
| `qwen3_1_7b_merged_v6-Q5_K_M` | 2.46 | 1402.11 | None | 21.01 | True |
| `qwen3_1_7b_merged_v6-Q6_K` | 2.44 | 1553.22 | None | 20.55 | True |

**Gate 5 winner (v6):** `qwen3_1_7b_merged_v6-Q5_K_M` composite=21.01
- GGUF: `/scratch/nz2212/adtc-hackathon/adtc/artifacts/gguf/adapted/qwen3_1_7b_merged_v6-Q5_K_M.gguf`
- TPS=2.46 peak_rss_mb=1402.11
