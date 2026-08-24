# Phase 5 — GGUF profiler Pareto

| key | TPS | peak RSS MB | steady RSS | throttled | ok |
|-----|-----|-------------|------------|-----------|----|
| `gemma3_4b_merged_v0-Q4_K_M` | 2.11 | 4140.74 | 3978.47 | False | True |
| `gemma3_4b_merged_v0-Q6_K` | 1.99 | 3351.65 | 3191.02 | False | True |
| `qwen25_3b_instruct_merged_v0-Q4_K_M` | 2.23 | 3287.02 | 3171.72 | False | True |
| `qwen25_3b_instruct_merged_v0-Q6_K` | 2.13 | 2613.71 | 2494.56 | False | True |
| `qwen3_1_7b_merged_v0-Q4_K_M` | 3.1 | 1897.39 | 1802.06 | False | True |
| `qwen3_1_7b_merged_v0-Q6_K` | 2.96 | 1552.92 | 1455.92 | False | True |
| `qwen3_4b_merged_v0-Q4_K_M` | 2.21 | 4293.85 | 4152.68 | False | True |
| `qwen3_4b_merged_v0-Q6_K` | 2.02 | 3398.52 | 3263.99 | False | True |
| `gemma3_4b_merged_v0-Q5_K_M` | 1.93 | 3007.16 | 2841.08 | False | True |
| `gemma3_4b_merged_v0-Q8_0` | 1.9 | 4247.61 | 4059.87 | False | True |
| `qwen25_3b_instruct_merged_v0-Q5_K_M` | 2.16 | 2313.67 | 2210.75 | False | True |
| `qwen25_3b_instruct_merged_v0-Q8_0` | 2.06 | 3325.13 | 3202.36 | False | True |
| `qwen3_1_7b_merged_v0-Q5_K_M` | 2.99 | 1402.1 | 1304.53 | False | True |
| `qwen3_1_7b_merged_v0-Q8_0` | 2.9 | 1951.68 | 1845.84 | False | True |
| `qwen3_4b_merged_v0-Q5_K_M` | 2.03 | 3003.18 | 2878.68 | False | True |
| `qwen3_4b_merged_v0-Q8_0` | 1.88 | 4326.74 | 4181.06 | False | True |

**Gate 5 winner (auto heuristic):** `qwen3_1_7b_merged_v0-Q4_K_M`
- GGUF: `/scratch/nz2212/adtc-hackathon/adtc/artifacts/gguf/adapted/qwen3_1_7b_merged_v0-Q4_K_M.gguf`
- TPS=3.1 peak_rss_mb=1897.39
