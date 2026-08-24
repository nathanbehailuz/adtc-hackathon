# GGUF performance leaderboard (v0)

Formula: `0.5*S_acc + 0.3*S_tps + 0.2*S_mem`

| rank | model | total | S_acc | S_tps | S_mem | TPS | peak RSS MB |
|------|-------|-------|-------|-------|-------|-----|-------------|
| 1 | `qwen3_1_7b_merged_v0-Q4_K_M` | 29.444910714285715 | 17.0 | 20.8 | 73.52455357142857 | 3.12 | 1897.76 |
| 2 | `gemma3_4b_merged_v0-Q4_K_M` | 25.956010044642856 | 26.5 | 14.2 | 42.230050223214285 | 2.13 | 4140.95 |
| 3 | `qwen3_4b_merged_v0-Q4_K_M` | 21.440061383928573 | 18.0 | 14.733333333333334 | 40.10030691964287 | 2.21 | 4293.61 |

**Recommended submission GGUF:** `/scratch/nz2212/adtc-hackathon/adtc/artifacts/gguf/adapted/qwen3_1_7b_merged_v0-Q4_K_M.gguf`

