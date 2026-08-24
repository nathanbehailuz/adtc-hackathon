# Adapted HF / GGUF eval — v6 English-only Qwen3-1.7B

| model | am MGSM | en MGSM | am MMLU | en holdout | tutoring | TPS | peak RSS MB | composite |
|-------|--------:|--------:|--------:|-----------:|---------:|----:|------------:|----------:|
| `qwen3_1_7b_merged_v6` HF | 0.024 | **0.392** | 0.216 | **0.370** | **0.980** | — | — | — |
| `qwen3_1_7b_merged_v6-Q5_K_M` GGUF | see JSON | see JSON | — | — | — | **2.46** | **1402** | **21.01** |

Full artifacts: [`v6/`](./v6/).

Deploy pick: `artifacts/gguf/adapted/qwen3_1_7b_merged_v6-Q5_K_M.gguf`
