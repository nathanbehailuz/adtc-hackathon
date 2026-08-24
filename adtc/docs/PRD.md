# PRD — ADTC Offline English STEM Tutor (v6)

**Status:** Submission-ready (v6)  
**Domain:** `math_scientific_reasoning`  
**Artifact:** one GGUF + llama.cpp (no multi-model runtime)

---

## 1. Product summary

### Problem

Students in low-connectivity settings need a **STEM tutor that runs fully offline** on an 8 GB budget laptop, in **English**, with tutoring behaviors (explain, hint, diagnose error)—not just MCQ answers.

### Goal

Ship a **single Qwen3-1.7B specialist GGUF** that maximizes the ADTC score on English STEM + tutoring prompts while fitting laptop memory and throughput budgets.

| Weight | Metric     | Notes                    |
| ------ | ---------- | ------------------------ |
| 50%    | Accuracy   | Domain prompts (EN)      |
| 30%    | Throughput | Profiler gen TPS         |
| 20%    | Memory     | Peak RSS vs 7 GB budget  |

### Non-goals

- Multilingual / Amharic adaptation (removed from repo)
- Multi-model routers
- CPT (SFT-only v6 track)

### Model choice

| Item | Choice |
|------|--------|
| Base | `Qwen/Qwen3-1.7B` |
| Adaptation | QLoRA SFT on GSM8K + SciQ tutoring mix |
| Deploy quant | **Q5_K_M** (Gate 5 profiler winner) |
| Runtime | llama.cpp GGUF |

---

## 2. Pipeline (v6)

```
mix_sft_v6 (GSM8K + SciQ)
  → QLoRA SFT (A100)
  → merge LoRA
  → GGUF f16 → Q8/Q6/Q5/Q4
  → HF + GGUF frozen eval
  → ADTC profiler → pick Q5_K_M
  → submission template
```

Slurm: `adtc/hpc/submit_chain.sh`

---

## 3. Success criteria

### Must ship (Gate 1)

- [ ] Public GitHub repo with reproducible v6 pipeline
- [ ] `metadata.json` (`domain: math_scientific_reasoning`, `budget_laptop_claim: true`)
- [ ] Exactly **2** domain `test_prompts` (English tutoring)
- [ ] Idempotent `download_model.sh` → `_runtime.model_path`
- [ ] One valid `.gguf`, offline inference only
- [ ] `REPORT.md` with measured design narrative
- [ ] 2-minute demo video (laptop specs, offline, EN STEM tutoring, profiler numbers)

### Measured v6 KPIs (on disk)

See [`RESULTS_REPORT.md`](./RESULTS_REPORT.md) and [`artifacts/v6/`](./artifacts/v6/).

Primary EN suites: AfriMGSM EN, EN STEM holdout, custom tutoring.  
Amharic suites run as secondary diagnostics only (out of scope for training).

---

## 4. Files (v6 only)

| Stage | Path |
|-------|------|
| Mix | `data/mix_sft_v6.py` → `data/train/sft_mix_v6.jsonl` |
| Config | `training/configs/qlora_qwen3_1_7b_v6.yaml` |
| Train | `training/train_sft_qlora.py` |
| Merge | `training/merge_lora.py` |
| Eval | `eval/run_hf_eval.py`, `eval/run_gguf_eval.py` |
| HPC | `hpc/submit_chain.sh`, `prepare_mix`/`train_sft`/`merge_lora`/`convert_gguf`/`eval_hf`/`eval_gguf`/`profile_gguf` |
| Deploy GGUF | `artifacts/gguf/adapted/qwen3_1_7b_merged_v6-Q5_K_M.gguf` |

---

## 5. Thinking mode

Qwen3 supports extended thinking. For scoring and deployment:

- HF eval: `enable_thinking=False`
- GGUF eval / try_prompt: `/no_think` prefix where applicable

---

## 6. Acknowledgement

If results used NYUAD Jubail: *This research was carried out on the High Performance Computing resources at New York University Abu Dhabi.*
