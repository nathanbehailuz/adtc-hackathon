# TebebAI — Offline English STEM Tutor (ADTC 2026)

Gate 1 project for the Africa Deep Tech Challenge 2026 Laptop LLM track (`math_scientific_reasoning`).

**Goal:** one dense **Qwen3-1.7B GGUF** (QLoRA SFT → merge → quantize) that runs on an 8 GB laptop via llama.cpp, with English tutoring behavior (explain / hint / diagnose)—not just MCQ answers.

**Deploy pick:** `tebeb_tutor_1.7b-Q5_K_M.gguf` (Gate 5 profiler winner).

| Metric | Value |
|--------|------:|
| Gen TPS | 2.46 |
| Peak RSS | 1402 MB |
| Profiler composite | 21.01 |
| Custom tutoring (HF) | 98% |
| AfriMGSM EN | 39.2% |

**Submission package:** [`adtc-2026-submission-template/`](adtc-2026-submission-template/) — `metadata.json`, `download_model.sh`, `REPORT.md`, GGUF via download, plus local `chat.py` demo.

Milestone report: [`adtc/docs/artifacts/v6/MILESTONE_REPORT.md`](adtc/docs/artifacts/v6/MILESTONE_REPORT.md).  
TebebAI writeup: [`adtc-2026-submission-template/REPORT.md`](adtc-2026-submission-template/REPORT.md).

## Try the tutor (submission demo)

```bash
cd adtc-2026-submission-template
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash download_model.sh
python chat.py
```

## Layout

```
adtc/
  docs/          PRD, datasets, tooling, v6 results
  data/          mix_sft_v6.py + frozen eval (data/eval/)
  training/      QLoRA SFT / merge (HF checkpoints)
  hpc/           Jubail Slurm jobs (scratch + A100)
  eval/          HF/GGUF eval, try_prompt, submission staging
adtc-2026-submission-template/   Gate packaging + chat.py demo
```

## Where to run what

| Work | Where | How |
|------|--------|-----|
| Interactive chat demo | Laptop / compute | [`adtc-2026-submission-template/`](adtc-2026-submission-template/) (`chat.py`) |
| Train data + QLoRA + merge | **NYUAD Jubail** (`$SCRATCH`) | [`adtc/hpc/README.md`](adtc/hpc/README.md) |
| Training details | GPU machine / job | [`adtc/training/README.md`](adtc/training/README.md) |
| Profiler / GGUF smoke | Laptop or Jubail compute | [`adtc/docs/TOOLING.md`](adtc/docs/TOOLING.md) |
| Stage OK/FAIL logs | Anywhere | [`adtc/docs/RUNLOGS.md`](adtc/docs/RUNLOGS.md) |

**Do not** run downloads or training on Jubail login nodes. Jobs must run from `/scratch/<NetID>/…` ([CRC storage](https://crc-docs.abudhabi.nyu.edu/hpc/storage/index.html)).

### Jubail quick start

```bash
cd /scratch/nz2212/adtc-hackathon/adtc/hpc
sbatch setup_env.sbatch          # once
sbatch download_models.sbatch    # Qwen3-1.7B base
bash submit_chain.sh             # prep → SFT → merge → GGUF → eval → profile
squeue -u $USER
```

SFT uses partition `nvidia` with **A100** (`bf16`). Slurm logs: `adtc/hpc/logs/`. Pipeline logs: `adtc/logs/`.

### Local / interactive training (non-Slurm)

```bash
cd adtc/training
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# then follow adtc/training/README.md
```

## Pipeline

HF base (`Qwen/Qwen3-1.7B`) → **QLoRA SFT** (GSM8K + SciQ mix) → merge LoRA → GGUF f16 → PTQ (Q8→Q4) → ADTC profiler → submission GGUF.

`QLoRA 4-bit during training ≠ GGUF Q4 at deployment.`

## Docs index

| Doc | Contents |
|-----|----------|
| [`adtc/docs/PIPELINE.md`](adtc/docs/PIPELINE.md) | End-to-end data → model → train → results |
| [`adtc/docs/PRD.md`](adtc/docs/PRD.md) | Product + v6 plan |
| [`adtc/docs/DATASETS.md`](adtc/docs/DATASETS.md) | Train / eval sources |
| [`adtc/docs/RESULTS_REPORT.md`](adtc/docs/RESULTS_REPORT.md) | v6 eval + profiler numbers |
| [`adtc/docs/TOOLING.md`](adtc/docs/TOOLING.md) | Profiler / llama.cpp pins |
| [`adtc/docs/DEVLOG.md`](adtc/docs/DEVLOG.md) | Day-to-day progress |
| [`adtc/docs/artifacts/v6/MILESTONE_REPORT.md`](adtc/docs/artifacts/v6/MILESTONE_REPORT.md) | v6 English-only Qwen3-1.7B |
| [`adtc/hpc/README.md`](adtc/hpc/README.md) | Jubail Slurm |

## Acknowledgement (HPC)

If results used NYUAD Jubail, include: *This research was carried out on the High Performance Computing resources at New York University Abu Dhabi.*
