# ADTC Hackathon — Offline English + Amharic STEM Tutor

Gate 1 project for the ADTC 2026 offline multilingual STEM tutor track (`math_scientific_reasoning`).

**Goal:** one dense **1.7–4B GGUF** (Qwen3 → QLoRA → merge → quantize) that runs on an 8 GB laptop via llama.cpp, in **English + Amharic**, with tutoring behavior (explain / hint / diagnose)—not just MCQ answers.

Product plan and phased gates: [`adtc/docs/PRD.md`](adtc/docs/PRD.md).

## Layout

```
adtc/
  docs/          PRD, datasets, tooling, run logs, methodology
  data/          Train builders + frozen eval (data/eval/)
  training/      QLoRA SFT / CPT / merge (HF checkpoints)
  hpc/           Jubail Slurm jobs (scratch + A100)
  eval/          Eval prep, dedup, fertility
adtc-2026-submission-template/   Gate packaging (metadata, download_model.sh)
```

## Where to run what

| Work | Where | How |
|------|--------|-----|
| Train data + QLoRA + merge | **NYUAD Jubail** (`$SCRATCH`) | [`adtc/hpc/README.md`](adtc/hpc/README.md) |
| Training details | GPU machine / job | [`adtc/training/README.md`](adtc/training/README.md) |
| Profiler / GGUF smoke | Laptop | [`adtc/docs/TOOLING.md`](adtc/docs/TOOLING.md) |
| Stage OK/FAIL logs | Anywhere | [`adtc/docs/RUNLOGS.md`](adtc/docs/RUNLOGS.md) |

**Do not** run downloads or training on Jubail login nodes. Jobs must run from `/scratch/<NetID>/…` ([CRC storage](https://crc-docs.abudhabi.nyu.edu/hpc/storage/index.html)).

### Jubail quick start

```bash
cd /scratch/nz2212/adtc-hackathon/adtc/hpc
bash submit_chain.sh   # setup → data → models → SFT 1.7B → merge
squeue -u $USER
```

SFT uses partition `nvidia` with **A100** (`bf16`). Slurm logs: `adtc/hpc/logs/`. Pipeline logs: `adtc/logs/`.

### Local / interactive training (non-Slurm)

```bash
cd adtc/training
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# then follow adtc/training/README.md (data → download_base_models → train_sft_qlora → merge)
```

## Pipeline (locked)

HF base → **QLoRA SFT** (cloud/HPC GPU) → merge LoRA → high-precision GGUF → PTQ (Q8→Q4) → ADTC profiler → one final GGUF in the submission template.

`QLoRA 4-bit during training ≠ GGUF Q4 at deployment.`

## Docs index

| Doc | Contents |
|-----|----------|
| [`adtc/docs/PRD.md`](adtc/docs/PRD.md) | Product + step-by-step plan |
| [`adtc/docs/DATASETS.md`](adtc/docs/DATASETS.md) | Train / eval sources |
| [`adtc/docs/RESULTS_REPORT.md`](adtc/docs/RESULTS_REPORT.md) | Measured eval / profiler / leaderboard |
| [`adtc/docs/LANGUAGE.md`](adtc/docs/LANGUAGE.md) | Amharic language lock |
| [`adtc/docs/TOOLING.md`](adtc/docs/TOOLING.md) | Profiler / llama.cpp pins |
| [`adtc/docs/DEVLOG.md`](adtc/docs/DEVLOG.md) | Day-to-day progress |
| [`adtc/docs/GEMMA_V4_CONTEXT.md`](adtc/docs/GEMMA_V4_CONTEXT.md) | Gemma v4 architecture / data / hypers / eval briefing for external LLMs |
| [`adtc/hpc/README.md`](adtc/hpc/README.md) | Jubail Slurm |

## Acknowledgement (HPC)

If results used NYUAD Jubail, include: *This research was carried out on the High Performance Computing resources at New York University Abu Dhabi.*
