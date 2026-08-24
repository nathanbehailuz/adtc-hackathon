# Jubail HPC (Slurm) — v6 English-only

Run the ADTC v6 pipeline on [NYUAD Jubail](https://crc-docs.abudhabi.nyu.edu/hpc/hpc.html): English-only Qwen3-1.7B SFT → merge → GGUF → eval → profiler.

**Do not run downloads or training on login nodes** — submit jobs with `sbatch`. Work from `$SCRATCH` only ([storage](https://crc-docs.abudhabi.nyu.edu/hpc/storage/index.html)).

Full pipeline overview: [`../docs/PIPELINE.md`](../docs/PIPELINE.md).

Repo path on this account:

```bash
cd /scratch/nz2212/adtc-hackathon/adtc/hpc
```

## Quick start

```bash
# NYUAD VPN + SSH to Jubail login, then:
cd /scratch/nz2212/adtc-hackathon/adtc/hpc   # required — Slurm uses this as SLURM_SUBMIT_DIR
sbatch setup_env.sbatch                       # once: scratch conda + CUDA torch
sbatch download_models.sbatch                 # Qwen/Qwen3-1.7B base
bash submit_chain.sh                          # full chain
```

Always `cd` into `adtc/hpc` before `sbatch`. Jobs resolve paths via `SLURM_SUBMIT_DIR`.

### Job scripts

| Script | Role |
|--------|------|
| `setup_env.sbatch` | Conda env + CUDA torch (once) |
| `download_models.sbatch` | HF base `Qwen/Qwen3-1.7B` |
| `prepare_mix.sbatch` | GSM8K + SciQ → `sft_mix_v6.jsonl` |
| `train_sft.sbatch` | QLoRA SFT (A100) |
| `merge_lora.sbatch` | Merge → `runs/qwen3_1_7b_merged_v6` |
| `convert_gguf.sbatch` | f16 + Q8/Q6/Q5/Q4 GGUFs |
| `eval_hf.sbatch` | HF frozen eval |
| `eval_gguf.sbatch` | GGUF frozen eval (`V6_QUANT`) |
| `profile_gguf.sbatch` | Profiler + Gate 5 pick |
| `try_prompt.sbatch` | Qualitative smoke (English prompts) |
| `submit_chain.sh` | Submit prep→SFT→merge→GGUF→eval→profile with dependencies |

Artifacts: `adtc/docs/artifacts/v6/`.

### try_prompt (qualitative smoke)

```bash
sbatch try_prompt.sbatch
PROMPTS=all sbatch try_prompt.sbatch
```

### One-off setup jobs

```bash
sbatch setup_profiler.sbatch
sbatch setup_build_llama_cpp.sbatch   # llama.cpp on Jubail (needed for GGUF convert)
```

Shared helpers: [`env.sh`](env.sh), [`profiler_env.sh`](profiler_env.sh).

## Monitor

```bash
squeue -u $USER
scancel <jobid>
tail -f logs/adtc_sft_v6-*.out
```

| Logs | Location |
|------|----------|
| Slurm stdout/err | `adtc/hpc/logs/%x-%j.out` |
| Pipeline stage OK/FAIL | `adtc/logs/<stage>/` (see [`../docs/RUNLOGS.md`](../docs/RUNLOGS.md)) |

## Environment

- Conda prefix: `adtc/training/.conda-env` (gitignored)
- Shared activator: [`env.sh`](env.sh) — `module purge`, Miniconda activate, `HF_*` caches on scratch
- Caches: `adtc/data/raw/hf_home` (models), `adtc/data/raw/hf` (GSM8K/SciQ datasets)

**GPU:** SFT jobs request `--gres=gpu:a100:1` because training config uses `bf16`.

## Useful CRC links

- [Introduction](https://crc-docs.abudhabi.nyu.edu/hpc/hpc.html)
- [Jobs / Slurm](https://crc-docs.abudhabi.nyu.edu/hpc/jobs/quick_start.html)
- [Storage](https://crc-docs.abudhabi.nyu.edu/hpc/storage/index.html)
