# Jubail HPC (Slurm)

Run the ADTC data → QLoRA → merge pipeline on [NYUAD Jubail](https://crc-docs.abudhabi.nyu.edu/hpc/hpc.html).

**Do not run downloads or training on login nodes** — submit jobs with `sbatch`. Work from `$SCRATCH` only ([storage](https://crc-docs.abudhabi.nyu.edu/hpc/storage/index.html)).

Repo path on this account:

```bash
cd /scratch/nz2212/adtc-hackathon/adtc/hpc
```

## Quick start

```bash
# NYUAD VPN + SSH to Jubail login, then:
cd /scratch/nz2212/adtc-hackathon/adtc/hpc   # required — Slurm uses this as SLURM_SUBMIT_DIR
bash submit_chain.sh
# or one stage:
sbatch setup_env.sbatch
```

Always `cd` into `adtc/hpc` before `sbatch`. Jobs resolve paths via `SLURM_SUBMIT_DIR` (Slurm’s spool copy of the script is not writable).

### v6 English-only Qwen3-1.7B (profiler track)

```bash
cd /scratch/nz2212/adtc-hackathon/adtc/hpc
bash submit_v6_chain.sh   # prep → SFT → merge → GGUF → HF/GGUF eval → profile
# try_prompt after GGUF exists:
MODEL=10 PROMPTS=all sbatch 11_try_prompt.sbatch
```

Artifacts: `adtc/docs/artifacts/v6/`. See [`docs/artifacts/v6/MILESTONE_REPORT.md`](../docs/artifacts/v6/MILESTONE_REPORT.md).

That submits (with `afterok` dependencies):

1. `setup_env.sbatch` — scratch conda env + CUDA torch + `training/requirements.txt`
2. `01_download_train.sbatch` — train corpora
3. `02_prepare_data.sbatch` — normalize → EN STEM → stub MT → SFT mix
4. `03_download_models.sbatch` — Phase 2 HF bases: Qwen3-1.7B/4B, Gemma 3 4B-IT, Qwen2.5-3B-Instruct, Qwen3.5-2B/4B

5. `04_train_sft_1_7b.sbatch` / `04b_train_sft_4b.sbatch` — single-model QLoRA
6. **`04_train_sft_all.sbatch`** — sequential QLoRA: 1.7B → 4B → Qwen2.5-3B → Gemma3-4B (48h A100)
7. `05_merge_lora.sbatch` — merge adapter → HF folder

## Phases 2–5 (eval / GGUF / profiler)

Profiler is already in the scratch conda env (`setup_profiler.sbatch`). llama.cpp must be **built on Jubail** (`setup_build_llama_cpp.sbatch`) — official ubuntu binaries need a newer glibc.

```bash
cd /scratch/nz2212/adtc-hackathon/adtc/hpc
sbatch setup_build_llama_cpp.sbatch
sbatch 06_fertility.sbatch
sbatch 06a_download_unadapted_gguf.sbatch
# after build + download:
sbatch --dependency=afterok:<build>:<dl> 06b_profile_unadapted.sbatch
sbatch 06c_translate_test.sbatch          # nvidia GPU — full AfriMGSM unless LIMIT=
sbatch 02b_nllb_mix_v1.sbatch             # nvidia GPU — NLLB-200 → sft_mix_v1
sbatch 07_eval_adapted.sbatch             # nvidia GPU — four merged HF evals (full frozen sets)
sbatch --dependency=afterok:<build> 08_convert_gguf.sbatch
sbatch --dependency=afterok:<convert> 09_profile_gguf.sbatch
sbatch 10_perf_eval.sbatch                # compute CPU — GGUF frozen + translate (full) + profiler
sbatch 11_try_prompt.sbatch               # compute CPU — one GGUF + one prompt (MODEL=/PROMPT=)

# Amharic STEM SFT v2 (Gemma + Qwen3-1.7B; does not overwrite v0):
bash submit_v2_chain.sh
# or stepwise:
#   sbatch 02c_download_am_gsm8k.sbatch
#   sbatch 02d_prepare_mix_v2.sbatch
#   sbatch 04c_train_sft_v2.sbatch
#   sbatch 05b_merge_lora_v2.sbatch
#   sbatch 08b_convert_gguf_v2.sbatch
#   sbatch 12_eval_v2.sbatch              # CPU GGUF (judge-like, slow)
#   sbatch 12b_eval_v2_hf.sbatch          # GPU HF merged v2 (fast gate)
```

Shared helpers: [`env.sh`](env.sh), [`profiler_env.sh`](profiler_env.sh).

Optional later:

```bash
# After downloading 4B: python training/download_base_models.py --only qwen3_4b
sbatch 04b_train_sft_4b.sbatch
```

### Skip setup / resume mid-chain

```bash
SKIP_SETUP=1 bash submit_chain.sh          # env already built
CHAIN_FROM=4 bash submit_chain.sh          # from SFT onward (also skips setup)
sbatch 01_download_train.sbatch            # single stage
```

## Monitor

```bash
squeue -u $USER
scancel <jobid>
tail -f logs/adtc_sft_1_7b-*.out
cat ../logs/train_sft/latest.summary.json | python -m json.tool | head
```

| Logs | Location |
|------|----------|
| Slurm stdout/err | `adtc/hpc/logs/%x-%j.out` |
| Pipeline stage OK/FAIL | `adtc/logs/<stage>/` (see [`../docs/RUNLOGS.md`](../docs/RUNLOGS.md)) |

## Environment

- Conda prefix: `adtc/training/.conda-env` (gitignored)
- Shared activator: [`env.sh`](env.sh) — `module purge`, Miniconda activate, `HF_*` caches on scratch
- Caches: `adtc/data/raw/hf_home` (models), `adtc/data/raw/hf` (datasets)

Gated Hugging Face datasets (e.g. `amharic_news`): set `HF_TOKEN` in your shell before `sbatch`, or log in once with `huggingface-cli login` so the token is available to jobs.

**GPU:** SFT jobs request `--gres=gpu:a100:1` because training configs use `bf16` (V100 is not supported for this path). See [PyTorch on HPC](https://crc-docs.abudhabi.nyu.edu/hpc/software/hpc_pytorch.html) and [job quick start](https://crc-docs.abudhabi.nyu.edu/hpc/jobs/quick_start.html).

## Useful CRC links

- [Introduction](https://crc-docs.abudhabi.nyu.edu/hpc/hpc.html)
- [Jobs / Slurm](https://crc-docs.abudhabi.nyu.edu/hpc/jobs/quick_start.html)
- [Storage](https://crc-docs.abudhabi.nyu.edu/hpc/storage/index.html)
- Web: [Open OnDemand](https://ood.hpc.abudhabi.nyu.edu) (VPN), [Slurm stats](https://slurm.hpc.abudhabi.nyu.edu)
