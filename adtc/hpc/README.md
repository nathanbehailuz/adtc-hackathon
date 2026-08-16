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
cd /scratch/nz2212/adtc-hackathon/adtc/hpc
bash submit_chain.sh
```

That submits (with `afterok` dependencies):

1. `setup_env.sbatch` — scratch conda env + CUDA torch + `training/requirements.txt`
2. `01_download_train.sbatch` — train corpora
3. `02_prepare_data.sbatch` — normalize → EN STEM → stub MT → SFT mix
4. `03_download_models.sbatch` — `Qwen/Qwen3-1.7B`
5. `04_train_sft_1_7b.sbatch` — QLoRA on **A100** (`nvidia` partition)
6. `05_merge_lora.sbatch` — merge adapter → HF folder

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
