#!/bin/bash
#SBATCH --job-name=adtc_dl_models
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Download Phase 2 HF bases (PRD Must + optional controls).
#   Qwen3-1.7B, Qwen3-4B, Gemma 3 4B-IT, Qwen2.5-3B-Instruct, Qwen3.5-2B/4B
# Submit from adtc/hpc/ after setup_env.sbatch:
#   sbatch 03_download_models.sbatch
# Gemma is gated — set HF_TOKEN (or huggingface-cli login) before submit if needed.
# Already-cached models (e.g. Qwen3) are skipped/fast via Hub cache.
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
# Activate scratch training conda (created by setup_env.sbatch)
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate /scratch/nz2212/adtc-hackathon/adtc/training/.conda-env
set -u

cd "${ADTC_ROOT}"
python training/download_base_models.py
