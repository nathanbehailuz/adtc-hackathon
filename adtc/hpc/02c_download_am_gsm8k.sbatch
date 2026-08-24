#!/bin/bash
#SBATCH --job-name=adtc_dl_am_gsm8k
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Download AfriqueLLM GSM8K (+ simonbutt fallback) for SFT mix v2.
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate /scratch/nz2212/adtc-hackathon/adtc/training/.conda-env
set -u

cd "${ADTC_ROOT}"
python data/download_train_sources.py --only afriquellm_gsm8k simonbutt_amharic_gsm8k
echo "[dl_am_gsm8k] done"
