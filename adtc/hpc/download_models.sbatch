#!/bin/bash
#SBATCH --job-name=adtc_dl_models
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Download v6 base: Qwen/Qwen3-1.7B
# Submit from adtc/hpc/ after setup_env.sbatch:
#   sbatch download_models.sbatch
# Already-cached weights are skipped/fast via Hub cache.
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
python training/download_base_models.py --only qwen3_1_7b
