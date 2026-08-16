#!/bin/bash
#SBATCH --job-name=adtc_merge
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Merge LoRA adapter into base HF weights (CPU).
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
# Activate scratch training conda (created by setup_env.sbatch)
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate /scratch/nz2212/adtc-hackathon/adtc/training/.conda-env
set -u

cd "${ADTC_ROOT}/training"
python merge_lora.py \
  --base Qwen/Qwen3-1.7B \
  --adapter runs/qwen3_1_7b_qlora_v0/adapter \
  --out runs/qwen3_1_7b_merged_v0
