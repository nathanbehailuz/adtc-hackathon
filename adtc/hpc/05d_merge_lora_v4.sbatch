#!/bin/bash
#SBATCH --job-name=adtc_merge_v4
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate /scratch/nz2212/adtc-hackathon/adtc/training/.conda-env
set -u
cd "${ADTC_ROOT}/training"

BASE="google/gemma-3-4b-it"
ADAPTER="runs/gemma3_4b_qlora_v4/adapter"
OUT="runs/gemma3_4b_merged_v4"
[[ -d "${ADAPTER}" ]] || { echo "missing ${ADAPTER}" >&2; exit 1; }
python merge_lora.py --base "${BASE}" --adapter "${ADAPTER}" --out "${OUT}"
echo "[merge_v4] OK ${OUT}"
