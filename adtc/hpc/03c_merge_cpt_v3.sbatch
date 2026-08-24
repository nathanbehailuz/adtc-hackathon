#!/bin/bash
#SBATCH --job-name=adtc_merge_cpt_v3
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Merge Gemma CPT LoRA → runs/gemma3_4b_cpt_merged_v3 (SFT base).
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
ADAPTER="runs/gemma3_4b_cpt_v3/adapter"
OUT="runs/gemma3_4b_cpt_merged_v3"

if [[ ! -d "${ADAPTER}" ]]; then
  echo "error: missing CPT adapter ${ADAPTER}" >&2
  exit 1
fi

echo "[merge_cpt_v3] base=${BASE} adapter=${ADAPTER} out=${OUT}"
python merge_lora.py --base "${BASE}" --adapter "${ADAPTER}" --out "${OUT}"
echo "[merge_cpt_v3] OK ${OUT}"
