#!/bin/bash
#SBATCH --job-name=adtc_merge_v6
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Merge LoRA → HF for qwen3_1_7b_merged_v6 (leave v0–v5 intact).
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate "${ADTC_ROOT}/training/.conda-env"
set -u

cd "${ADTC_ROOT}/training"
BASE="Qwen/Qwen3-1.7B"
ADAPTER="runs/qwen3_1_7b_qlora_v6/adapter"
OUT="runs/qwen3_1_7b_merged_v6"
[[ -d "${ADAPTER}" ]] || { echo "missing ${ADAPTER}" >&2; exit 1; }
python merge_lora.py --base "${BASE}" --adapter "${ADAPTER}" --out "${OUT}"
echo "[merge_v6] OK ${OUT}"
