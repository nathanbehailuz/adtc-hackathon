#!/bin/bash
#SBATCH --job-name=adtc_merge_v2
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Merge LoRA adapters → HF folders for v2 runs only (leave *_merged_v0 intact).
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate /scratch/nz2212/adtc-hackathon/adtc/training/.conda-env
set -u

cd "${ADTC_ROOT}/training"

MERGES=(
  "google/gemma-3-4b-it|runs/gemma3_4b_qlora_v2/adapter|runs/gemma3_4b_merged_v2"
  "Qwen/Qwen3-1.7B|runs/qwen3_1_7b_qlora_v2/adapter|runs/qwen3_1_7b_merged_v2"
)

FAILED=0
for entry in "${MERGES[@]}"; do
  IFS='|' read -r BASE ADAPTER OUT <<<"${entry}"
  echo "========================================"
  echo "[merge_v2] START base=${BASE} adapter=${ADAPTER} out=${OUT}"
  echo "========================================"
  if [[ ! -d "${ADAPTER}" ]]; then
    echo "[merge_v2] SKIP missing adapter ${ADAPTER}" >&2
    FAILED=$((FAILED + 1))
    continue
  fi
  if python merge_lora.py --base "${BASE}" --adapter "${ADAPTER}" --out "${OUT}"; then
    echo "[merge_v2] OK ${OUT}"
  else
    echo "[merge_v2] FAIL ${OUT}" >&2
    FAILED=$((FAILED + 1))
  fi
done

echo "[merge_v2] DONE failed=${FAILED} / total=${#MERGES[@]}"
exit "${FAILED}"
