#!/bin/bash
#SBATCH --job-name=adtc_merge_all
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Merge LoRA adapters into base HF weights (CPU) for every trained model.
#   sbatch 05_merge_lora.sbatch
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate /scratch/nz2212/adtc-hackathon/adtc/training/.conda-env
set -u

cd "${ADTC_ROOT}/training"

# base_hf_id | adapter_dir | merged_out_dir
MERGES=(
  "Qwen/Qwen3-1.7B|runs/qwen3_1_7b_qlora_v0/adapter|runs/qwen3_1_7b_merged_v0"
  "Qwen/Qwen3-4B|runs/qwen3_4b_qlora_v0/adapter|runs/qwen3_4b_merged_v0"
  "Qwen/Qwen2.5-3B-Instruct|runs/qwen25_3b_instruct_qlora_v0/adapter|runs/qwen25_3b_instruct_merged_v0"
  "google/gemma-3-4b-it|runs/gemma3_4b_qlora_v0/adapter|runs/gemma3_4b_merged_v0"
)

FAILED=0
for entry in "${MERGES[@]}"; do
  IFS='|' read -r BASE ADAPTER OUT <<<"${entry}"
  echo "========================================"
  echo "[merge] START base=${BASE} adapter=${ADAPTER} out=${OUT}"
  echo "========================================"
  if [[ ! -d "${ADAPTER}" ]]; then
    echo "[merge] SKIP missing adapter ${ADAPTER}" >&2
    FAILED=$((FAILED + 1))
    continue
  fi
  if python merge_lora.py --base "${BASE}" --adapter "${ADAPTER}" --out "${OUT}"; then
    echo "[merge] OK ${OUT}"
  else
    echo "[merge] FAIL ${OUT}" >&2
    FAILED=$((FAILED + 1))
  fi
done

echo "[merge] DONE failed=${FAILED} / total=${#MERGES[@]}"
exit "${FAILED}"
