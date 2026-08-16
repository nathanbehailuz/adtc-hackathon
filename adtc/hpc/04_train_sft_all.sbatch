#!/bin/bash
#SBATCH --job-name=adtc_sft_all
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# QLoRA SFT for all Phase-4-ready configs (sequential on one A100).
# Order: efficiency → accuracy → middle control → Gemma (if cached).
# Qwen3.5 is NOT trained here (PRD: profiler compat check only).
#
#   sbatch 04_train_sft_all.sbatch
# Single models: 04_train_sft_1_7b.sbatch / 04b_train_sft_4b.sbatch
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate /scratch/nz2212/adtc-hackathon/adtc/training/.conda-env
set -u

MIX="${ADTC_ROOT}/data/train/sft_mix_v0.jsonl"
if [[ ! -f "${MIX}" ]]; then
  echo "error: missing ${MIX} — run 02_prepare_data.sbatch first" >&2
  exit 1
fi

cd "${ADTC_ROOT}/training"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; print(torch.cuda.get_device_name(0))"

CONFIGS=(
  configs/qlora_qwen3_1_7b.yaml
  configs/qlora_qwen3_4b.yaml
  configs/qlora_qwen25_3b_instruct.yaml
  configs/qlora_gemma3_4b.yaml
)

FAILED=0
for cfg in "${CONFIGS[@]}"; do
  echo "========================================"
  echo "[sft_all] START ${cfg}"
  echo "========================================"
  if python train_sft_qlora.py --config "${cfg}"; then
    echo "[sft_all] OK ${cfg}"
  else
    echo "[sft_all] FAIL ${cfg}" >&2
    FAILED=$((FAILED + 1))
    # Continue so one failure (e.g. gated Gemma) does not skip the rest
  fi
done

echo "[sft_all] DONE failed=${FAILED} / total=${#CONFIGS[@]}"
exit "${FAILED}"
