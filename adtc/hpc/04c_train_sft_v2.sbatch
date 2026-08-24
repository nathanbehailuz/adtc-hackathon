#!/bin/bash
#SBATCH --job-name=adtc_sft_v2
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# QLoRA SFT v2: Gemma 3 4B then Qwen3-1.7B on sft_mix_v2 (does not overwrite v0).
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate /scratch/nz2212/adtc-hackathon/adtc/training/.conda-env
set -u

MIX="${ADTC_ROOT}/data/train/sft_mix_v2.jsonl"
if [[ ! -f "${MIX}" ]]; then
  echo "error: missing ${MIX} — run 02d_prepare_mix_v2.sbatch first" >&2
  exit 1
fi

cd "${ADTC_ROOT}/training"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; print(torch.cuda.get_device_name(0))"

CONFIGS=(
  configs/qlora_gemma3_4b_v2.yaml
  configs/qlora_qwen3_1_7b_v2.yaml
)

FAILED=0
for cfg in "${CONFIGS[@]}"; do
  echo "========================================"
  echo "[sft_v2] START ${cfg}"
  echo "========================================"
  if python train_sft_qlora.py --config "${cfg}"; then
    echo "[sft_v2] OK ${cfg}"
  else
    echo "[sft_v2] FAIL ${cfg}" >&2
    FAILED=$((FAILED + 1))
  fi
done

echo "[sft_v2] DONE failed=${FAILED} / total=${#CONFIGS[@]}"
exit "${FAILED}"
