#!/usr/bin/env bash
# QLoRA SFT v7 on AGH (run inside tmux).
set -euo pipefail
AGH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${AGH_DIR}/env.sh"
mkdir -p "${AGH_DIR}/logs"
LOG="${AGH_DIR}/logs/train_sft.log"
exec > >(tee -a "${LOG}") 2>&1

if [[ ! -d "${CONDA_ENV}" ]]; then
  echo "error: missing ${CONDA_ENV} — run bash setup_env.sh first" >&2
  exit 1
fi
set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

MIX="${ADTC_ROOT}/data/train/sft_mix_v7.jsonl"
CONFIG="${ADTC_ROOT}/training/configs/qlora_qwen3_1_7b_v7.yaml"
[[ -f "${MIX}" ]] || { echo "missing ${MIX} — run prepare_mix.sh first" >&2; exit 1; }
[[ -f "${CONFIG}" ]] || { echo "missing ${CONFIG}" >&2; exit 1; }

cd "${ADTC_ROOT}/training"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; print(torch.cuda.get_device_name(0))"
echo "[sft_v7] config=configs/qlora_qwen3_1_7b_v7.yaml mix=${MIX}"
python train_sft_qlora.py --config configs/qlora_qwen3_1_7b_v7.yaml
echo "[sft_v7] OK"
