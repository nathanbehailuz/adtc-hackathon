#!/usr/bin/env bash
# SFT v5. Env: V5_ALIAS, V5_CONFIG (optional)
#SBATCH --job-name=adtc_sft_v5
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 36:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate "${ADTC_ROOT}/training/.conda-env"
set -u

ALIAS="${V5_ALIAS:?}"
CONFIG="${V5_CONFIG:-configs/qlora_${ALIAS}_v5.yaml}"
MIX="${ADTC_ROOT}/data/train/sft_mix_v5.jsonl"
[[ -f "${MIX}" ]] || { echo "missing ${MIX}" >&2; exit 1; }
cd "${ADTC_ROOT}/training"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
echo "[sft_v5] alias=${ALIAS} config=${CONFIG}"
python train_sft_qlora.py --config "${CONFIG}"
echo "[sft_v5] OK"
