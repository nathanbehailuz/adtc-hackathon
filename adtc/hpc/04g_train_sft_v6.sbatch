#!/bin/bash
#SBATCH --job-name=adtc_sft_v6
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# QLoRA SFT v6: Qwen3-1.7B on English-only sft_mix_v6.
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate "${ADTC_ROOT}/training/.conda-env"
set -u

MIX="${ADTC_ROOT}/data/train/sft_mix_v6.jsonl"
CONFIG="${ADTC_ROOT}/training/configs/qlora_qwen3_1_7b_v6.yaml"
[[ -f "${MIX}" ]] || { echo "missing ${MIX} — run 02h_prepare_mix_v6.sbatch first" >&2; exit 1; }
[[ -f "${CONFIG}" ]] || { echo "missing ${CONFIG}" >&2; exit 1; }

cd "${ADTC_ROOT}/training"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
echo "[sft_v6] config=configs/qlora_qwen3_1_7b_v6.yaml mix=${MIX}"
python train_sft_qlora.py --config configs/qlora_qwen3_1_7b_v6.yaml
echo "[sft_v6] OK"
