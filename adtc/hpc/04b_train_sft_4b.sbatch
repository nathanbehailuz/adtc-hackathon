#!/bin/bash
#SBATCH --job-name=adtc_sft_4b
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# QLoRA SFT — Qwen3-4B (submit separately; not in default chain).
# Download 4B first: sbatch 03_download_models.sbatch --only is baked for 1.7B;
#   run: python training/download_base_models.py --only qwen3_4b
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"

cd "${ADTC_ROOT}/training"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; print(torch.cuda.get_device_name(0))"
python train_sft_qlora.py --config configs/qlora_qwen3_4b.yaml
