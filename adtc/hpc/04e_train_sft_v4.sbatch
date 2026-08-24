#!/bin/bash
#SBATCH --job-name=adtc_sft_v4
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Gemma-only SFT v4 on sft_mix_v4 (stock base; no CPT).
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate /scratch/nz2212/adtc-hackathon/adtc/training/.conda-env
set -u

MIX="${ADTC_ROOT}/data/train/sft_mix_v4.jsonl"
[[ -f "${MIX}" ]] || { echo "missing ${MIX}" >&2; exit 1; }

cd "${ADTC_ROOT}/training"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
echo "[sft_v4] START configs/qlora_gemma3_4b_v4.yaml"
python train_sft_qlora.py --config configs/qlora_gemma3_4b_v4.yaml
echo "[sft_v4] OK"
