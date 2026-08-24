#!/bin/bash
#SBATCH --job-name=adtc_sft_v3
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# QLoRA SFT v3: Gemma 3 4B only on sft_mix_v3, base = CPT-merged v3.
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate /scratch/nz2212/adtc-hackathon/adtc/training/.conda-env
set -u

MIX="${ADTC_ROOT}/data/train/sft_mix_v3.jsonl"
CPT_MERGED="${ADTC_ROOT}/training/runs/gemma3_4b_cpt_merged_v3"
if [[ ! -f "${MIX}" ]]; then
  echo "error: missing ${MIX} — run 02e_prepare_mix_v3.sbatch first" >&2
  exit 1
fi
if [[ ! -d "${CPT_MERGED}" ]]; then
  echo "error: missing ${CPT_MERGED} — run 03c_merge_cpt_v3.sbatch first" >&2
  exit 1
fi

cd "${ADTC_ROOT}/training"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; print(torch.cuda.get_device_name(0))"

echo "[sft_v3] START configs/qlora_gemma3_4b_v3.yaml"
python train_sft_qlora.py --config configs/qlora_gemma3_4b_v3.yaml
echo "[sft_v3] OK"
