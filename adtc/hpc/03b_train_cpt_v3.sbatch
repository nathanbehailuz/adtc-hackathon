#!/bin/bash
#SBATCH --job-name=adtc_cpt_v3
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# CPT Gemma 3 4B on cpt_mix_v3 (stock tokenizer). Does not overwrite v0/v2.
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate /scratch/nz2212/adtc-hackathon/adtc/training/.conda-env
set -u

MIX="${ADTC_ROOT}/data/train/cpt_mix_v3.jsonl"
if [[ ! -f "${MIX}" ]]; then
  echo "error: missing ${MIX} — run 02e_prepare_mix_v3.sbatch first" >&2
  exit 1
fi

cd "${ADTC_ROOT}/training"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; print(torch.cuda.get_device_name(0))"

echo "[cpt_v3] START configs/cpt_gemma3_4b_v3.yaml"
python train_cpt_qlora.py --config configs/cpt_gemma3_4b_v3.yaml
echo "[cpt_v3] OK"
