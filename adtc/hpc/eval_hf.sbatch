#!/bin/bash
#SBATCH --job-name=adtc_eval_hf_v6
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# HF frozen eval for qwen3_1_7b_merged_v6.
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate "${ADTC_ROOT}/training/.conda-env"
set -u
cd "${ADTC_ROOT}"

MODEL="${ADTC_ROOT}/training/runs/qwen3_1_7b_merged_v6"
[[ -d "${MODEL}" ]] || { echo "missing ${MODEL}" >&2; exit 1; }
mkdir -p docs/artifacts/v6
OUT="docs/artifacts/v6/qwen3_1_7b_merged_v6_hf_eval.json"
python eval/run_hf_eval.py --model "${MODEL}" --out "${OUT}"
echo "[eval_hf_v6] OK ${OUT}"
