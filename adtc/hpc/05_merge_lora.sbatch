#!/bin/bash
#SBATCH --job-name=adtc_merge
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Merge LoRA adapter into base HF weights (CPU).
set -euo pipefail
HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"

cd "${ADTC_ROOT}/training"
python merge_lora.py \
  --base Qwen/Qwen3-1.7B \
  --adapter runs/qwen3_1_7b_qlora_v0/adapter \
  --out runs/qwen3_1_7b_merged_v0
