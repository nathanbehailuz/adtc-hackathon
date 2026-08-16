#!/bin/bash
#SBATCH --job-name=adtc_dl_models
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Download HF base models for QLoRA (first pass: 1.7B only).
set -euo pipefail
HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"

cd "${ADTC_ROOT}"
python training/download_base_models.py --only qwen3_1_7b "$@"
