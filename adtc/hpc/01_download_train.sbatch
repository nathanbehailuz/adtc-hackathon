#!/bin/bash
#SBATCH --job-name=adtc_dl_train
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Download first-experiment training corpora (HF → scratch cache).
set -euo pipefail
HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"

cd "${ADTC_ROOT}"
python data/download_train_sources.py --profile first_experiment "$@"
