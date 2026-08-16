#!/bin/bash
#SBATCH --job-name=adtc_dl_gguf
#SBATCH -p compute
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 6:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Download unadapted official/community GGUFs for Phase 2 screen.
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
cd "${ADTC_ROOT}"
python eval/download_unadapted_gguf.py --out-dir artifacts/gguf/unadapted
echo "[dl_gguf] Done"
