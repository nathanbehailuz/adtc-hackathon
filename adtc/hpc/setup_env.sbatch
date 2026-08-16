#!/bin/bash
#SBATCH --job-name=adtc_setup_env
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# One-time: create scratch conda env + install training deps (CUDA torch).
# Submit from adtc/hpc/ (login node):  sbatch setup_env.sbatch
set -euo pipefail

# Slurm spool ≠ submit dir — never use BASH_SOURCE for paths on Jubail.
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
ADTC_ROOT="$(cd "${HPC_DIR}/.." && pwd)"
CONDA_ENV="${ADTC_ROOT}/training/.conda-env"
MINICONDA_ACTIVATE="/share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate"
REQS="${ADTC_ROOT}/training/requirements.txt"

cd "${HPC_DIR}"
mkdir -p "${HPC_DIR}/logs"

if command -v module >/dev/null 2>&1; then
  module purge 2>/dev/null || true
fi

# Conda activate scripts reference unset vars (e.g. MKL_INTERFACE_LAYER);
# nounset must be off while sourcing/activating.
set +u
# shellcheck disable=SC1090
source "${MINICONDA_ACTIVATE}"

if [[ ! -d "${CONDA_ENV}" ]]; then
  echo "[setup] Creating conda env at ${CONDA_ENV}"
  conda create -p "${CONDA_ENV}" python=3.11 -y
else
  echo "[setup] Reusing existing env at ${CONDA_ENV}"
fi

conda activate "${CONDA_ENV}"
set -u

# Install CUDA-capable torch first, then the rest of training requirements.
# A100 (sm_80) — matches nvidia partition jobs that request --gres=gpu:a100:1
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  "torch>=2.2.0"
python -m pip install -r "${REQS}"

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available=", torch.cuda.is_available())
PY

echo "[setup] Done. Activate with: conda activate ${CONDA_ENV}"
