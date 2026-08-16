#!/usr/bin/env bash
# Shared Jubail environment for ADTC Slurm jobs.
# Source from sbatch scripts after #SBATCH directives.
# See: https://crc-docs.abudhabi.nyu.edu/hpc/software/hpc_pytorch.html
set -euo pipefail

# Slurm copies the batch script to a spool path; BASH_SOURCE is NOT the submit dir.
# Always prefer SLURM_SUBMIT_DIR (user must sbatch from adtc/hpc/).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  HPC_DIR="${SLURM_SUBMIT_DIR}"
else
  HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
ADTC_ROOT="$(cd "${HPC_DIR}/.." && pwd)"
cd "${ADTC_ROOT}"

CONDA_ENV="${ADTC_ROOT}/training/.conda-env"
MINICONDA_ACTIVATE="/share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate"

# Caches on $SCRATCH (avoid $HOME quota)
export HF_HOME="${ADTC_ROOT}/data/raw/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${ADTC_ROOT}/data/raw/hf"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false

# Clean module env; CUDA comes from the pip torch wheel (no cuda module load)
if command -v module >/dev/null 2>&1; then
  module purge 2>/dev/null || true
fi

if [[ ! -f "${MINICONDA_ACTIVATE}" ]]; then
  echo "error: Miniconda activate not found at ${MINICONDA_ACTIVATE}" >&2
  exit 1
fi
# Conda activate.d scripts reference unset vars; nounset must be off here.
set +u
# shellcheck disable=SC1090
source "${MINICONDA_ACTIVATE}"

if [[ ! -d "${CONDA_ENV}" ]]; then
  echo "error: conda env missing at ${CONDA_ENV}" >&2
  echo "  Submit setup_env.sbatch first (or: bash submit_chain.sh)." >&2
  exit 1
fi
conda activate "${CONDA_ENV}"
set -u

echo "[env] HPC_DIR=${HPC_DIR}"
echo "[env] ADTC_ROOT=${ADTC_ROOT}"
echo "[env] CONDA_PREFIX=${CONDA_PREFIX:-}"
echo "[env] HF_HOME=${HF_HOME}"
echo "[env] python=$(command -v python) ($(python -V 2>&1))"
