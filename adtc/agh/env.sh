#!/usr/bin/env bash
# Shared Africa GPU Hub environment for ADTC interactive / tmux jobs.
# Docs: https://docs.gpuhub.com/quickstart
# Console: https://console.aghcloud.ai/
set -euo pipefail

AGH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADTC_ROOT="$(cd "${AGH_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${ADTC_ROOT}/.." && pwd)"
cd "${ADTC_ROOT}"

# Prefer AGH data disk for caches/runs when present; fall back to repo-local paths.
# https://docs.gpuhub.com/container-instance/overview
if [[ -d /root/gpuhub-tmp ]]; then
  WORK="${ADTC_WORK:-/root/gpuhub-tmp/adtc}"
else
  WORK="${ADTC_WORK:-${ADTC_ROOT}/.agh-work}"
fi
mkdir -p "${WORK}"

CONDA_ENV="${ADTC_CONDA_ENV:-${ADTC_ROOT}/training/.conda-env}"

# Platform Miniconda (AGH images) or override.
if [[ -z "${MINICONDA_ACTIVATE:-}" ]]; then
  if [[ -f /root/miniconda3/bin/activate ]]; then
    MINICONDA_ACTIVATE="/root/miniconda3/bin/activate"
  elif [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
    MINICONDA_ACTIVATE="${HOME}/miniconda3/bin/activate"
  else
    MINICONDA_ACTIVATE=""
  fi
fi

export HF_HOME="${WORK}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${WORK}/hf_datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

# Optional durable File Storage mount (same region): /root/autodl-fs
# https://docs.gpuhub.com/data/file-storage
if [[ -d /root/autodl-fs ]]; then
  export ADTC_FILE_STORAGE="/root/autodl-fs"
fi

ENV_FILE="${ADTC_ROOT}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi
if [[ -n "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc 2>/dev/null || echo 8)}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [[ -n "${MINICONDA_ACTIVATE}" && -f "${MINICONDA_ACTIVATE}" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${MINICONDA_ACTIVATE}"
  if [[ -d "${CONDA_ENV}" ]]; then
    conda activate "${CONDA_ENV}"
  else
    echo "[env] WARN conda env missing at ${CONDA_ENV} — run setup_env.sh first" >&2
  fi
  set -u
elif command -v conda >/dev/null 2>&1 && [[ -d "${CONDA_ENV}" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  set -u
else
  echo "[env] WARN no conda activate found; using current python" >&2
fi

export AGH_DIR ADTC_ROOT REPO_ROOT WORK CONDA_ENV

echo "[env] AGH_DIR=${AGH_DIR}"
echo "[env] ADTC_ROOT=${ADTC_ROOT}"
echo "[env] WORK=${WORK}"
echo "[env] CONDA_PREFIX=${CONDA_PREFIX:-}"
echo "[env] HF_HOME=${HF_HOME}"
if [[ -n "${HF_TOKEN:-}" || -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "[env] HF_TOKEN=set"
else
  echo "[env] HF_TOKEN=unset (gated Hub models/datasets may fail)"
fi
if command -v python >/dev/null 2>&1; then
  echo "[env] python=$(command -v python) ($(python -V 2>&1))"
fi
