#!/usr/bin/env bash
# Shared Africa GPU Hub / Shadeform environment for ADTC jobs.
# Safe to `source` from an interactive shell (does not enable set -e).
# Docs: https://docs.gpuhub.com/quickstart | https://docs.shadeform.ai
#
# Usage:
#   cd adtc/agh && source env.sh
#   bash setup_env.sh

AGH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADTC_ROOT="$(cd "${AGH_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${ADTC_ROOT}/.." && pwd)"
# Do NOT cd here — callers stay in agh/ when sourcing interactively.

# Prefer AGH data disk for caches/runs when present; fall back to repo-local paths.
if [[ -d /root/gpuhub-tmp ]]; then
  WORK="${ADTC_WORK:-/root/gpuhub-tmp/adtc}"
elif [[ -d /home/shadeform && -w /home/shadeform ]]; then
  WORK="${ADTC_WORK:-${ADTC_ROOT}/.agh-work}"
else
  WORK="${ADTC_WORK:-${ADTC_ROOT}/.agh-work}"
fi
mkdir -p "${WORK}"

CONDA_ENV="${ADTC_CONDA_ENV:-${ADTC_ROOT}/training/.conda-env}"

# Platform Miniconda / conda locations (AGH, Shadeform, user installs).
if [[ -z "${MINICONDA_ACTIVATE:-}" ]]; then
  for _cand in \
    /root/miniconda3/bin/activate \
    "${HOME}/miniconda3/bin/activate" \
    /opt/conda/bin/activate \
    /opt/miniconda3/bin/activate \
    "${HOME}/anaconda3/bin/activate"
  do
    if [[ -f "${_cand}" ]]; then
      MINICONDA_ACTIVATE="${_cand}"
      break
    fi
  done
  unset _cand
fi

export HF_HOME="${WORK}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${WORK}/hf_datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

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

# Pinned llama.cpp b10451 binaries on PATH (gitignored under tools/).
# Do NOT put them on LD_LIBRARY_PATH here — that shadows llama-cpp-python's
# bundled libllama and breaks GGUF load in eval/try_prompt/judge_smoke.
# convert_gguf.sh / profile_gguf.sh set LD_LIBRARY_PATH only for native tools.
_LLAMA_BIN="${ADTC_ROOT}/tools/llama.cpp/llama-b10451"
if [[ -d "${_LLAMA_BIN}" ]]; then
  export PATH="${_LLAMA_BIN}:${PATH}"
fi
unset _LLAMA_BIN

if [[ -n "${MINICONDA_ACTIVATE:-}" && -f "${MINICONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${MINICONDA_ACTIVATE}"
  if [[ -d "${CONDA_ENV}" ]]; then
    conda activate "${CONDA_ENV}" || true
  else
    echo "[env] WARN conda env missing at ${CONDA_ENV} — run setup_env.sh first" >&2
  fi
elif command -v conda >/dev/null 2>&1 && [[ -d "${CONDA_ENV}" ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}" || true
else
  echo "[env] WARN no conda activate found; using current python" >&2
fi

export AGH_DIR ADTC_ROOT REPO_ROOT WORK CONDA_ENV
export MINICONDA_ACTIVATE="${MINICONDA_ACTIVATE:-}"

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
if command -v llama-quantize >/dev/null 2>&1; then
  echo "[env] llama-quantize=$(command -v llama-quantize)"
fi
