#!/usr/bin/env bash
# One-time: create conda env + install training deps (CUDA torch) on AGH.
# Usage: cd adtc/agh && bash setup_env.sh
set -euo pipefail
AGH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${AGH_DIR}/env.sh"
mkdir -p "${AGH_DIR}/logs"

REQS="${ADTC_ROOT}/training/requirements.txt"
LOG="${AGH_DIR}/logs/setup_env.log"
exec > >(tee -a "${LOG}") 2>&1

if [[ -z "${MINICONDA_ACTIVATE:-}" || ! -f "${MINICONDA_ACTIVATE}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "error: Miniconda not found. Install or use an AGH image with /root/miniconda3" >&2
    exit 1
  fi
fi

set +u
if [[ -n "${MINICONDA_ACTIVATE:-}" && -f "${MINICONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${MINICONDA_ACTIVATE}"
fi

if [[ ! -d "${CONDA_ENV}" ]]; then
  echo "[setup] Creating conda env at ${CONDA_ENV}"
  conda create -p "${CONDA_ENV}" python=3.11 -y
else
  echo "[setup] Reusing existing env at ${CONDA_ENV}"
fi
conda activate "${CONDA_ENV}"
set -u

python -m pip install --upgrade pip
# CUDA wheel — adjust cu121/cu124 to match the instance driver if needed.
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  "torch>=2.2.0"
python -m pip install -r "${REQS}"

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

echo "[setup] Done. Activate via: source ${AGH_DIR}/env.sh"
