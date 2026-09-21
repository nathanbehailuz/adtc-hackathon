#!/usr/bin/env bash
# One-time: create conda env + install training deps (CUDA torch) on AGH/Shadeform.
# Usage: cd adtc/agh && bash setup_env.sh
set -euo pipefail
AGH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${AGH_DIR}/env.sh"
mkdir -p "${AGH_DIR}/logs"

REQS="${ADTC_ROOT}/training/requirements.txt"
LOG="${AGH_DIR}/logs/setup_env.log"
exec > >(tee -a "${LOG}") 2>&1

install_miniconda() {
  local prefix="${HOME}/miniconda3"
  if [[ -x "${prefix}/bin/conda" ]]; then
    return 0
  fi
  echo "[setup] Installing Miniconda to ${prefix} …"
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "${prefix}"
  MINICONDA_ACTIVATE="${prefix}/bin/activate"
  export MINICONDA_ACTIVATE
}

if [[ -z "${MINICONDA_ACTIVATE:-}" || ! -f "${MINICONDA_ACTIVATE}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    echo "[setup] Using conda on PATH: $(command -v conda)"
  else
    install_miniconda
  fi
fi

set +u
if [[ -n "${MINICONDA_ACTIVATE:-}" && -f "${MINICONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${MINICONDA_ACTIVATE}"
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "error: conda still not available after install attempt" >&2
  exit 1
fi

# Recent Miniconda requires explicit ToS acceptance before `conda create`.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

if [[ ! -d "${CONDA_ENV}" ]]; then
  echo "[setup] Creating conda env at ${CONDA_ENV}"
  conda create -p "${CONDA_ENV}" python=3.11 -y
else
  echo "[setup] Reusing existing env at ${CONDA_ENV}"
fi
conda activate "${CONDA_ENV}"
set -u

python -m pip install --upgrade pip
# CUDA 12.x wheel — A6000 driver 580 is fine with cu121/cu124.
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  "torch>=2.2.0" || \
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
