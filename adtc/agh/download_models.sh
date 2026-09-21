#!/usr/bin/env bash
# Download Qwen/Qwen3-1.7B into HF_HOME on AGH/Shadeform.
set -euo pipefail
AGH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${AGH_DIR}/env.sh"
mkdir -p "${AGH_DIR}/logs"
LOG="${AGH_DIR}/logs/download_models.log"
exec > >(tee -a "${LOG}") 2>&1

if [[ ! -d "${CONDA_ENV}" ]]; then
  echo "error: training conda env missing at ${CONDA_ENV}" >&2
  echo "  Run: bash setup_env.sh   (from adtc/agh, wait until it finishes)" >&2
  exit 1
fi

# Ensure we are on the training env, not base miniconda.
set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

python -c "import huggingface_hub" || {
  echo "error: huggingface_hub missing — re-run bash setup_env.sh" >&2
  exit 1
}

cd "${ADTC_ROOT}"
python training/download_base_models.py --only qwen3_1_7b
echo "[download] OK"
