#!/usr/bin/env bash
# Download Qwen/Qwen3-1.7B into HF_HOME on AGH.
set -euo pipefail
AGH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${AGH_DIR}/env.sh"
mkdir -p "${AGH_DIR}/logs"
LOG="${AGH_DIR}/logs/download_models.log"
exec > >(tee -a "${LOG}") 2>&1

cd "${ADTC_ROOT}"
python training/download_base_models.py --only qwen3_1_7b
echo "[download] OK"
