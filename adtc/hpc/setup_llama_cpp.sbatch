#!/bin/bash
#SBATCH --job-name=adtc_setup_llama
#SBATCH -p compute
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Download pinned llama.cpp linux binaries (b10451) under adtc/tools/llama.cpp/
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
ADTC_ROOT="$(cd "${HPC_DIR}/.." && pwd)"
TOOLS="${ADTC_ROOT}/tools"
URL="https://github.com/ggml-org/llama.cpp/releases/download/b10451/llama-b10451-bin-ubuntu-x64.tar.gz"
ARCHIVE="${TOOLS}/llama-b10451-bin-ubuntu-x64.tar.gz"

mkdir -p "${TOOLS}/llama.cpp" "${HPC_DIR}/logs"
cd "${TOOLS}"

if [[ -x "${TOOLS}/llama.cpp/llama-b10451/llama-bench" ]]; then
  echo "[llama] already present: ${TOOLS}/llama.cpp/llama-b10451/llama-bench"
else
  curl -L --fail -o "${ARCHIVE}" "${URL}"
  tar -xzf "${ARCHIVE}" -C "${TOOLS}/llama.cpp"
fi

# shellcheck disable=SC1091
source "${HPC_DIR}/profiler_env.sh"
llama-bench --version || true
echo "[llama] Done."
