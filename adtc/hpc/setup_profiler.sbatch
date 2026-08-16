#!/bin/bash
#SBATCH --job-name=adtc_setup_profiler
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Install pinned adtc-profiler (+ llama-cpp-python) into the scratch conda env.
# llama-cpp-python must compile C++ — needs gcc+g++ modules (login node often fails).
# Submit from adtc/hpc/:  sbatch setup_profiler.sbatch
set -euo pipefail

HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
ADTC_ROOT="$(cd "${HPC_DIR}/.." && pwd)"
CONDA_ENV="${ADTC_ROOT}/training/.conda-env"
MINICONDA_ACTIVATE="/share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate"
# Matches adtc/docs/TOOLING.md
PROFILER_SHA="ac2e137dca65ea3b09d997774f17dd8907b489fb"

cd "${HPC_DIR}"
mkdir -p "${HPC_DIR}/logs"

if command -v module >/dev/null 2>&1; then
  module purge 2>/dev/null || true
fi
# gcc/13.2.0 on Jubail is C-only; use a full toolchain with g++.
module load gcc/12.2.0
module load cmake/3.31.2
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
# Avoid host-tuned ISA so the wheel runs on any compute node CPU.
export CMAKE_ARGS="-DGGML_NATIVE=OFF"
export CMAKE_GENERATOR="Unix Makefiles"

set +u
# shellcheck disable=SC1090
source "${MINICONDA_ACTIVATE}"
if [[ ! -d "${CONDA_ENV}" ]]; then
  echo "error: conda env missing at ${CONDA_ENV}; run setup_env.sbatch first" >&2
  exit 1
fi
conda activate "${CONDA_ENV}"
set -u

echo "[setup_profiler] gcc=$(gcc --version | head -1)"
echo "[setup_profiler] g++=$(g++ --version | head -1)"
echo "[setup_profiler] cmake=$(cmake --version | head -1)"
echo "[setup_profiler] python=$(python -V) at $(command -v python)"

python -m pip install --upgrade pip
python -m pip install \
  "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git@${PROFILER_SHA}"

python - <<'PY'
import importlib.metadata as md
import llama_cpp
print("adtc-profiler", md.version("adtc-profiler"))
print("llama-cpp-python", md.version("llama-cpp-python"))
print("llama_cpp", llama_cpp.__file__)
PY
command -v adtc-profiler
adtc-profiler --help | head -30

echo "[setup_profiler] Done."
