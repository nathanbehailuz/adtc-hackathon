#!/bin/bash
#SBATCH --job-name=adtc_build_llama
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Build llama.cpp b10451 from source (Jubail RHEL8 cannot run official ubuntu-x64
# binaries — they need GLIBC_2.34+). Installs into tools/llama.cpp/llama-b10451/.
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
ADTC_ROOT="$(cd "${HPC_DIR}/.." && pwd)"
SRC="${ADTC_ROOT}/tools/llama.cpp/src-b10451"
DEST="${ADTC_ROOT}/tools/llama.cpp/llama-b10451"
BUILD="${SRC}/build"

mkdir -p "${HPC_DIR}/logs" "${ADTC_ROOT}/tools/llama.cpp"

if command -v module >/dev/null 2>&1; then
  module purge 2>/dev/null || true
fi
module load gcc/12.2.0
module load cmake/3.31.2
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"

if [[ ! -f "${SRC}/CMakeLists.txt" ]]; then
  echo "[build_llama] Fetching llama.cpp b10451 source archive..."
  rm -rf "${SRC}"
  mkdir -p "${ADTC_ROOT}/tools/llama.cpp"
  ZIP="${ADTC_ROOT}/tools/llama.cpp/llama.cpp-b10451.zip"
  # Prefer zip over git clone (more reliable on Jubail)
  curl -L --fail --retry 5 --retry-delay 5 \
    -o "${ZIP}" \
    "https://github.com/ggml-org/llama.cpp/archive/refs/tags/b10451.zip"
  unzip -q "${ZIP}" -d "${ADTC_ROOT}/tools/llama.cpp"
  mv "${ADTC_ROOT}/tools/llama.cpp/llama.cpp-b10451" "${SRC}"
fi

cmake -S "${SRC}" -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF \
  -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF
# Note: llama-cli is gated behind LLAMA_BUILD_SERVER=ON in b10451.
# Profiler needs llama-bench; convert/quantize needs llama-quantize.
cmake --build "${BUILD}" -j "${SLURM_CPUS_PER_TASK:-8}" --target llama-bench llama-quantize

rm -rf "${DEST}"
mkdir -p "${DEST}"
for b in llama-bench llama-quantize; do
  if [[ -x "${BUILD}/bin/${b}" ]]; then
    cp -a "${BUILD}/bin/${b}" "${DEST}/"
  elif [[ -x "${BUILD}/${b}" ]]; then
    cp -a "${BUILD}/${b}" "${DEST}/"
  else
    echo "error: missing binary ${b}" >&2
    find "${BUILD}" -name "${b}" 2>/dev/null | head
    exit 1
  fi
done
# Shared libs next to binaries (RPATH / LD_LIBRARY_PATH)
find "${BUILD}/bin" -maxdepth 1 -name "*.so*" -exec cp -a {} "${DEST}/" \; 2>/dev/null || true
find "${BUILD}" -name "libggml*.so*" -exec cp -a {} "${DEST}/" \; 2>/dev/null || true
find "${BUILD}" -name "libllama*.so*" -exec cp -a {} "${DEST}/" \; 2>/dev/null || true

# Prefer linking against compiler libstdc++ from the build toolchain
GCC_LIB="$(dirname "$(${CXX} -print-file-name=libstdc++.so)")"
export PATH="${DEST}:${PATH}"
export LD_LIBRARY_PATH="${DEST}:${GCC_LIB}:${LD_LIBRARY_PATH:-}"
echo "[build_llama] DEST=${DEST}"
ls -la "${DEST}" | head -40
ldd "${DEST}/llama-bench" | head -20 || true
"${DEST}/llama-bench" -h | head -8 || true
"${DEST}/llama-quantize" -h | head -5 || true
echo "[build_llama] Done."
