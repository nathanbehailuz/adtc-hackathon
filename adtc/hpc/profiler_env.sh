#!/usr/bin/env bash
# Put pinned linux llama.cpp binaries + conda adtc-profiler on PATH.
# Source after env.sh (or after conda activate).
# Usage: source "${HPC_DIR}/profiler_env.sh"
set -euo pipefail

_PROFILER_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ADTC_ROOT="$(cd "${_PROFILER_ENV_DIR}/.." && pwd)"
_LLAMA_BIN="${_ADTC_ROOT}/tools/llama.cpp/llama-b10451"

if [[ ! -x "${_LLAMA_BIN}/llama-bench" ]]; then
  echo "error: llama-bench not found at ${_LLAMA_BIN}/llama-bench" >&2
  echo "  Build on Jubail: sbatch setup_build_llama_cpp.sbatch" >&2
  exit 1
fi

export PATH="${_LLAMA_BIN}:${PATH}"

# Jubail-built binaries need gcc-12 libstdc++ (and local .so next to binaries).
_GCC_LIB=""
if [[ -d /share/apps/NYUAD5/gcc/12.2.0/lib64 ]]; then
  _GCC_LIB="/share/apps/NYUAD5/gcc/12.2.0/lib64"
fi
_LIBS="${_LLAMA_BIN}"
if [[ -n "${_GCC_LIB}" ]]; then
  _LIBS="${_LIBS}:${_GCC_LIB}"
fi
if [[ -d "${_ADTC_ROOT}/training/.conda-env/lib" ]]; then
  _LIBS="${_LIBS}:${_ADTC_ROOT}/training/.conda-env/lib"
fi
export LD_LIBRARY_PATH="${_LIBS}:${LD_LIBRARY_PATH:-}"

if ! command -v adtc-profiler >/dev/null 2>&1; then
  if [[ -x "${_ADTC_ROOT}/training/.conda-env/bin/adtc-profiler" ]]; then
    export PATH="${_ADTC_ROOT}/training/.conda-env/bin:${PATH}"
  fi
fi

echo "[profiler_env] llama-bench=$(command -v llama-bench)"
echo "[profiler_env] llama-quantize=$(command -v llama-quantize)"
echo "[profiler_env] adtc-profiler=$(command -v adtc-profiler || echo MISSING)"
