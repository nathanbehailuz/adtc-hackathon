#!/usr/bin/env bash
# One-time: llama.cpp b10451 source + linux binaries on AGH / Shadeform.
# convert_gguf.sh needs convert_hf_to_gguf.py (source) and llama-quantize (binary).
#
# Usage: cd adtc/agh && bash setup_llama_cpp.sh
set -euo pipefail
AGH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${AGH_DIR}/env.sh"
mkdir -p "${AGH_DIR}/logs"
LOG="${AGH_DIR}/logs/setup_llama_cpp.log"
exec > >(tee -a "${LOG}") 2>&1

PIN="b10451"
TOOLS="${ADTC_ROOT}/tools/llama.cpp"
SRC="${TOOLS}/src-${PIN}"
DEST="${TOOLS}/llama-${PIN}"
ZIP="${TOOLS}/llama.cpp-${PIN}.zip"
TARBALL="${TOOLS}/llama-${PIN}-bin-ubuntu-x64.tar.gz"
SRC_URL="https://github.com/ggml-org/llama.cpp/archive/refs/tags/${PIN}.zip"
BIN_URL="https://github.com/ggml-org/llama.cpp/releases/download/${PIN}/llama-${PIN}-bin-ubuntu-x64.tar.gz"

mkdir -p "${TOOLS}"

extract_zip() {
  local zip_path="$1"
  local dest_dir="$2"
  if command -v unzip >/dev/null 2>&1; then
    unzip -q "${zip_path}" -d "${dest_dir}"
  else
    python - "${zip_path}" "${dest_dir}" <<'PY'
import sys, zipfile
from pathlib import Path
zf, dest = sys.argv[1], Path(sys.argv[2])
with zipfile.ZipFile(zf) as z:
    z.extractall(dest)
PY
  fi
}

copy_bins_from() {
  local root="$1"
  mkdir -p "${DEST}"
  local found=0
  local b f bindir
  for b in llama-quantize llama-bench; do
    f="$(find "${root}" -type f -name "${b}" 2>/dev/null | head -n 1 || true)"
    if [[ -n "${f}" ]]; then
      cp -a "${f}" "${DEST}/${b}"
      chmod +x "${DEST}/${b}"
      bindir="$(dirname "${f}")"
      find "${bindir}" -maxdepth 1 -name "*.so*" -exec cp -a {} "${DEST}/" \; 2>/dev/null || true
      found=$((found + 1))
    fi
  done
  find "${root}" \( -name "libggml*.so*" -o -name "libllama*.so*" \) \
    -exec cp -a {} "${DEST}/" \; 2>/dev/null || true
  [[ "${found}" -ge 2 ]]
}

bins_run() {
  [[ -x "${DEST}/llama-quantize" ]] || return 1
  export PATH="${DEST}:${PATH}"
  export LD_LIBRARY_PATH="${DEST}:${LD_LIBRARY_PATH:-}"
  local err
  err="$("${DEST}/llama-quantize" 2>&1 >/dev/null || true)"
  if echo "${err}" | grep -qiE 'GLIBC|error while loading|cannot open shared object'; then
    return 1
  fi
  return 0
}

ensure_build_deps() {
  if command -v cmake >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
    return 0
  fi
  echo "[setup_llama] cmake/g++ missing — installing build packages"
  if [[ "$(id -u)" -eq 0 ]]; then
    apt-get update -y
    apt-get install -y cmake g++ build-essential unzip
  elif command -v sudo >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y cmake g++ build-essential unzip
  else
    echo "error: need cmake + g++ (or sudo) to build llama.cpp" >&2
    exit 1
  fi
}

if [[ ! -f "${SRC}/convert_hf_to_gguf.py" ]]; then
  echo "[setup_llama] Fetching llama.cpp ${PIN} source"
  rm -rf "${SRC}"
  curl -L --fail --retry 5 --retry-delay 5 -o "${ZIP}" "${SRC_URL}"
  extract_zip "${ZIP}" "${TOOLS}"
  if [[ -d "${TOOLS}/llama.cpp-${PIN}" ]]; then
    mv "${TOOLS}/llama.cpp-${PIN}" "${SRC}"
  else
    echo "error: unexpected zip layout under ${TOOLS}" >&2
    ls -la "${TOOLS}" >&2
    exit 1
  fi
else
  echo "[setup_llama] Reusing source ${SRC}"
fi

need_bins=1
if bins_run; then
  echo "[setup_llama] Reusing binaries ${DEST}"
  need_bins=0
fi

if [[ "${need_bins}" -eq 1 ]]; then
  echo "[setup_llama] Fetching ubuntu-x64 binaries ${PIN}"
  if curl -L --fail --retry 5 --retry-delay 5 -o "${TARBALL}" "${BIN_URL}"; then
    TMP="$(mktemp -d)"
    tar -xzf "${TARBALL}" -C "${TMP}"
    if copy_bins_from "${TMP}" && bins_run; then
      echo "[setup_llama] Official ubuntu-x64 binaries OK"
      need_bins=0
    else
      echo "[setup_llama] Official binaries unusable (likely GLIBC) — will build"
    fi
    rm -rf "${TMP}"
  else
    echo "[setup_llama] Binary tarball download failed — will build"
    rm -f "${TARBALL}"
  fi
fi

if [[ "${need_bins}" -eq 1 ]]; then
  ensure_build_deps
  BUILD="${SRC}/build"
  echo "[setup_llama] cmake build ${PIN}"
  cmake -S "${SRC}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF \
    -DGGML_CUDA=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF
  cmake --build "${BUILD}" -j "$(nproc 2>/dev/null || echo 8)" --target llama-bench llama-quantize
  copy_bins_from "${BUILD}" || {
    echo "error: cmake build did not produce llama-quantize / llama-bench" >&2
    find "${BUILD}" -name "llama-quantize" -o -name "llama-bench" >&2
    exit 1
  }
  bins_run || {
    echo "error: built binaries failed to run" >&2
    ldd "${DEST}/llama-quantize" || true
    exit 1
  }
fi

export PATH="${DEST}:${PATH}"
export LD_LIBRARY_PATH="${DEST}:${LD_LIBRARY_PATH:-}"
echo "[setup_llama] SRC=${SRC}"
echo "[setup_llama] DEST=${DEST}"
ls -la "${DEST}" | head -20
echo "[setup_llama] convert=$(test -f "${SRC}/convert_hf_to_gguf.py" && echo OK || echo MISSING)"
echo "[setup_llama] llama-quantize=$(command -v llama-quantize)"
echo "[setup_llama] llama-bench=$(command -v llama-bench)"
echo "[setup_llama] Done. Re-source env.sh (or re-run convert_gguf.sh)."
