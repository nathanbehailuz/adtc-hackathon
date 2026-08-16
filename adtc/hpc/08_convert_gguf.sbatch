#!/bin/bash
#SBATCH --job-name=adtc_convert_gguf
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Convert merged HF → high-precision GGUF, then post-training quantize Q8/Q6/Q5/Q4.
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
# shellcheck disable=SC1091
source "${HPC_DIR}/profiler_env.sh"
cd "${ADTC_ROOT}"
export ADTC_ROOT

LLAMA_SRC="${ADTC_ROOT}/tools/llama.cpp/src-b10451"
CONVERT="${LLAMA_SRC}/convert_hf_to_gguf.py"
OUT_DIR="${ADTC_ROOT}/artifacts/gguf/adapted"
mkdir -p "${OUT_DIR}"

if [[ ! -f "${CONVERT}" ]]; then
  echo "[convert] Fetching llama.cpp b10451 source for convert script..."
  rm -rf "${LLAMA_SRC}"
  ZIP="${ADTC_ROOT}/tools/llama.cpp/llama.cpp-b10451.zip"
  mkdir -p "${ADTC_ROOT}/tools/llama.cpp"
  if [[ ! -f "${ZIP}" ]]; then
    curl -L --fail --retry 5 --retry-delay 5 \
      -o "${ZIP}" \
      "https://github.com/ggml-org/llama.cpp/archive/refs/tags/b10451.zip"
  fi
  unzip -q "${ZIP}" -d "${ADTC_ROOT}/tools/llama.cpp"
  mv "${ADTC_ROOT}/tools/llama.cpp/llama.cpp-b10451" "${LLAMA_SRC}"
fi

# Ensure gguf python deps
python -m pip install -q 'gguf>=0.10' sentencepiece protobuf || true

# Priority: 1.7B + 4B first, then Gemma + Qwen2.5
MODELS=(
  "qwen3_1_7b_merged_v0"
  "qwen3_4b_merged_v0"
  "gemma3_4b_merged_v0"
  "qwen25_3b_instruct_merged_v0"
)

QUANTS=(Q8_0 Q6_K Q5_K_M Q4_K_M)
FAILED=0

for name in "${MODELS[@]}"; do
  HF_DIR="${ADTC_ROOT}/training/runs/${name}"
  if [[ ! -d "${HF_DIR}" ]]; then
    echo "[convert] SKIP missing ${HF_DIR}" >&2
    FAILED=$((FAILED + 1))
    continue
  fi
  F16="${OUT_DIR}/${name}-f16.gguf"
  if [[ ! -f "${F16}" ]]; then
    echo "[convert] HF→F16 ${name}"
    if ! python "${CONVERT}" "${HF_DIR}" --outfile "${F16}" --outtype f16; then
      echo "[convert] FAIL convert ${name}" >&2
      FAILED=$((FAILED + 1))
      continue
    fi
  else
    echo "[convert] reuse ${F16}"
  fi
  for q in "${QUANTS[@]}"; do
    QOUT="${OUT_DIR}/${name}-${q}.gguf"
    if [[ -f "${QOUT}" ]]; then
      echo "[quantize] reuse ${QOUT}"
      continue
    fi
    echo "[quantize] ${name} → ${q}"
    if ! llama-quantize "${F16}" "${QOUT}" "${q}"; then
      echo "[quantize] FAIL ${name} ${q}" >&2
      FAILED=$((FAILED + 1))
    fi
  done
done

# Manifest
python - <<'PY'
import json
from pathlib import Path
import os
out = Path(os.environ["ADTC_ROOT"]) / "artifacts/gguf/adapted"
files = sorted(p.name for p in out.glob("*.gguf"))
(out / "manifest_v0.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
print("files", len(files))
PY

echo "[convert] failed=${FAILED}"
exit "${FAILED}"
