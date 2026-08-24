#!/bin/bash
#SBATCH --job-name=adtc_convert_gguf_v4
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

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
[[ -f "${CONVERT}" ]] || { echo "missing ${CONVERT}" >&2; exit 1; }
python -m pip install -q 'gguf>=0.10' sentencepiece protobuf || true

name="gemma3_4b_merged_v4"
HF_DIR="${ADTC_ROOT}/training/runs/${name}"
[[ -d "${HF_DIR}" ]] || { echo "missing ${HF_DIR}" >&2; exit 1; }
F16="${OUT_DIR}/${name}-f16.gguf"
[[ -f "${F16}" ]] || python "${CONVERT}" "${HF_DIR}" --outfile "${F16}" --outtype f16
FAILED=0
for q in Q8_0 Q6_K Q5_K_M Q4_K_M; do
  QOUT="${OUT_DIR}/${name}-${q}.gguf"
  if [[ -f "${QOUT}" ]]; then
    echo "reuse ${QOUT}"
    continue
  fi
  llama-quantize "${F16}" "${QOUT}" "${q}" || FAILED=$((FAILED + 1))
done
python - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["ADTC_ROOT"]) / "artifacts/gguf/adapted"
files = sorted(p.name for p in out.glob("*_merged_v4*.gguf"))
(out / "manifest_v4.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
print("v4 files", len(files))
PY
exit "${FAILED}"
