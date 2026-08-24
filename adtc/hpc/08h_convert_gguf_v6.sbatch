#!/bin/bash
#SBATCH --job-name=adtc_gguf_v6
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Convert qwen3_1_7b_merged_v6 → f16 + Q8/Q6/Q5/Q4.
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
NAME="qwen3_1_7b_merged_v6"
HF_DIR="${ADTC_ROOT}/training/runs/${NAME}"
mkdir -p "${OUT_DIR}" docs/artifacts/v6

[[ -d "${HF_DIR}" ]] || { echo "missing ${HF_DIR}" >&2; exit 1; }
[[ -f "${CONVERT}" ]] || { echo "missing ${CONVERT}" >&2; exit 1; }
python -m pip install -q 'gguf>=0.10' sentencepiece protobuf || true

F16="${OUT_DIR}/${NAME}-f16.gguf"
if [[ ! -f "${F16}" ]]; then
  echo "[gguf_v6] HF→F16 ${NAME}"
  python "${CONVERT}" "${HF_DIR}" --outfile "${F16}" --outtype f16
else
  echo "[gguf_v6] reuse ${F16}"
fi

FAILED=0
for q in Q8_0 Q6_K Q5_K_M Q4_K_M; do
  QOUT="${OUT_DIR}/${NAME}-${q}.gguf"
  if [[ -f "${QOUT}" ]]; then
    echo "[gguf_v6] reuse ${QOUT}"
    continue
  fi
  echo "[gguf_v6] quantize ${q}"
  llama-quantize "${F16}" "${QOUT}" "${q}" || FAILED=$((FAILED + 1))
done

python - <<PY
import json, hashlib
from pathlib import Path
from datetime import datetime, timezone
out_dir = Path("${OUT_DIR}")
name = "${NAME}"
files = {}
for p in sorted(out_dir.glob(f"{name}-*.gguf")):
    files[p.name] = {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
manifest = {
  "name": name,
  "files": files,
  "updated": datetime.now(timezone.utc).isoformat(),
}
path = Path("docs/artifacts/v6/gguf_manifest.json")
path.write_text(json.dumps(manifest, indent=2) + "\n")
print("wrote", path, "n=", len(files))
PY

echo "[gguf_v6] failed=${FAILED}"
exit "${FAILED}"
