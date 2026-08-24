#!/usr/bin/env bash
# Convert merged HF → GGUF quants for v5. Env: V5_ALIAS
#SBATCH --job-name=adtc_gguf_v5
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
# shellcheck disable=SC1091
source "${HPC_DIR}/profiler_env.sh" || true
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate "${ADTC_ROOT}/training/.conda-env"
set -u
cd "${ADTC_ROOT}"

ALIAS="${V5_ALIAS:?}"
HF_DIR="${ADTC_ROOT}/training/runs/${ALIAS}_merged_v5"
[[ -d "${HF_DIR}" ]] || { echo "missing ${HF_DIR}" >&2; exit 1; }

LLAMA_SRC="${ADTC_ROOT}/tools/llama.cpp/src-b10451"
CONVERT="${LLAMA_SRC}/convert_hf_to_gguf.py"
OUT_DIR="${ADTC_ROOT}/artifacts/gguf/adapted"
mkdir -p "${OUT_DIR}"
[[ -f "${CONVERT}" ]] || { echo "missing ${CONVERT}" >&2; exit 1; }
python -m pip install -q 'gguf>=0.10' sentencepiece protobuf || true

NAME="${ALIAS}_merged_v5"
F16="${OUT_DIR}/${NAME}-f16.gguf"
[[ -f "${F16}" ]] || python "${CONVERT}" "${HF_DIR}" --outfile "${F16}" --outtype f16

FAILED=0
for q in Q8_0 Q6_K Q5_K_M Q4_K_M; do
  QOUT="${OUT_DIR}/${NAME}-${q}.gguf"
  if [[ -f "${QOUT}" ]]; then
    echo "reuse ${QOUT}"
    continue
  fi
  llama-quantize "${F16}" "${QOUT}" "${q}" || FAILED=$((FAILED + 1))
done

python - <<PY
import json, hashlib
from pathlib import Path
from datetime import datetime, timezone
alias="${ALIAS}"
out_dir=Path("${OUT_DIR}")
name="${NAME}"
files={}
for p in sorted(out_dir.glob(f"{name}-*.gguf")):
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    files[p.name]={"bytes": p.stat().st_size, "sha256": h}
manifest={
  "alias": alias,
  "hf_dir": "${HF_DIR}",
  "llama_cpp_src": "${LLAMA_SRC}",
  "files": files,
  "updated": datetime.now(timezone.utc).isoformat(),
}
path=out_dir/f"manifest_{alias}_v5.json"
path.write_text(json.dumps(manifest, indent=2)+"\n")
print("wrote", path)
PY

[[ ${FAILED} -eq 0 ]] || { echo "quantize failures=${FAILED}" >&2; exit 1; }
echo "[gguf_v5] OK ${NAME}"
