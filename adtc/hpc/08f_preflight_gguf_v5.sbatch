#!/usr/bin/env bash
#SBATCH --job-name=adtc_preflight_gguf_v5
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Convert HF AfriqueQwen candidate → f16 GGUF → Q4_K_M and smoke-test text gen.
# Env: V5_ALIAS=qwen35_extcm|qwen35_afrique|qwen3_afrique
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

export ALIAS="${V5_ALIAS:?set V5_ALIAS}"
HF_ID="$(python - <<'PY'
import os
import yaml
from pathlib import Path

m = yaml.safe_load(Path("training/configs/v5_models.yaml").read_text())["models"][os.environ["ALIAS"]]
print(m["hf_id"])
PY
)"
echo "[gguf_preflight] alias=${ALIAS} hf=${HF_ID}"

LLAMA_SRC="${ADTC_ROOT}/tools/llama.cpp/src-b10451"
CONVERT="${LLAMA_SRC}/convert_hf_to_gguf.py"
OUT_DIR="${ADTC_ROOT}/artifacts/gguf/adapted"
mkdir -p "${OUT_DIR}" docs/artifacts/v5
[[ -f "${CONVERT}" ]] || { echo "missing ${CONVERT}" >&2; exit 1; }
python -m pip install -q 'gguf>=0.10' sentencepiece protobuf || true

# Resolve local snapshot that actually contains weight shards (not tokenizer-only stubs).
LOCAL="$(python - <<PY
from pathlib import Path
from huggingface_hub import snapshot_download

hf_id = "${HF_ID}"

def has_weights(root: Path) -> bool:
    return any(root.glob("*.safetensors")) or any(root.glob("pytorch_model*.bin"))

local = None
try:
    local = Path(snapshot_download(hf_id, local_files_only=True))
except Exception:
    local = None
if local is None or not has_weights(local):
    local = Path(snapshot_download(hf_id))
if not has_weights(local):
    raise SystemExit(f"no weight files under snapshot: {local}")
print(local)
PY
)"
echo "[gguf_preflight] local=${LOCAL}"

NAME="${ALIAS}_base_v5"
F16="${OUT_DIR}/${NAME}-f16.gguf"
Q4="${OUT_DIR}/${NAME}-Q4_K_M.gguf"
Q5="${OUT_DIR}/${NAME}-Q5_K_M.gguf"

set +e
python "${CONVERT}" "${LOCAL}" --outfile "${F16}" --outtype f16
CONVERT_RC=$?
set -e
if [[ ${CONVERT_RC} -ne 0 ]]; then
  python - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone
p=Path("docs/artifacts/v5/gguf_preflight_${ALIAS}.json")
p.write_text(json.dumps({
  "alias": "${ALIAS}",
  "hf_id": "${HF_ID}",
  "status": "convert_failed",
  "llama_cpp_src": "${LLAMA_SRC}",
  "updated": datetime.now(timezone.utc).isoformat(),
}, indent=2)+"\n")
print("wrote", p)
PY
  exit 1
fi

llama-quantize "${F16}" "${Q4}" Q4_K_M
llama-quantize "${F16}" "${Q5}" Q5_K_M || true

# Smoke prompts via llama-cli if available
BIN="$(command -v llama-cli || command -v llama-cli || true)"
if [[ -z "${BIN}" ]]; then
  BIN="$(ls "${ADTC_ROOT}/tools/llama.cpp"/bin*/llama-cli 2>/dev/null | head -1 || true)"
fi

python - <<PY
import json, os, subprocess, time, resource
from pathlib import Path
from datetime import datetime, timezone
alias=os.environ["ALIAS"]
q4=Path("${Q4}")
q5=Path("${Q5}")
bin_path="${BIN}"
result={
  "alias": alias,
  "hf_id": "${HF_ID}",
  "status": "ok",
  "llama_cpp_src": "${LLAMA_SRC}",
  "f16": str(Path("${F16}")),
  "q4": str(q4),
  "q5": str(q5) if q5.exists() else None,
  "q4_bytes": q4.stat().st_size if q4.exists() else None,
  "q5_bytes": q5.stat().st_size if q5.exists() else None,
  "single_gguf_text": True,
  "mmproj_required_for_text": False,
  "prompts": {},
  "updated": datetime.now(timezone.utc).isoformat(),
}
prompts={
  "en": "Hello. Explain what photosynthesis is.",
  "am": "አንድ ተማሪ 12 እርሳሶች አሉት። 5 ለጓደኞቹ ከሰጠ፣ ስንት ይቀሩታል?",
}
if not bin_path:
  result["status"]="quantize_ok_no_llama_cli"
else:
  for k,p in prompts.items():
    cmd=[bin_path,"-m",str(q4),"-p",p,"-n","48","--temp","0","-ngl","0"]
    t0=time.time()
    try:
      proc=subprocess.run(cmd, capture_output=True, text=True, timeout=180)
      out=(proc.stdout or "")[-500:]
      result["prompts"][k]={"rc":proc.returncode,"secs":round(time.time()-t0,2),"tail":out}
      if proc.returncode!=0:
        result["status"]="smoke_fail"
    except Exception as e:
      result["prompts"][k]={"error":str(e)}
      result["status"]="smoke_fail"
  # peak RSS of this python process is not the llama RSS; record file sizes for gate.
result["peak_rss_note"]="Measure llama-cli RSS on laptop; Jubail compute RSS is secondary."
outp=Path(f"docs/artifacts/v5/gguf_preflight_{alias}.json")
outp.write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
print("wrote", outp)
PY

echo "[gguf_preflight] done ${ALIAS}"
