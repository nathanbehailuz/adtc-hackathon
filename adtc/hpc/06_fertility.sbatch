#!/bin/bash
#SBATCH --job-name=adtc_fertility
#SBATCH -p compute
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Tokenizer fertility for Phase 2 bases (local HF snapshots preferred).
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
cd "${ADTC_ROOT}"

OUT_DIR="${ADTC_ROOT}/docs/artifacts"
mkdir -p "${OUT_DIR}"
COMBINED="${OUT_DIR}/fertility_v0.json"

MODELS=(
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-4B"
  "Qwen/Qwen2.5-3B-Instruct"
  "google/gemma-3-4b-it"
)

TMP_DIR="${OUT_DIR}/fertility_parts"
mkdir -p "${TMP_DIR}"
: > "${TMP_DIR}/list.txt"

for mid in "${MODELS[@]}"; do
  safe="$(echo "${mid}" | tr '/.' '__')"
  per="${TMP_DIR}/fertility_${safe}.json"
  echo "======== fertility ${mid} ========"
  if python eval/fertility.py --tokenizer "${mid}" --out "${per}"; then
    echo "${per}" >> "${TMP_DIR}/list.txt"
  else
    echo "{\"tokenizer\": \"${mid}\", \"error\": \"fertility_failed\", \"fragmentation_flag\": \"unknown\"}" > "${per}"
    echo "${per}" >> "${TMP_DIR}/list.txt"
  fi
done

python - <<PY
import json
from pathlib import Path
parts = Path("${TMP_DIR}/list.txt").read_text().splitlines()
reports = []
for p in parts:
    r = json.loads(Path(p).read_text(encoding="utf-8"))
    if "error" not in r:
        r_am = float(r.get("mean_R_am_en", 0))
        f_am = float(r.get("mean_F_am", 0))
        r["fragmentation_flag"] = "severe" if (r_am >= 2.5 or f_am >= 4.0) else "ok"
        r["pairs"] = r.get("pairs", [])[:3]
    reports.append(r)
out = Path("${COMBINED}")
out.write_text(json.dumps({"models": reports}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {out}")
for r in reports:
    print(r.get("tokenizer"), r.get("mean_F_am"), r.get("mean_R_am_en"), r.get("fragmentation_flag"), r.get("error"))
PY

echo "[fertility] Done -> ${COMBINED}"
