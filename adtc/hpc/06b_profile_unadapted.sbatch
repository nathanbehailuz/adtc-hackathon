#!/bin/bash
#SBATCH --job-name=adtc_prof_unadapted
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Profile unadapted GGUFs with pinned adtc-profiler (participant, skip-accuracy for speed;
# accuracy path available via ACCURACY=1).
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
# shellcheck disable=SC1091
source "${HPC_DIR}/profiler_env.sh"
cd "${ADTC_ROOT}"
export ADTC_ROOT

MANIFEST="${ADTC_ROOT}/artifacts/gguf/unadapted/download_manifest_v0.json"
OUT_DIR="${ADTC_ROOT}/docs/artifacts"
STAGE_ROOT="${ADTC_ROOT}/artifacts/profiler_stage/unadapted"
mkdir -p "${OUT_DIR}" "${STAGE_ROOT}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "error: missing ${MANIFEST}; run 06a_download_unadapted_gguf.sbatch first" >&2
  exit 1
fi

python - <<'PY'
import json, subprocess, os
from pathlib import Path

root = Path(os.environ["ADTC_ROOT"])
manifest = json.loads((root / "artifacts/gguf/unadapted/download_manifest_v0.json").read_text())
out_dir = root / "docs/artifacts"
stage_root = root / "artifacts/profiler_stage/unadapted"
summary = []
accuracy = os.environ.get("ACCURACY", "0") == "1"

for row in manifest:
    if not row.get("ok") or not row.get("path"):
        summary.append({**row, "profile_ok": False, "error": row.get("error") or "missing"})
        continue
    key = row["key"]
    gguf = Path(row["path"])
    stage = stage_root / key
    subprocess.check_call([
        "python", "eval/stage_gguf_submission.py",
        "--gguf", str(gguf),
        "--out-dir", str(stage),
        "--name", key,
        "--quant", row.get("quant", "GGUF"),
        "--params", row.get("params", "unknown"),
    ])
    out_json = out_dir / f"phase2_profile_{key}.json"
    cmd = [
        "adtc-profiler", "run",
        "--submission", str(stage),
        "--mode", "participant",
        "--output", str(out_json),
    ]
    if not accuracy:
        cmd.append("--skip-accuracy")
    # Compat-only (Qwen3.5): hard 1h wall already via Slurm; catch failures
    print("RUN", " ".join(cmd), flush=True)
    try:
        subprocess.check_call(cmd)
        data = json.loads(out_json.read_text())
        summary.append({
            "key": key,
            "profile_ok": True,
            "compat_only": bool(row.get("compat_only")),
            "tps": data.get("throughput", {}).get("tokens_per_second_generation"),
            "peak_rss_mb": data.get("memory", {}).get("peak_rss_mb"),
            "steady_rss_mb": data.get("memory", {}).get("steady_state_rss_mb"),
            "throttled": data.get("cpu_thermal", {}).get("throttled"),
            "artifact": str(out_json),
        })
    except Exception as e:
        summary.append({
            "key": key,
            "profile_ok": False,
            "compat_only": bool(row.get("compat_only")),
            "error": str(e),
        })
        if row.get("compat_only"):
            print(f"[compat] Qwen3.5-style drop candidate: {key}: {e}")

(out_dir / "phase2_unadapted_profile_summary_v0.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2))
PY

echo "[prof_unadapted] Done"
