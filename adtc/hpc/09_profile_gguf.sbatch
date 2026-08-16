#!/bin/bash
#SBATCH --job-name=adtc_prof_adapted
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Profile quantized adapted GGUFs; write Pareto summary for Gate 5.
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
# shellcheck disable=SC1091
source "${HPC_DIR}/profiler_env.sh"
cd "${ADTC_ROOT}"
export ADTC_ROOT

GGUF_DIR="${ADTC_ROOT}/artifacts/gguf/adapted"
OUT_DIR="${ADTC_ROOT}/docs/artifacts"
STAGE_ROOT="${ADTC_ROOT}/artifacts/profiler_stage/adapted"
mkdir -p "${OUT_DIR}" "${STAGE_ROOT}"

# Prefer Q4/Q6 for Pareto; include Q5/Q8 if present
python - <<'PY'
import json, subprocess, os, re
from pathlib import Path

root = Path(os.environ["ADTC_ROOT"])
gguf_dir = root / "artifacts/gguf/adapted"
out_dir = root / "docs/artifacts"
stage_root = root / "artifacts/profiler_stage/adapted"
accuracy = os.environ.get("ACCURACY", "0") == "1"

# Prefer smaller quants for laptop-like screen; still profile all non-f16
cands = sorted(
    [p for p in gguf_dir.glob("*.gguf") if "-f16.gguf" not in p.name],
    key=lambda p: (0 if "Q4_K_M" in p.name or "Q6_K" in p.name else 1, p.name),
)
# Cap if huge: prefer Q4/Q6 of 1.7B and 4B first already via sort
summary = []
for gguf in cands:
    key = gguf.stem
    stage = stage_root / key
    subprocess.check_call([
        "python", "eval/stage_gguf_submission.py",
        "--gguf", str(gguf),
        "--out-dir", str(stage),
        "--name", key,
    ])
    out_json = out_dir / f"phase5_profile_{key}.json"
    cmd = [
        "adtc-profiler", "run",
        "--submission", str(stage),
        "--mode", "participant",
        "--output", str(out_json),
    ]
    if not accuracy:
        cmd.append("--skip-accuracy")
    print("RUN", " ".join(cmd), flush=True)
    try:
        subprocess.check_call(cmd)
        data = json.loads(out_json.read_text())
        # Sustained thermal already in report; note throttled
        summary.append({
            "key": key,
            "gguf": str(gguf),
            "profile_ok": True,
            "tps": data.get("throughput", {}).get("tokens_per_second_generation"),
            "peak_rss_mb": data.get("memory", {}).get("peak_rss_mb"),
            "steady_rss_mb": data.get("memory", {}).get("steady_state_rss_mb"),
            "throttled": data.get("cpu_thermal", {}).get("throttled"),
            "accuracy": data.get("accuracy"),
            "artifact": str(out_json),
        })
    except Exception as e:
        summary.append({"key": key, "gguf": str(gguf), "profile_ok": False, "error": str(e)})

# Pareto: among profile_ok, prefer higher tps and lower rss; accuracy empty if skipped
# Pick Gate 5: best Q4/Q6 under ~7GB RSS with highest TPS among 1.7B/4B
ok = [s for s in summary if s.get("profile_ok")]
def score(s):
    tps = s.get("tps") or 0
    rss = s.get("peak_rss_mb") or 1e9
    # Prefer staying under ~7000 MB peak
    pen = 0 if rss <= 7000 else (rss - 7000) / 1000
    return tps - pen

ok_sorted = sorted(ok, key=score, reverse=True)
winner = ok_sorted[0] if ok_sorted else None

(out_dir / "phase5_profile_summary_v0.json").write_text(json.dumps(summary, indent=2) + "\n")

md = ["# Phase 5 — GGUF profiler Pareto", "",
      "| key | TPS | peak RSS MB | steady RSS | throttled | ok |",
      "|-----|-----|-------------|------------|-----------|----|"]
for s in summary:
    md.append(
        f"| `{s.get('key')}` | {s.get('tps')} | {s.get('peak_rss_mb')} | {s.get('steady_rss_mb')} | {s.get('throttled')} | {s.get('profile_ok')} |"
    )
md.append("")
if winner:
    md.append(f"**Gate 5 winner (auto heuristic):** `{winner['key']}`")
    md.append(f"- GGUF: `{winner['gguf']}`")
    md.append(f"- TPS={winner.get('tps')} peak_rss_mb={winner.get('peak_rss_mb')}")
    (out_dir / "phase5_gate5_winner.json").write_text(json.dumps(winner, indent=2) + "\n")
(out_dir / "phase5_pareto_v0.md").write_text("\n".join(md) + "\n")
print(json.dumps({"winner": winner, "n_ok": len(ok)}, indent=2))
PY

echo "[prof_adapted] Done"
