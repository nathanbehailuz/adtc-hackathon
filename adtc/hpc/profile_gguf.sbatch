#!/bin/bash
#SBATCH --job-name=adtc_prof_v6
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Profile qwen3_1_7b_merged_v6 Q4/Q5/Q6. Write Gate 5 pick for v6.
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
# shellcheck disable=SC1091
source "${HPC_DIR}/profiler_env.sh"
cd "${ADTC_ROOT}"
export ADTC_ROOT
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export ACC_LIMIT="${ACC_LIMIT:-50}"

mkdir -p docs/artifacts/v6 artifacts/profiler_stage/v6

python - <<'PY'
import json
import os
import subprocess
from pathlib import Path

root = Path(os.environ["ADTC_ROOT"])
gguf_dir = root / "artifacts/gguf/adapted"
out_dir = root / "docs/artifacts/v6"
stage_root = root / "artifacts/profiler_stage/v6"
acc_limit = os.environ.get("ACC_LIMIT", "50")
keys = [
    "qwen3_1_7b_merged_v6-Q4_K_M",
    "tebeb_tutor_1.7b-Q5_K_M",
    "qwen3_1_7b_merged_v6-Q6_K",
]
summary = []
failed = 0
for key in keys:
    gguf = gguf_dir / f"{key}.gguf"
    if not gguf.is_file():
        summary.append({"key": key, "profile_ok": False, "error": f"missing {gguf}"})
        failed += 1
        continue
    stage = stage_root / key
    subprocess.check_call(
        [
            "python",
            "eval/stage_gguf_submission.py",
            "--gguf",
            str(gguf),
            "--out-dir",
            str(stage),
            "--name",
            key,
        ]
    )
    out_json = out_dir / f"phase5_profile_{key}.json"
    cmd = [
        "adtc-profiler",
        "run",
        "--submission",
        str(stage),
        "--mode",
        "participant",
        "--accuracy-task",
        "arc_easy",
        "--accuracy-limit",
        str(acc_limit),
        "--output",
        str(out_json),
    ]
    print("RUN", " ".join(cmd), flush=True)
    try:
        subprocess.check_call(cmd)
        data = json.loads(out_json.read_text())
        tps = data.get("throughput", {}).get("tokens_per_second_generation")
        rss = data.get("memory", {}).get("peak_rss_mb")
        acc = data.get("accuracy") or {}
        arc = None
        if isinstance(acc, dict):
            arc = acc.get("acc_norm") or acc.get("acc") or acc.get("accuracy")
        s_acc = 100.0 * float(arc) if arc is not None else 0.0
        s_tps = 100.0 * min(1.0, float(tps or 0) / 15.0)
        peak_gb = float(rss or 0) / 1024.0
        s_mem = max(0.0, min(100.0, 100.0 * (7.0 - peak_gb) / 7.0))
        composite = 0.5 * s_acc + 0.3 * s_tps + 0.2 * s_mem
        summary.append(
            {
                "key": key,
                "gguf": str(gguf),
                "profile_ok": True,
                "tps": tps,
                "peak_rss_mb": rss,
                "steady_rss_mb": data.get("memory", {}).get("steady_state_rss_mb"),
                "throttled": data.get("cpu_thermal", {}).get("throttled"),
                "arc_easy": arc,
                "S_acc": round(s_acc, 2),
                "S_tps": round(s_tps, 2),
                "S_mem": round(s_mem, 2),
                "composite": round(composite, 2),
                "artifact": str(out_json),
            }
        )
    except Exception as e:  # noqa: BLE001
        summary.append({"key": key, "gguf": str(gguf), "profile_ok": False, "error": str(e)})
        failed += 1

ok = [s for s in summary if s.get("profile_ok") and (s.get("peak_rss_mb") or 1e9) <= 7000]
ok_sorted = sorted(ok, key=lambda s: s.get("composite") or 0, reverse=True)
winner = ok_sorted[0] if ok_sorted else None

(out_dir / "phase5_profile_summary_v6.json").write_text(json.dumps(summary, indent=2) + "\n")
if winner:
    (out_dir / "phase5_gate5_winner_v6.json").write_text(json.dumps(winner, indent=2) + "\n")

md = [
    "# Phase 5 — v6 profiler Pareto (Qwen3-1.7B EN)",
    "",
    "| key | TPS | peak RSS MB | ARC | composite | ok |",
    "|-----|-----|-------------|-----|-----------|----|",
]
for s in summary:
    md.append(
        f"| `{s.get('key')}` | {s.get('tps')} | {s.get('peak_rss_mb')} | {s.get('arc_easy')} | {s.get('composite')} | {s.get('profile_ok')} |"
    )
md.append("")
if winner:
    md.append(f"**Gate 5 winner (v6):** `{winner['key']}` composite={winner.get('composite')}")
    md.append(f"- GGUF: `{winner['gguf']}`")
    md.append(f"- TPS={winner.get('tps')} peak_rss_mb={winner.get('peak_rss_mb')}")
(out_dir / "phase5_pareto_v6.md").write_text("\n".join(md) + "\n")
print(json.dumps({"winner": winner, "n_ok": len(ok), "failed": failed}, indent=2))
raise SystemExit(failed)
PY

echo "[prof_v6] Done"
