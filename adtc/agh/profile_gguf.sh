#!/usr/bin/env bash
# Profile qwen3_1_7b_merged_v7 Q4/Q5/Q6. Write Gate 5 pick for v7.
set -euo pipefail
AGH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${AGH_DIR}/env.sh"
mkdir -p "${AGH_DIR}/logs"
LOG="${AGH_DIR}/logs/profile_gguf.log"
exec > >(tee -a "${LOG}") 2>&1

cd "${ADTC_ROOT}"
export ADTC_ROOT
export ACC_LIMIT="${ACC_LIMIT:-50}"
# Profiler shells out to llama-bench; those shared libs must be findable.
LLAMA_BIN="${ADTC_ROOT}/tools/llama.cpp/llama-b10451"
if [[ -d "${LLAMA_BIN}" ]]; then
  export PATH="${LLAMA_BIN}:${PATH}"
  export LD_LIBRARY_PATH="${LLAMA_BIN}:${LD_LIBRARY_PATH:-}"
fi

mkdir -p docs/artifacts/v7 artifacts/profiler_stage/v7

python - <<'PY'
import json
import os
import subprocess
from pathlib import Path

root = Path(os.environ["ADTC_ROOT"])
gguf_dir = root / "artifacts/gguf/adapted"
out_dir = root / "docs/artifacts/v7"
stage_root = root / "artifacts/profiler_stage/v7"
acc_limit = os.environ.get("ACC_LIMIT", "50")
keys = [
    "qwen3_1_7b_merged_v7-Q4_K_M",
    "qwen3_1_7b_merged_v7-Q5_K_M",
    "qwen3_1_7b_merged_v7-Q6_K",
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
    except FileNotFoundError:
        print("WARN adtc-profiler not on PATH; writing stub summary only", flush=True)
        summary.append({"key": key, "gguf": str(gguf), "profile_ok": False, "error": "adtc-profiler missing"})
        failed += 1
        continue
    except Exception as e:  # noqa: BLE001
        summary.append({"key": key, "gguf": str(gguf), "profile_ok": False, "error": str(e)})
        failed += 1
        continue

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

ok = [s for s in summary if s.get("profile_ok") and (s.get("peak_rss_mb") or 1e9) <= 7000]
ok_sorted = sorted(ok, key=lambda s: s.get("composite") or 0, reverse=True)
winner = ok_sorted[0] if ok_sorted else None

(out_dir / "phase5_profile_summary_v7.json").write_text(json.dumps(summary, indent=2) + "\n")
if winner:
    (out_dir / "phase5_gate5_winner_v7.json").write_text(json.dumps(winner, indent=2) + "\n")

md = [
    "# Phase 5 — v7 profiler Pareto (Qwen3-1.7B EN)",
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
    md.append(f"**Gate 5 winner (v7):** `{winner['key']}` composite={winner.get('composite')}")
    md.append(f"- GGUF: `{winner['gguf']}`")
    md.append(f"- TPS={winner.get('tps')} peak_rss_mb={winner.get('peak_rss_mb')}")
(out_dir / "phase5_pareto_v7.md").write_text("\n".join(md) + "\n")
print(json.dumps({"winner": winner, "n_ok": len(ok), "failed": failed}, indent=2))
raise SystemExit(failed)
PY

echo "[prof_v7] Done"
