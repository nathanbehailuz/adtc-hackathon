#!/bin/bash
#SBATCH --job-name=adtc_eval_adapted
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Evaluate all four merged HF models on frozen suites.
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
cd "${ADTC_ROOT}"

LIMIT="${LIMIT:-50}"
OUT_DIR="${ADTC_ROOT}/docs/artifacts"
mkdir -p "${OUT_DIR}"

MODELS=(
  "training/runs/qwen3_1_7b_merged_v0"
  "training/runs/qwen3_4b_merged_v0"
  "training/runs/qwen25_3b_instruct_merged_v0"
  "training/runs/gemma3_4b_merged_v0"
)

FAILED=0
for m in "${MODELS[@]}"; do
  name="$(basename "${m}")"
  out="${OUT_DIR}/phase4_eval_${name}.json"
  echo "======== EVAL ${m} ========"
  if python eval/run_hf_eval.py --model "${m}" --limit "${LIMIT}" --out "${out}"; then
    echo "[eval] OK ${out}"
  else
    echo "[eval] FAIL ${m}" >&2
    FAILED=$((FAILED + 1))
  fi
done

python - <<'PY'
import json
from pathlib import Path
out_dir = Path("docs/artifacts")
rows = []
for p in sorted(out_dir.glob("phase4_eval_*_merged_v0.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    suites = d.get("suites", {})
    rows.append({
        "model": d["model"],
        "afrimgsm_amh": suites.get("afrimgsm_amh", {}).get("acc"),
        "afrimgsm_eng": suites.get("afrimgsm_eng", {}).get("acc"),
        "afrimmlu_amh": suites.get("afrimmlu_amh", {}).get("acc"),
        "en_stem_holdout": suites.get("en_stem_holdout", {}).get("acc"),
        "custom_tutoring": suites.get("custom_tutoring", {}).get("acc"),
    })
md = ["# Phase 4 — Adapted HF eval", "", "| model | am MGSM | en MGSM | am MMLU | en holdout | tutoring |",
      "|-------|---------|---------|---------|------------|----------|"]
for r in rows:
    md.append(
        f"| `{r['model']}` | {r['afrimgsm_amh']} | {r['afrimgsm_eng']} | {r['afrimmlu_amh']} | {r['en_stem_holdout']} | {r['custom_tutoring']} |"
    )
md.append("")
(out_dir / "phase4_adapted_eval_v0.md").write_text("\n".join(md) + "\n", encoding="utf-8")
(out_dir / "phase4_adapted_eval_v0.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
print("wrote phase4_adapted_eval_v0.md")
PY

echo "[eval_adapted] failed=${FAILED}"
exit "${FAILED}"
