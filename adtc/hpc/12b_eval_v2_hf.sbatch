#!/bin/bash
#SBATCH --job-name=adtc_eval_v2_hf
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Fast gate: HF merged v2 on GPU (transformers). Not judge-equivalent to CPU GGUF;
# use for iteration. Official packaging numbers still come from llama.cpp Q4 on compute.
#
#   sbatch 12b_eval_v2_hf.sbatch
#   LIMIT=50 sbatch 12b_eval_v2_hf.sbatch
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate /scratch/nz2212/adtc-hackathon/adtc/training/.conda-env
set -u
cd "${ADTC_ROOT}"

export PYTHONUNBUFFERED=1

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS+=(--limit "${LIMIT}")
fi

OUT_DIR="${ADTC_ROOT}/docs/artifacts"
PERF_DIR="${OUT_DIR}/perf"
mkdir -p "${OUT_DIR}" "${PERF_DIR}"

MODELS=(
  "training/runs/gemma3_4b_merged_v2"
  "training/runs/qwen3_1_7b_merged_v2"
)

FAILED=0
for m in "${MODELS[@]}"; do
  name="$(basename "${m}")"
  if [[ ! -d "${m}" ]]; then
    echo "[eval_v2_hf] SKIP missing ${m}" >&2
    FAILED=$((FAILED + 1))
    continue
  fi
  out="${OUT_DIR}/phase4_eval_${name}.json"
  echo "======== HF EVAL ${m} ========"
  if python eval/run_hf_eval.py --model "${m}" "${LIMIT_ARGS[@]}" --out "${out}"; then
    echo "[eval_v2_hf] OK ${out}"
  else
    echo "[eval_v2_hf] FAIL ${m}" >&2
    FAILED=$((FAILED + 1))
  fi

  echo "======== TRANSLATE-TEST ${m} ========"
  tt_out="${PERF_DIR}/translate_${name}.md"
  if python eval/run_translate_test.py --model "${m}" "${LIMIT_ARGS[@]}" --out "${tt_out}"; then
    echo "[eval_v2_hf] OK ${tt_out}"
  else
    echo "[eval_v2_hf] FAIL translate ${m}" >&2
    FAILED=$((FAILED + 1))
  fi
done

python - <<'PY'
import json
from pathlib import Path

out_dir = Path("docs/artifacts")
rows = []
for name in ("gemma3_4b_merged_v2", "qwen3_1_7b_merged_v2"):
    p = out_dir / f"phase4_eval_{name}.json"
    if not p.exists():
        continue
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

md = [
    "# Adapted HF eval — SFT mix v2",
    "",
    "GPU / transformers on `*_merged_v2`. Compare to Phase 4 v0 table in RESULTS_REPORT.",
    "Not a substitute for CPU llama.cpp GGUF packaging scores.",
    "",
    "| model | am MGSM | en MGSM | am MMLU | en holdout | tutoring |",
    "|-------|--------:|--------:|--------:|-----------:|---------:|",
]
for r in rows:
    short = Path(r["model"]).name
    md.append(
        f"| `{short}` | {r['afrimgsm_amh']} | {r['afrimgsm_eng']} | "
        f"{r['afrimmlu_amh']} | {r['en_stem_holdout']} | {r['custom_tutoring']} |"
    )
md.append("")
(out_dir / "phase4_adapted_eval_v2.md").write_text("\n".join(md) + "\n", encoding="utf-8")
(out_dir / "phase4_adapted_eval_v2.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
print("wrote phase4_adapted_eval_v2.md")
PY

echo "[eval_v2_hf] failed=${FAILED}"
exit "${FAILED}"
