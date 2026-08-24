#!/bin/bash
#SBATCH --job-name=adtc_eval_v4_hf
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

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

m="training/runs/gemma3_4b_merged_v4"
[[ -d "${m}" ]] || { echo "missing ${m}" >&2; exit 1; }
OUT_DIR="${ADTC_ROOT}/docs/artifacts"
PERF_DIR="${OUT_DIR}/perf"
mkdir -p "${OUT_DIR}" "${PERF_DIR}"

python eval/run_hf_eval.py --model "${m}" --out "${OUT_DIR}/phase4_eval_gemma3_4b_merged_v4.json"
python eval/run_translate_test.py --model "${m}" --out "${PERF_DIR}/translate_gemma3_4b_merged_v4.md"

python - <<'PY'
import json
from pathlib import Path
out_dir = Path("docs/artifacts")
p = out_dir / "phase4_eval_gemma3_4b_merged_v4.json"
d = json.loads(p.read_text(encoding="utf-8"))
suites = d.get("suites", {})
row = {
    "model": d["model"],
    "afrimgsm_amh": suites.get("afrimgsm_amh", {}).get("acc"),
    "afrimgsm_eng": suites.get("afrimgsm_eng", {}).get("acc"),
    "afrimmlu_amh": suites.get("afrimmlu_amh", {}).get("acc"),
    "en_stem_holdout": suites.get("en_stem_holdout", {}).get("acc"),
    "custom_tutoring": suites.get("custom_tutoring", {}).get("acc"),
}
md = [
    "# Adapted HF eval — SFT mix v4 (AfriqueLLM-first, no CPT)",
    "",
    "Compare to v3: am MGSM 0.232, translate direct 0.10.",
    "",
    "| model | am MGSM | en MGSM | am MMLU | en holdout | tutoring |",
    "|-------|--------:|--------:|--------:|-----------:|---------:|",
    f"| `gemma3_4b_merged_v4` | {row['afrimgsm_amh']} | {row['afrimgsm_eng']} | {row['afrimmlu_amh']} | {row['en_stem_holdout']} | {row['custom_tutoring']} |",
    "",
]
(out_dir / "phase4_adapted_eval_v4.md").write_text("\n".join(md) + "\n", encoding="utf-8")
(out_dir / "phase4_adapted_eval_v4.json").write_text(json.dumps([row], indent=2) + "\n", encoding="utf-8")
print("wrote phase4_adapted_eval_v4.md")
PY
echo "[eval_v4_hf] done"
