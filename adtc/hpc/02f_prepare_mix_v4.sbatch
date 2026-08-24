#!/bin/bash
#SBATCH --job-name=adtc_prep_mix_v4
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Prep mix v4: try AfriqueLLM; fallback NLLB + capped simonbutt; fat tutoring; mix_sft_v4.
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate /scratch/nz2212/adtc-hackathon/adtc/training/.conda-env
set -u
cd "${ADTC_ROOT}"

echo "[prep_v4] try AfriqueLLM download (Hub often 404 from HPC)"
if ! python data/download_train_sources.py --only afriquellm_gsm8k; then
  echo "[prep_v4] WARN AfriqueLLM download failed — will use on-disk NLLB + capped simonbutt"
fi

echo "[prep_v4] normalize (afriquellm if snapshot) + instruct sources"
python data/normalize_sft_sources.py --sources afriquellm_gsm8k walia dolly

echo "[prep_v4] build AM stem pool"
python data/build_am_stem_pool_v4.py --out data/train/am_stem_pool_v4.jsonl | tee docs/artifacts/am_stem_pool_v4_stats.json
AM_POOL="data/train/am_stem_pool_v4.jsonl"
N_POOL=$(wc -l < "${AM_POOL}")
echo "[prep_v4] AM stem pool rows=${N_POOL}"
if (( N_POOL < 1500 )); then
  echo "error: AM stem pool ${N_POOL} < 1500" >&2
  exit 1
fi

echo "[prep_v4] tutoring EN+AM authored"
python data/build_tutoring_sft_v4.py --en-limit 2000

echo "[prep_v4] derive tutoring from AM stem pool"
python data/derive_am_tutoring_v4.py \
  --in "${AM_POOL}" \
  --out data/train/am_tutoring_derived_v4.jsonl \
  --max-rows 1200

AM_INSTR_ARGS=()
for f in data/train/sources/walia_sft_v0.jsonl data/train/sources/dolly_am_sft_v0.jsonl; do
  [[ -f "${f}" && -s "${f}" ]] && AM_INSTR_ARGS+=("${f}")
done

echo "[prep_v4] mix_sft_v4"
MIX_CMD=(
  python data/mix_sft_v4.py
  --en-stem data/train/en_stem_sft_v4.jsonl
  --am-stem "${AM_POOL}"
  --am-tutoring data/train/am_tutoring_sft_v4.jsonl data/train/am_tutoring_derived_v4.jsonl
  --out data/train/sft_mix_v4.jsonl
  --total 8000
  --counts-out docs/artifacts/sft_mix_v4_counts.json
)
if ((${#AM_INSTR_ARGS[@]} > 0)); then
  MIX_CMD+=(--am-instruct "${AM_INSTR_ARGS[@]}")
fi
"${MIX_CMD[@]}"

python - <<'PY'
import json, random
from pathlib import Path
from collections import defaultdict
src = Path("data/train/sft_mix_v4.jsonl")
out = Path("docs/artifacts/amharic_review_sample_v4.jsonl")
by_dir = defaultdict(list)
with src.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        text = " ".join(m["content"] for m in obj.get("messages", []))
        if any("\u1200" <= ch <= "\u137F" for ch in text):
            by_dir[obj.get("direction", "unk")].append(obj)
rng = random.Random(42)
sample = []
per = max(1, 32 // max(1, len(by_dir)))
for _d, rows in by_dir.items():
    rng.shuffle(rows)
    sample.extend(rows[:per])
rng.shuffle(sample)
sample = sample[:32]
with out.open("w", encoding="utf-8") as f:
    for r in sample:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {out} n={len(sample)}")
PY

wc -l data/train/sft_mix_v4.jsonl "${AM_POOL}" data/train/am_tutoring_sft_v4.jsonl data/train/am_tutoring_derived_v4.jsonl
echo "[prep_v4] Done"
