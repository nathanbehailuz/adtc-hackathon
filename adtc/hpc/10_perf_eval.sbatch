#!/bin/bash
#SBATCH --job-name=adtc_perf_eval
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 48:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Full ADTC-shaped perf eval on three adapted Q4 GGUFs:
#   profiler (with accuracy) + frozen GGUF suites + translate-test → perf/<key>_v0.json
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

PERF_DIR="${ADTC_ROOT}/docs/artifacts/perf"
STAGE_ROOT="${ADTC_ROOT}/artifacts/profiler_stage/perf"
GGUF_DIR="${ADTC_ROOT}/artifacts/gguf/adapted"
# Frozen + translate-test: full suites unless LIMIT= is exported.
# Profiler ARC-Easy is a separate lm-eval task; cap with ACC_LIMIT (default 50).
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS+=(--limit "${LIMIT}")
fi
ACC_LIMIT="${ACC_LIMIT:-50}"
mkdir -p "${PERF_DIR}" "${STAGE_ROOT}"

KEYS=(
  "qwen3_1_7b_merged_v0-Q4_K_M"
  "gemma3_4b_merged_v0-Q4_K_M"
  "qwen3_4b_merged_v0-Q4_K_M"
)

FAILED=0
for key in "${KEYS[@]}"; do
  gguf="${GGUF_DIR}/${key}.gguf"
  if [[ ! -f "${gguf}" ]]; then
    echo "[perf] SKIP missing ${gguf}" >&2
    FAILED=$((FAILED + 1))
    continue
  fi
  echo "========================================"
  echo "[perf] START ${key}"
  echo "========================================"

  stage="${STAGE_ROOT}/${key}"
  python eval/stage_gguf_submission.py \
    --gguf "${gguf}" \
    --out-dir "${stage}" \
    --name "${key}" \
    --quant "Q4_K_M"

  profiler_out="${PERF_DIR}/profiler_${key}.json"
  echo "[perf] profiler+accuracy → ${profiler_out}"
  if ! adtc-profiler run \
      --submission "${stage}" \
      --mode participant \
      --accuracy-task arc_easy \
      --accuracy-limit "${ACC_LIMIT}" \
      --output "${profiler_out}"; then
    echo "[perf] FAIL profiler ${key}" >&2
    FAILED=$((FAILED + 1))
    continue
  fi

  frozen_out="${PERF_DIR}/frozen_${key}.json"
  echo "[perf] frozen GGUF eval → ${frozen_out}"
  if ! python eval/run_gguf_eval.py \
      --gguf "${gguf}" \
      "${LIMIT_ARGS[@]}" \
      --n-threads "${SLURM_CPUS_PER_TASK:-8}" \
      --out "${frozen_out}"; then
    echo "[perf] FAIL frozen ${key}" >&2
    FAILED=$((FAILED + 1))
    continue
  fi

  tt_md="${PERF_DIR}/translate_${key}.md"
  tt_json="${PERF_DIR}/translate_${key}.json"
  echo "[perf] translate-test → ${tt_json}"
  if ! python eval/run_translate_test.py \
      --gguf "${gguf}" \
      "${LIMIT_ARGS[@]}" \
      --n-threads "${SLURM_CPUS_PER_TASK:-8}" \
      --out "${tt_md}"; then
    echo "[perf] FAIL translate ${key}" >&2
    FAILED=$((FAILED + 1))
    continue
  fi
  # run_translate_test writes .json beside .md
  if [[ ! -f "${tt_json}" ]]; then
    tt_json="${tt_md%.md}.json"
  fi

  out="${PERF_DIR}/${key}_v0.json"
  python eval/build_perf_record.py \
    --model-key "${key}" \
    --gguf "${gguf}" \
    --profiler "${profiler_out}" \
    --frozen "${frozen_out}" \
    --translate "${tt_json}" \
    --out "${out}"

  echo "[perf] OK ${out}"
done

python eval/aggregate_perf.py --perf-dir "${PERF_DIR}" --keys "${KEYS[@]}"

echo "[perf] DONE failed=${FAILED}"
exit "${FAILED}"
