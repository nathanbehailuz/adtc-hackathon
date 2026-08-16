#!/bin/bash
#SBATCH --job-name=adtc_translate_test
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Direct Amharic vs English translate-test (Qwen3-1.7B HF or local snapshot).
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
cd "${ADTC_ROOT}"

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
# Prefer local merged if present for post-adapt; Phase 2 default is base.
python eval/run_translate_test.py \
  --model "${MODEL}" \
  --limit "${LIMIT:-50}" \
  --out docs/artifacts/phase2_translate_test_v0.md

echo "[translate_test] Done"
