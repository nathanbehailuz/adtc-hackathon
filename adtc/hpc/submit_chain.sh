#!/usr/bin/env bash
# Submit the default Jubail chain with afterok dependencies.
# Run from a login node (not on compute):
#   cd /scratch/nz2212/adtc-hackathon/adtc/hpc && bash submit_chain.sh
#
# Env vars:
#   SKIP_SETUP=1  — skip setup_env.sbatch if conda env already exists
#   CHAIN_FROM=N  — start from stage N (1=download_train … 5=merge); implies SKIP_SETUP
set -euo pipefail

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HPC_DIR}"
mkdir -p logs

SKIP_SETUP="${SKIP_SETUP:-0}"
CHAIN_FROM="${CHAIN_FROM:-0}"
if [[ "${CHAIN_FROM}" != "0" ]]; then
  SKIP_SETUP=1
fi

submit() {
  local script="$1"
  shift
  sbatch --parsable "$@" "${script}"
}

PREV=""
if [[ "${SKIP_SETUP}" != "1" ]]; then
  PREV="$(submit setup_env.sbatch)"
  echo "submitted setup_env.sbatch → ${PREV}"
fi

stage_ge() {
  local n="$1"
  [[ "${CHAIN_FROM}" == "0" || "${CHAIN_FROM}" -le "${n}" ]]
}

dep_flag() {
  if [[ -n "${PREV}" ]]; then
    echo "--dependency=afterok:${PREV}"
  fi
}

if stage_ge 1; then
  # shellcheck disable=SC2046
  PREV="$(submit 01_download_train.sbatch $(dep_flag))"
  echo "submitted 01_download_train.sbatch → ${PREV}"
fi

if stage_ge 2; then
  # shellcheck disable=SC2046
  PREV="$(submit 02_prepare_data.sbatch $(dep_flag))"
  echo "submitted 02_prepare_data.sbatch → ${PREV}"
fi

if stage_ge 3; then
  # shellcheck disable=SC2046
  PREV="$(submit 03_download_models.sbatch $(dep_flag))"
  echo "submitted 03_download_models.sbatch → ${PREV}"
fi

if stage_ge 4; then
  # shellcheck disable=SC2046
  PREV="$(submit 04_train_sft_1_7b.sbatch $(dep_flag))"
  echo "submitted 04_train_sft_1_7b.sbatch → ${PREV}"
fi

if stage_ge 5; then
  # shellcheck disable=SC2046
  PREV="$(submit 05_merge_lora.sbatch $(dep_flag))"
  echo "submitted 05_merge_lora.sbatch → ${PREV}"
fi

echo
echo "Monitor:  squeue -u \$USER"
echo "Cancel:   scancel <jobid>"
echo "Slurm logs: ${HPC_DIR}/logs/"
echo "Stage logs: ${HPC_DIR}/../logs/"
echo "4B SFT (optional): sbatch 04b_train_sft_4b.sbatch"
