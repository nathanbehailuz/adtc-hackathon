#!/usr/bin/env bash
# Milestone B — DO NOT RUN until Milestone A preflight is approved.
# Usage (from adtc/hpc):
#   V5_ALIAS=qwen35_extcm bash submit_v5_chain.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

ALIAS="${V5_ALIAS:?set V5_ALIAS=qwen35_extcm|qwen35_afrique|qwen3_afrique}"
EXPORT="ALL,V5_ALIAS=${ALIAS}"

J_PREP=$(sbatch --parsable --export="${EXPORT}" 02g_prepare_mix_v5.sbatch)
echo "prep ${J_PREP}"

J_CPT=$(sbatch --parsable --export="${EXPORT}" --dependency=afterok:"${J_PREP}" 03d_train_cpt_v5.sbatch)
echo "cpt ${J_CPT}"

J_MCPT=$(sbatch --parsable --export="${EXPORT}" --dependency=afterok:"${J_CPT}" 03e_merge_cpt_v5.sbatch)
echo "merge_cpt ${J_MCPT}"

J_M1=$(sbatch --parsable --export="${EXPORT},V5_STAGE=m1" --dependency=afterok:"${J_MCPT}" 12h_eval_v5_hf.sbatch)
echo "eval_m1 ${J_M1}"

J_SFT=$(sbatch --parsable --export="${EXPORT}" --dependency=afterok:"${J_MCPT}" 04f_train_sft_v5.sbatch)
echo "sft ${J_SFT}"

J_MSFT=$(sbatch --parsable --export="${EXPORT}" --dependency=afterok:"${J_SFT}" 05e_merge_lora_v5.sbatch)
echo "merge_sft ${J_MSFT}"

J_M2=$(sbatch --parsable --export="${EXPORT},V5_STAGE=m2" --dependency=afterok:"${J_MSFT}" 12h_eval_v5_hf.sbatch)
echo "eval_m2 ${J_M2}"

J_GGUF=$(sbatch --parsable --export="${EXPORT}" --dependency=afterok:"${J_MSFT}" 08g_convert_gguf_v5.sbatch)
echo "gguf ${J_GGUF}"

J_EVAL=$(sbatch --parsable --export="${EXPORT},V5_QUANT=Q4_K_M" --dependency=afterok:"${J_GGUF}" 12g_eval_v5.sbatch)
echo "eval_gguf_q4 ${J_EVAL}"

J_EVAL5=$(sbatch --parsable --export="${EXPORT},V5_QUANT=Q5_K_M" --dependency=afterok:"${J_GGUF}" 12g_eval_v5.sbatch)
echo "eval_gguf_q5 ${J_EVAL5}"

echo "----"
echo "v5 train chain for ${ALIAS}: ${J_PREP}->${J_CPT}->${J_MCPT}->{${J_M1},${J_SFT}->${J_MSFT}->{${J_M2},${J_GGUF}->{${J_EVAL},${J_EVAL5}}}}"
echo "monitor: squeue -u \$USER"
