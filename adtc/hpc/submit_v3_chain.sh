#!/bin/bash
# Submit Gemma-only Amharic v3 chain (on-disk remix → CPT → SFT → GGUF → eval).
# Run from adtc/hpc/:
#   bash submit_v3_chain.sh
set -euo pipefail
cd "$(dirname "$0")"

J_PREP=$(sbatch --parsable 02e_prepare_mix_v3.sbatch)
echo "submitted prepare_mix_v3 ${J_PREP}"

J_CPT=$(sbatch --parsable --dependency=afterok:"${J_PREP}" 03b_train_cpt_v3.sbatch)
echo "submitted cpt_v3 ${J_CPT} afterok:${J_PREP}"

J_MERGE_CPT=$(sbatch --parsable --dependency=afterok:"${J_CPT}" 03c_merge_cpt_v3.sbatch)
echo "submitted merge_cpt_v3 ${J_MERGE_CPT} afterok:${J_CPT}"

J_SFT=$(sbatch --parsable --dependency=afterok:"${J_MERGE_CPT}" 04d_train_sft_v3.sbatch)
echo "submitted train_sft_v3 ${J_SFT} afterok:${J_MERGE_CPT}"

J_MERGE=$(sbatch --parsable --dependency=afterok:"${J_SFT}" 05c_merge_lora_v3.sbatch)
echo "submitted merge_v3 ${J_MERGE} afterok:${J_SFT}"

J_GGUF=$(sbatch --parsable --dependency=afterok:"${J_MERGE}" 08c_convert_gguf_v3.sbatch)
echo "submitted convert_gguf_v3 ${J_GGUF} afterok:${J_MERGE}"

J_EVAL=$(sbatch --parsable --dependency=afterok:"${J_GGUF}" 12c_eval_v3.sbatch)
echo "submitted eval_v3 ${J_EVAL} afterok:${J_GGUF}"

J_HF=$(sbatch --parsable --dependency=afterok:"${J_MERGE}" 12d_eval_v3_hf.sbatch)
echo "submitted eval_v3_hf ${J_HF} afterok:${J_MERGE} (parallel with GGUF)"

echo "----"
echo "v3 chain: ${J_PREP} -> ${J_CPT} -> ${J_MERGE_CPT} -> ${J_SFT} -> ${J_MERGE} -> {${J_GGUF}->${J_EVAL}, ${J_HF}}"
echo "monitor: squeue -u \$USER"
