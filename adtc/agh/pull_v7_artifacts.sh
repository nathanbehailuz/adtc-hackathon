#!/usr/bin/env bash
# Pull Gate 2 Track A v7 artifacts from Shadeform/AGH → this Mac.
#
# Usage (from anywhere):
#   export SSH_HOST='root@connect.REGION.gpuhub.com'
#   export SSH_PORT='XXXX'
#   # optional if repo is not under ~/adtc-hackathon:
#   export REMOTE_ADTC='~/adtc-hackathon/adtc'
#   # or:  export REMOTE_ADTC='/root/autodl-fs/adtc-hackathon/adtc'
#   bash adtc/agh/pull_v7_artifacts.sh              # evals + reports
#   bash adtc/agh/pull_v7_artifacts.sh --with-gguf  # also winner/provisional GGUFs
#
# After A3/A4 finish on the instance, re-run this script to fetch judge_smoke + profile winners.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADTC_ROOT="$(cd "${HERE}/.." && pwd)"
LOCAL_ART="${ADTC_ROOT}/docs/artifacts/v7"
LOCAL_GGUF="${ADTC_ROOT}/artifacts/gguf/adapted"

SSH_HOST="${SSH_HOST:?set SSH_HOST e.g. root@connect.<region>.gpuhub.com}"
SSH_PORT="${SSH_PORT:?set SSH_PORT from the console card}"
REMOTE_ADTC="${REMOTE_ADTC:-~/adtc-hackathon/adtc}"
WITH_GGUF=0
for a in "$@"; do
  case "$a" in
    --with-gguf) WITH_GGUF=1 ;;
    -h|--help)
      sed -n '2,16p' "$0"
      exit 0
      ;;
  esac
done

mkdir -p "${LOCAL_ART}" "${LOCAL_GGUF}" "${ADTC_ROOT}/agh/logs"

echo "[pull] ${SSH_HOST}:${SSH_PORT}  REMOTE_ADTC=${REMOTE_ADTC}"
echo "[pull] → ${LOCAL_ART}"

# Expand ~ on remote via ssh shell
remote_ls() {
  ssh -p "${SSH_PORT}" "${SSH_HOST}" "ls -1 ${REMOTE_ADTC}/$1 2>/dev/null || true"
}

echo "[pull] remote docs/artifacts/v7:"
remote_ls "docs/artifacts/v7" | sed 's/^/  /' || true

# JSON / MD artifacts (best-effort: skip missing)
ARTIFACTS=(
  "docs/artifacts/v7/qwen3_1_7b_merged_v7-Q4_K_M_eval.json"
  "docs/artifacts/v7/qwen3_1_7b_merged_v7-Q5_K_M_eval.json"
  "docs/artifacts/v7/qwen3_1_7b_merged_v7-Q6_K_eval.json"
  "docs/artifacts/v7/qwen3_1_7b_merged_v7_hf_eval.json"
  "docs/artifacts/v7/gguf_manifest.json"
  "docs/artifacts/v7/sft_mix_v7_counts.json"
  "docs/artifacts/v7/sft_mix_v7_report.md"
  "docs/artifacts/v7/phase5_profile_summary_v7.json"
  "docs/artifacts/v7/phase5_gate5_winner_v7.json"
  "docs/artifacts/v7/phase5_pareto_v7.md"
  "docs/artifacts/v7/phase5_profile_qwen3_1_7b_merged_v7-Q4_K_M.json"
  "docs/artifacts/v7/phase5_profile_qwen3_1_7b_merged_v7-Q5_K_M.json"
  "docs/artifacts/v7/phase5_profile_qwen3_1_7b_merged_v7-Q6_K.json"
  "docs/artifacts/v7/judge_smoke_Q4_K_M.json"
  "docs/artifacts/v7/judge_smoke_Q5_K_M.json"
  "docs/artifacts/v7/judge_smoke_Q6_K.json"
)

pulled=0
missing=0
for rel in "${ARTIFACTS[@]}"; do
  base="$(basename "${rel}")"
  if ssh -p "${SSH_PORT}" "${SSH_HOST}" "test -f ${REMOTE_ADTC}/${rel}"; then
    scp -P "${SSH_PORT}" "${SSH_HOST}:${REMOTE_ADTC}/${rel}" "${LOCAL_ART}/${base}"
    echo "  OK  ${base}"
    pulled=$((pulled + 1))
  else
    echo "  ..  ${base} (not on remote yet)"
    missing=$((missing + 1))
  fi
done

# Any other judge_smoke_*.json
while IFS= read -r rel; do
  [[ -z "${rel}" ]] && continue
  base="$(basename "${rel}")"
  [[ -f "${LOCAL_ART}/${base}" ]] && continue
  scp -P "${SSH_PORT}" "${SSH_HOST}:${REMOTE_ADTC}/${rel}" "${LOCAL_ART}/${base}"
  echo "  OK  ${base}"
  pulled=$((pulled + 1))
done < <(ssh -p "${SSH_PORT}" "${SSH_HOST}" \
  "ls ${REMOTE_ADTC}/docs/artifacts/v7/judge_smoke_*.json 2>/dev/null | sed \"s|^${REMOTE_ADTC}/||\"" || true)

# Logs (optional)
for rel in agh/logs/eval_gguf.log agh/logs/judge_smoke.log agh/logs/profile_gguf.log; do
  if ssh -p "${SSH_PORT}" "${SSH_HOST}" "test -f ${REMOTE_ADTC}/${rel}"; then
    scp -P "${SSH_PORT}" "${SSH_HOST}:${REMOTE_ADTC}/${rel}" "${ADTC_ROOT}/agh/logs/$(basename "${rel}")"
    echo "  OK  logs/$(basename "${rel}")"
  fi
done

if [[ "${WITH_GGUF}" -eq 1 ]]; then
  echo "[pull] GGUFs → ${LOCAL_GGUF}"
  # Prefer Gate 5 winner if present; else pull Q4/Q5/Q6
  winner_key=""
  if [[ -f "${LOCAL_ART}/phase5_gate5_winner_v7.json" ]]; then
    winner_key="$(python3 -c "import json; print(json.load(open('${LOCAL_ART}/phase5_gate5_winner_v7.json')).get('key',''))")"
  fi
  GGUFS=()
  if [[ -n "${winner_key}" ]]; then
    GGUFS+=("artifacts/gguf/adapted/${winner_key}.gguf")
  fi
  for q in Q4_K_M Q5_K_M Q6_K; do
    GGUFS+=("artifacts/gguf/adapted/qwen3_1_7b_merged_v7-${q}.gguf")
  done
  # unique
  declare -A seen=()
  for rel in "${GGUFS[@]}"; do
    [[ -n "${seen[$rel]:-}" ]] && continue
    seen[$rel]=1
    base="$(basename "${rel}")"
    if ssh -p "${SSH_PORT}" "${SSH_HOST}" "test -f ${REMOTE_ADTC}/${rel}"; then
      if [[ -f "${LOCAL_GGUF}/${base}" ]]; then
        echo "  skip ${base} (already local)"
      else
        echo "  scp ${base} (large)…"
        scp -P "${SSH_PORT}" "${SSH_HOST}:${REMOTE_ADTC}/${rel}" "${LOCAL_GGUF}/${base}"
        echo "  OK  ${base}"
      fi
    else
      echo "  ..  ${base} missing remote"
    fi
  done
fi

echo
echo "[pull] done: pulled≈${pulled}, not-yet-on-remote≈${missing}"
echo "[pull] local v7 artifacts:"
ls -lah "${LOCAL_ART}" | sed 's/^/  /'

if [[ ! -f "${LOCAL_ART}/phase5_gate5_winner_v7.json" ]] || [[ ! -f "${LOCAL_ART}/judge_smoke_Q5_K_M.json" ]]; then
  cat <<'EOF'

[next] A3/A4 still need to run ON the instance (GGUF + profiler). In tmux:

  cd ~/adtc-hackathon/adtc/agh && source env.sh
  # A4 first (sets LD_LIBRARY_PATH for llama-bench):
  bash profile_gguf.sh
  # A3:
  unset LD_LIBRARY_PATH
  V7_QUANT=Q5_K_M bash judge_smoke.sh

Or one shot:
  bash agh/track_a_remaining.sh

Then re-run this pull script (add --with-gguf after A4 for the winner).
EOF
fi
