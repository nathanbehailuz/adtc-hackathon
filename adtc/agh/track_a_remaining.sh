#!/usr/bin/env bash
# Track A remaining steps ON the Shadeform/AGH instance (after Q5 eval).
# Run inside tmux — profile + judge_smoke can take a while.
#
#   cd ~/adtc-hackathon/adtc/agh
#   source env.sh
#   bash track_a_remaining.sh
#
# Order: profile_gguf (A4) then judge_smoke (A3). Profile needs LD_LIBRARY_PATH
# for llama-bench; judge_smoke must unset the b10451 libs for llama-cpp-python.

set -euo pipefail
AGH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${AGH_DIR}/env.sh"

cd "${AGH_DIR}"
mkdir -p "${AGH_DIR}/logs" "${ADTC_ROOT}/docs/artifacts/v7"

echo "=============================================="
echo "[track_a] A4 profile_gguf (Q4/Q5/Q6 Gate 5)"
echo "=============================================="
bash "${AGH_DIR}/profile_gguf.sh"

echo "=============================================="
echo "[track_a] A3 judge_smoke (Q5_K_M default)"
echo "=============================================="
# strip b10451 so llama-cpp-python loads its own libllama
export LD_LIBRARY_PATH="$(
  printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/tools/llama.cpp/llama-b10451' | paste -sd: - || true
)"
V7_QUANT="${V7_QUANT:-Q5_K_M}" bash "${AGH_DIR}/judge_smoke.sh"

# Optional: also smoke the Gate 5 winner if different from Q5
WINNER_JSON="${ADTC_ROOT}/docs/artifacts/v7/phase5_gate5_winner_v7.json"
if [[ -f "${WINNER_JSON}" ]]; then
  WKEY="$(python -c "import json; print(json.load(open('${WINNER_JSON}')).get('key',''))")"
  WQUANT="${WKEY##*-}"
  if [[ -n "${WQUANT}" && "${WQUANT}" != "${V7_QUANT:-Q5_K_M}" ]]; then
    echo "[track_a] also judge_smoke winner quant=${WQUANT}"
    V7_QUANT="${WQUANT}" bash "${AGH_DIR}/judge_smoke.sh"
  fi
fi

echo "=============================================="
echo "[track_a] Done. Artifacts under docs/artifacts/v7/:"
ls -lah "${ADTC_ROOT}/docs/artifacts/v7/" | sed 's/^/  /'
echo
echo "On your laptop, pull results:"
echo "  export SSH_HOST=... SSH_PORT=..."
echo "  bash adtc/agh/pull_v7_artifacts.sh --with-gguf"
echo "Then shut down the instance (A6) when idle."
