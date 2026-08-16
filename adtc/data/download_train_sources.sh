#!/usr/bin/env bash
# Download first-experiment Amharic/EN training corpora (not eval).
# Progress + OK/FAIL: adtc/logs/download_train/ (see docs/RUNLOGS.md)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADTC="$(cd "$HERE/.." && pwd)"
cd "$ADTC"

if [[ -f "$ADTC/tools/adtc-profiler-venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ADTC/tools/adtc-profiler-venv/bin/activate"
fi

python -c "import datasets" 2>/dev/null || {
  echo "error: install datasets first, e.g. pip install datasets" >&2
  exit 1
}

exec python "$HERE/download_train_sources.py" --profile first_experiment "$@"
