#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run.sh PARAM1 [DATASETS_CONFIG]
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 RUN_CONFIG [DATASETS_CONFIG]"
  exit 1
fi

if [[ ! -f "$1" ]]; then
  echo "Error: RUN_CONFIG file '$1' does not exist."
  exit 1
fi
RUN_CONFIG=$(realpath "$1")

# — locate script and project root —
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( dirname "${SCRIPT_DIR}" )"

# — default config path if not provided —
DEFAULT_CONFIG="${PROJECT_ROOT}/data/common/datasets_config.json"
DATASETS_CONFIG="${2:-$DEFAULT_CONFIG}"

eval "$(conda shell.bash hook)"
conda activate RetroDMTA

PYTHON_SCRIPT="${PROJECT_ROOT}/src/run.py"

# — sanity checks —
if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  echo "Error: cannot find Python script at '${PYTHON_SCRIPT}'"
  exit 1
fi

if [[ ! -f "${DATASETS_CONFIG}" ]]; then
  echo "Error: config file not found at '${DATASETS_CONFIG}'"
  exit 1
fi

echo "🚀  Running run.py with:"
echo "    RUN_CONFIG  = ${RUN_CONFIG}"
echo "    DATASETS_CONFIG = ${DATASETS_CONFIG}"

cd "${PROJECT_ROOT}/src"
python "run.py" "${RUN_CONFIG}" "${DATASETS_CONFIG}"

cat << EOF

🎉  Done! You’ve ran all simulations.

Next steps:
  - Run ./scripts/3_postprocessing.sh to analyze the simulations.

EOF

