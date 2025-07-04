#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/preprocessing.sh DATASET [DATASET...]
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 DATASET [DATASET...]"
  exit 1
fi

# — locate script and project root —
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( dirname "${SCRIPT_DIR}" )"

# — which preprocessing scripts to run (relative to code/<DATASET>) —
declare -a SCRIPTS=(
  "2_1_exploitation.py"
  "2_2_exploration.py"
  "2_3_tmap.py"
)

# — loop over every dataset you passed in —
for DATASET in "$@"; do
  echo "🔄  Postprocessing of '${DATASET}'"

  CODE_DIR="${PROJECT_ROOT}/code/${DATASET}"

  if [[ ! -d "${CODE_DIR}" ]]; then
    echo "❌  Skipping '${DATASET}': code directory not found at '${CODE_DIR}'"
    continue
  fi

  for script in "${SCRIPTS[@]}"; do
    # pick the right env
    if [[ "$script" == "2_3_tmap.py" ]]; then
      ENV_NAME="RetroDMTA_TMAP"
    else
      ENV_NAME="RetroDMTA"
    fi
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}"
    cd "${CODE_DIR}"
    python3 "${script}"
  done

done

cat << EOF

🎉  Done! You’ve postprocessed: $*

EOF