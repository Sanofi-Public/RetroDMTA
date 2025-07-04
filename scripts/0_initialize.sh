#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/initial.sh DATASET [DATASET...]
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 DATASET [DATASET...]"
  exit 1
fi

# — locate script and project root —
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( dirname "${SCRIPT_DIR}" )"
TEMPLATE_DIR="${SCRIPT_DIR}/template"

# — sanity check: template must exist —
if [[ ! -d "${TEMPLATE_DIR}" ]]; then
  echo "Error: template directory not found at '${TEMPLATE_DIR}'"
  exit 1
fi

# — loop over every dataset you passed in —
for DATASET in "$@"; do
  CODE_DEST="${PROJECT_ROOT}/code/${DATASET}"
  DATA_DEST="${PROJECT_ROOT}/data/${DATASET}"

  # — ensure data folder exists (silent if already there) —
  if [[ ! -d "${DATA_DEST}" ]]; then
    mkdir -p "${DATA_DEST}"
  fi

  # — recreate code folder from template —
  if [[ -d "${CODE_DEST}" ]]; then
    rm -rf "${CODE_DEST}"
    # echo "   • Removed existing code directory at '${CODE_DEST}'"
  fi
  mkdir -p "$( dirname "${CODE_DEST}" )"
  cp -r "${TEMPLATE_DIR}" "${CODE_DEST}"
done


cat << EOF

🎉  Done! You’ve initialized: $*

Next steps for each dataset:
  - Generate the data.csv, data_aggregated.csv and blueprint.csv files and put them into ./data/<DATASET>
  - Run the 0_*.ipynb notebooks in ./code/<DATASET> to determine simulation parameters and TMAP coordinates
  - Run ./scripts/1_preprocessing.sh to generate the mandatory files for simulations

EOF
