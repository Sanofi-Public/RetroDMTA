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
  "1_1_top_molecules.py"
  "1_2_matrix.py"
  "1_3_mmp.py"
  "1_4_eda.py"
)

eval "$(conda shell.bash hook)"
conda activate RetroDMTA

for DATASET in "$@"; do
  echo "🔄  Preprocessing of '${DATASET}'"

  CODE_DIR="${PROJECT_ROOT}/code/${DATASET}"

  if [[ ! -d "${CODE_DIR}" ]]; then
    echo "❌  Skipping '${DATASET}': code directory not found at '${CODE_DIR}'"
    continue
  fi
  for script in "${SCRIPTS[@]}"; do
    (
      cd "${CODE_DIR}"
      python3 "${script}"
    )
  done
done

cat << EOF

🎉  Done! You’ve preprocessed: $*

Next steps:
  - Configure a simulation setup in ./config/ (you can copy ./config/template.json)
  - Run ./scripts/2_simulation.sh to launch the simulations

EOF
