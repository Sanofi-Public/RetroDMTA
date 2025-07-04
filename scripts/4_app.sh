#!/usr/bin/env bash
set -euo pipefail

# — locate script and project root —
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( dirname "${SCRIPT_DIR}" )"

APP_DIR="${PROJECT_ROOT}/app"

if [[ ! -d "${APP_DIR}" ]]; then
  echo "❌  App directory not found at '${APP_DIR}'"
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate RetroDMTA

echo "🚀  Launching application..."

(
  cd "${APP_DIR}"
  streamlit run app.py "$@"
)

cat << EOF

🎉  Done! The application has finished running.
EOF
