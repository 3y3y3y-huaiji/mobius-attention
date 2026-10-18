#!/usr/bin/env bash
# Strict LaTeX warning linter for undefined citations and references.
# Dual-Licensed under Apache-2.0 and MulanPSL-2.0.
set -euo pipefail

LOG_FILE="${1:-main.log}"

if [ ! -f "$LOG_FILE" ]; then
  echo "::error::LaTeX log file '$LOG_FILE' does not exist."
  exit 1
fi

if grep -E "LaTeX Warning: (Citation .* undefined|Reference .* undefined|There were undefined references)" "$LOG_FILE"; then
  echo "::error::Undefined citations or references detected in LaTeX log: $LOG_FILE"
  exit 1
fi

echo "[✓] Zero undefined citations or references detected in $LOG_FILE"
