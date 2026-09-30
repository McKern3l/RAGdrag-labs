#!/bin/bash
# Start the combined R1-R6 lab on the loopback interface only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
        PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2}" exec "$PYTHON_BIN" "$SCRIPT_DIR/targets/rag_server_fullchain.py"
