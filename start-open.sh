#!/bin/bash
# Start the open RAG target (no guardrails)
# Credentials flow straight through. Use for R1 Fingerprint and R3 Exfiltrate.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:0.6b}" python3 "$SCRIPT_DIR/targets/rag_server_open.py"
