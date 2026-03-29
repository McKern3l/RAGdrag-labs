#!/bin/bash
# Start the guarded RAG target (regex output guardrails, bypassable)
# Direct credential queries get blocked. Use --deep to bypass.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:0.6b}" python3 "$SCRIPT_DIR/targets/rag_server_guarded.py"
