#!/bin/bash
# Start the ingestible RAG server (R4 Poison + R5 Hijack testing)
# Requires: pip install fastapi uvicorn chromadb httpx pydantic
# Requires: Ollama running locally

OLLAMA_MODEL=${OLLAMA_MODEL:-"qwen3:0.6b"} python targets/rag_server_ingestible.py
