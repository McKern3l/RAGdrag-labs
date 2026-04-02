#!/bin/bash
# Start the agentic RAG server (R5 Hijack RD-0503 testing)
# Requires: pip install fastapi uvicorn chromadb httpx pydantic
# Requires: Ollama running locally

OLLAMA_MODEL=${OLLAMA_MODEL:-"qwen3:0.6b"} python targets/rag_server_agentic.py
