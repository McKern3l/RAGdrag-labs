#!/bin/bash
# Start the monitored RAG server (R6 Evade testing)
# Requires: pip install fastapi uvicorn chromadb httpx pydantic
# Requires: Ollama running locally

OLLAMA_MODEL=${OLLAMA_MODEL:-"qwen3:0.6b"} python targets/rag_server_monitored.py
