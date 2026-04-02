<h1 align="center">RAGdrag Labs</h1>

<p align="center">
Test targets, sample results, and exercises for <a href="https://github.com/McKern3l/RAGdrag">RAGdrag</a>
</p>

---

## What is this?

This repo contains the lab environment for testing and learning with RAGdrag. It's separate from the main tool so the core stays lean. Everything here is optional.

## Server Fleet

| Server | File | Tests |
|--------|------|-------|
| **Open** (no guardrails) | `rag_server_open.py` | R1 Fingerprint, R2 Probe, R3 Exfiltrate |
| **Guarded** (regex filter) | `rag_server_guarded.py` | R3 Exfiltrate (deep bypass) |
| **Ingestible** (POST /ingest) | `rag_server_ingestible.py` | R4 Poison, R5 Hijack |
| **Monitored** (logging + anomaly) | `rag_server_monitored.py` | R6 Evade |
| **Agentic** (tool calling) | `rag_server_agentic.py` | R5 Hijack (RD-0503) |

All servers are intentionally vulnerable. **Do not deploy anywhere accessible.**

## Quick Start

```bash
# Install the tool
pip install -e ../ragdrag

# Install lab dependencies
pip install fastapi uvicorn chromadb httpx pydantic

# Start a server (requires Ollama running locally)
OLLAMA_MODEL=qwen3:0.6b ./start-open.sh
OLLAMA_MODEL=qwen3:0.6b ./start-ingestible.sh
OLLAMA_MODEL=qwen3:0.6b ./start-monitored.sh
OLLAMA_MODEL=qwen3:0.6b ./start-agentic.sh

# Run the full kill chain
ragdrag scan -t http://localhost:8899/chat
```

## Running Tests

```bash
pip install pytest respx
pytest tests/ -v
```

178 tests covering all 6 phases including full kill chain integration.

## Links

- **RAGdrag Tool:** [github.com/McKern3l/RAGdrag](https://github.com/McKern3l/RAGdrag)
- **Blog:** [github.com/McKern3l](https://github.com/McKern3l)

## License

MIT

## Author

**McKern3l** / [github.com/McKern3l](https://github.com/McKern3l)
