<h1 align="center">RAGdrag Labs</h1>

<p align="center">
Test targets, sample results, and exercises for <a href="https://github.com/McKern3l/RAGdrag">RAGdrag</a>
</p>

---

## What is this?

This repo contains the lab environment for testing and learning with RAGdrag. It's separate from the main tool so the core stays lean. Everything here is optional.

## Contents

| Path | What |
|------|------|
| `targets/rag_server.py` | Vulnerable RAG server with planted credentials, connection strings, and internal documents |
| `dogfood-results.json` | Real output from a test run against the bundled target |
| `tests/` | Test suite for validating RAGdrag modules |

## Quick Start

```bash
# Install the tool first
pip install -e ../ragdrag  # or: git clone + pip install from McKern3l/RAGdrag

# Install lab dependencies
pip install fastapi uvicorn chromadb

# Start the vulnerable RAG target (requires Ollama running locally)
OLLAMA_MODEL=your-model python targets/rag_server.py

# In another terminal, run RAGdrag against it
ragdrag fingerprint -t http://localhost:8899/chat
ragdrag exfiltrate -t http://localhost:8899/chat
```

## Test Target

`targets/rag_server.py` is an intentionally vulnerable RAG server built with FastAPI + ChromaDB + Ollama. It contains:

- Fake credentials and API keys in the knowledge base
- Internal documents with connection strings
- No guardrails on retrieval output
- URL fetcher functionality (for RD-0304 testing)

**This is designed to be broken into.** Do not deploy it anywhere accessible.

## Running Tests

```bash
pip install pytest pytest-asyncio respx
pytest tests/ -v
```

## Links

- **RAGdrag Tool:** [github.com/McKern3l/RAGdrag](https://github.com/McKern3l/RAGdrag)
- **Blog:** [github.com/McKern3l](https://github.com/McKern3l)

## License

MIT

## Author

**McKern3l** / [github.com/McKern3l](https://github.com/McKern3l)
