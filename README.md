<h1 align="center">RAGdrag Labs</h1>

<p align="center">
Test targets, sample results, and exercises for <a href="https://github.com/McKern3l/RAGdrag">RAGdrag</a>
</p>

---

## What is this?

This repo contains the lab environment for testing and learning with RAGdrag. It's separate from the main tool so the core stays lean. Everything here is optional.

## Primary Labs

| Lab | File | Coverage |
|--------|------|-------|
| **Open** (no guardrails) | `rag_server_open.py` | R1 Fingerprint, R2 Probe, R3 Exfiltrate |
| **Guarded** (regex filter) | `rag_server_guarded.py` | R3 Exfiltrate (deep bypass) |
| **Full Chain** | `rag_server_fullchain.py` | R1 through R6 in one authorized run |

The Full Chain lab combines exposed pipeline metadata, synthetic sensitive data,
create-only ingestion, verified deletion, simulated tool calls, guardrails,
monitoring, and real conversation history. Tool calls are evidence only; the lab
does not make outbound requests.

Specialized `ingestible`, `monitored`, and `agentic` targets remain available for
isolated R4, R5, and R6 exercises.

All servers are intentionally vulnerable. The Full Chain launcher binds to
loopback for local use; treat the specialized targets as untrusted lab services.
**Do not deploy any target or expose it to another network.**

## Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com/)** running locally with at least one model pulled
- **[RAGdrag](https://github.com/McKern3l/RAGdrag)** installed

```bash
# Pull a model (any Ollama model works)
ollama pull llama3.2
```

## Quick Start

```bash
# Install the tool
pip install -e ../ragdrag

# Install lab dependencies
pip install -e .

# Start the combined target (requires Ollama running locally)
./start-fullchain.sh

# In another terminal, run the authorized R1-R6 chain
ragdrag scan --target http://127.0.0.1:8899/chat \
  --phases R1,R2,R3,R4,R5,R6 \
  --max-requests 300 \
  --allow-write \
  --cleanup-url 'http://127.0.0.1:8899/documents/{id}' \
  --established-control baseline \
  --established-control negative-control \
  --established-control cleanup-verification \
  --response-field response \
  --history-field messages \
  --output fullchain-report.json
```

The three `--established-control` flags declare the lab controls you have
already established: a disposable baseline collection, a negative control, and
post-run cleanup verification. RAGdrag records every R4/R5 mutation and reports
exit code 5 if cleanup remains unresolved.

The Full Chain target intentionally returns `403` during R6's rapid-probe
sequence. A complete lab exercise can therefore exit 2 (`partial`) while still
executing all six capabilities. Confirm that the report contains R1 through R6,
that R4/R5 cleanup states are `removed`, and that unresolved cleanup is zero.

## Configuration

All servers are configured via environment variables. Set them before launching to customize behavior.

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3.2` | Model for generation. Any Ollama model works. |
| `CHROMA_DIR` | `/tmp/ragdrag-lab-<variant>` | ChromaDB persistence directory |
| `GUARDRAILS` | `0` | Set to `1` to run the guarded server variant |
| `REQUIRE_API_KEY` | *(empty)* | Set to `1` on ingestible server to require API key auth |

### Using a different model

Any model available in your Ollama instance works. Set `OLLAMA_MODEL` before starting:

```bash
# Use a small model for quick testing
OLLAMA_MODEL=qwen3:0.6b python targets/rag_server.py

# Use a larger model for more realistic responses
OLLAMA_MODEL=llama3.1:8b python targets/rag_server.py

# Use a custom fine-tuned model
OLLAMA_MODEL=my-custom-model python targets/rag_server.py

# Point at a remote Ollama instance
OLLAMA_URL=http://gpu-box:11434 OLLAMA_MODEL=llama3.2 python targets/rag_server.py
```

### Running specific server variants

```bash
# Open server (no guardrails) — default
python targets/rag_server.py

# Guarded server (regex output filtering)
GUARDRAILS=1 python targets/rag_server.py

# Full R1-R6 chain
./start-fullchain.sh

# Specialized component targets
python targets/rag_server_ingestible.py    # R4 Poison, R5 Hijack
python targets/rag_server_monitored.py     # R6 Evade
python targets/rag_server_agentic.py       # R5 Hijack (tool calling)
```

## Running Tests

Tests use mocked HTTP and do not require Ollama or ChromaDB running.

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

190 tests covering all 6 phases including full chain integration.

## Links

- **RAGdrag Tool:** [github.com/McKern3l/RAGdrag](https://github.com/McKern3l/RAGdrag)
- **Blog:** [github.com/McKern3l](https://github.com/McKern3l)

## License

MIT

## Author

**McKern3l** / [github.com/McKern3l](https://github.com/McKern3l) / [git.zero-lab.ai](https://git.zero-lab.ai/zero-lab-ai/ragdrag-labs)
