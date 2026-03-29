"""RAG test target — default entry point.

Two configurations available:

  rag_server_open.py      No guardrails. Credentials flow straight through.
                          Best for: R1 Fingerprint, R3 Exfiltrate basics.

  rag_server_guarded.py   Regex-based output guardrails (bypassable).
                          Best for: R3 deep mode (--deep) guardrail bypass.

This file runs the open version by default. Set GUARDRAILS=1 to run guarded.

Usage:
    python targets/rag_server.py                    # open (no guardrails)
    GUARDRAILS=1 python targets/rag_server.py       # guarded (bypassable)
"""

import os
import sys

if os.environ.get("GUARDRAILS", "0") == "1":
    print("[*] Loading guarded server (regex output guardrails enabled)")
    from rag_server_guarded import app
else:
    print("[*] Loading open server (no guardrails)")
    from rag_server_open import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8899)
