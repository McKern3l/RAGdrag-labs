"""E2E tests for R5 Hijack against ingestible + agentic server patterns.

Uses httpx MockTransport to simulate server behavior.
"""

from __future__ import annotations

import json
import re
import uuid

import httpx
import pytest

from ragdrag.core.hijack import (
    redirect_retrieval,
    saturate_context_window,
    manipulate_tool_calls,
    assess_persistence,
    run_hijack,
)
from ragdrag.core.poison import InjectedDocument, inject_document


# --- Mock agentic server (ingestible + tool calls) ---

_injected_docs: dict[str, dict] = {}


def _agentic_handler(request: httpx.Request) -> httpx.Response:
    """Simulate the agentic RAG server with ingestion + tool calling."""
    path = request.url.path

    if request.method == "POST" and path == "/ingest":
        body = json.loads(request.content)
        text = body.get("text", "") or body.get("content", "")
        if not text:
            return httpx.Response(422, json={"detail": "Missing text"})
        doc_id = body.get("id", f"injected-{uuid.uuid4().hex[:8]}")
        _injected_docs[doc_id] = {"text": text, "metadata": body.get("metadata", {})}
        return httpx.Response(201, json={"status": "ok", "id": doc_id})

    if request.method == "POST" and path == "/chat":
        body = json.loads(request.content)
        query = body.get("query", "").lower()

        answer_parts = []
        sources = [{"source": "hr_policies.pdf", "relevance": 0.80}]
        context = ["Company policies and procedures."]
        tool_calls = []

        for doc_id, doc in _injected_docs.items():
            doc_text = doc["text"]
            doc_lower = doc_text.lower()
            query_words = set(query.split())
            if any(w in doc_lower for w in query_words if len(w) > 3):
                answer_parts.append(doc_text)
                context.append(doc_text)
                sources.append({"source": f"injected-{doc_id}", "relevance": 0.95})

                urls = re.findall(r"https?://\S+", doc_text)
                for url in urls:
                    if any(kw in doc_lower for kw in ("fetch", "request", "navigate")):
                        tool_calls.append({
                            "tool": "fetch_url", "status": "executed",
                            "args": {"url": url},
                        })

        if not answer_parts:
            answer_parts.append("Based on our documentation, here is the relevant information.")

        return httpx.Response(200, json={
            "response": " ".join(answer_parts),
            "sources": sources,
            "context": context,
            "tool_calls": tool_calls,
        })

    if request.method == "OPTIONS":
        return httpx.Response(200)

    return httpx.Response(404)


def _client():
    _injected_docs.clear()
    transport = httpx.MockTransport(_agentic_handler)
    return httpx.Client(transport=transport, base_url="http://testserver")


TARGET = "http://testserver/chat"
INGEST = "http://testserver/ingest"


class TestRetrievalRedirectionE2E:
    def test_redirect_password_reset(self):
        with _client() as client:
            findings = redirect_retrieval(TARGET, client, ingest_url=INGEST)
            injection_findings = [f for f in findings if f.technique_id == "RD-0401"]
            redir_findings = [f for f in findings if f.technique_id == "RD-0501"]
            assert len(injection_findings) >= 1
            assert len(redir_findings) >= 1

    def test_redirect_with_camouflage(self):
        with _client() as client:
            findings = redirect_retrieval(
                TARGET, client, ingest_url=INGEST, use_camouflage=True,
            )
            assert any(f.technique_id == "RD-0401" for f in findings)


class TestContextSaturationE2E:
    def test_saturate_security_topic(self):
        with _client() as client:
            findings = saturate_context_window(
                TARGET, client, "security", num_documents=3, ingest_url=INGEST,
            )
            injection_findings = [f for f in findings if f.technique_id == "RD-0401"]
            assert len(injection_findings) >= 1

    def test_saturation_measured(self):
        with _client() as client:
            findings = saturate_context_window(
                TARGET, client, "security", num_documents=3, ingest_url=INGEST,
            )
            sat_findings = [f for f in findings if f.technique_id == "RD-0502"]
            assert len(sat_findings) >= 1


class TestToolManipulationE2E:
    def test_inject_tool_trigger(self):
        with _client() as client:
            findings = manipulate_tool_calls(
                TARGET, client, "evil.callback.com", ingest_url=INGEST,
            )
            assert any(f.technique_id == "RD-0401" for f in findings)
            tool_findings = [f for f in findings if f.technique_id == "RD-0503"]
            assert len(tool_findings) >= 1


class TestPersistenceE2E:
    def test_injected_content_persists(self):
        with _client() as client:
            _, doc = inject_document(
                TARGET, client,
                "PERSISTENT MARKER: This content should remain in the knowledge base across queries.",
                ingest_url=INGEST,
            )
            assert doc is not None

            findings = assess_persistence(TARGET, client, doc, "persistent marker")
            assert len(findings) >= 1
            assert findings[0].evidence["persistent_checks"] >= 1


class TestRunHijackE2E:
    def test_full_hijack_run(self):
        with _client() as client:
            result = run_hijack(
                TARGET, client,
                callback_url="evil.callback.com",
                ingest_url=INGEST,
            )
            assert len(result.findings) >= 2
            assert result.target == TARGET

    def test_hijack_result_serializes(self):
        with _client() as client:
            result = run_hijack(TARGET, client, ingest_url=INGEST)
            d = result.to_dict()
            assert isinstance(d["findings"], list)
