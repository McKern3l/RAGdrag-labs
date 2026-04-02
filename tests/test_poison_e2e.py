"""E2E tests for R4 Poison against the ingestible server pattern.

Uses httpx MockTransport to simulate rag_server_ingestible.py behavior.
"""

from __future__ import annotations

import json
import uuid

import httpx
import pytest

from ragdrag.core.poison import (
    inject_document,
    verify_injection,
    assess_embedding_dominance,
    deploy_credential_trap,
    inject_instructions,
    run_poison,
    InjectedDocument,
)


# --- Mock ingestible server ---

_injected_docs: dict[str, dict] = {}


def _ingestible_handler(request: httpx.Request) -> httpx.Response:
    """Simulate the ingestible RAG server."""
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

        # Simulate retrieval: check if any injected doc matches the query
        answer_parts = ["Based on our documentation:"]
        sources = [{"source": "hr_policies.pdf", "relevance": 0.85}]
        context = ["Company Password Reset Policy: Default temporary password is Welcome123!"]

        for doc_id, doc in _injected_docs.items():
            doc_text = doc["text"].lower()
            query_words = set(query.split())
            doc_words = set(doc_text.split()[:20])
            if query_words & doc_words or any(w in doc_text for w in query.split() if len(w) > 4):
                answer_parts.append(doc["text"])
                context.append(doc["text"])
                sources.append({"source": f"injected-{doc_id}", "relevance": 0.92})

        return httpx.Response(200, json={
            "response": " ".join(answer_parts),
            "sources": sources,
            "context": context,
        })

    if request.method == "DELETE" and path.startswith("/documents/"):
        doc_id = path.split("/")[-1]
        if doc_id in _injected_docs:
            del _injected_docs[doc_id]
            return httpx.Response(200, json={"status": "deleted", "id": doc_id})
        return httpx.Response(404, json={"detail": "Not found"})

    if request.method == "OPTIONS":
        return httpx.Response(200)

    return httpx.Response(404)


def _client():
    _injected_docs.clear()
    transport = httpx.MockTransport(_ingestible_handler)
    return httpx.Client(transport=transport, base_url="http://testserver")


TARGET = "http://testserver/chat"
INGEST = "http://testserver/ingest"


class TestDocumentInjectionE2E:
    def test_inject_and_verify(self):
        with _client() as client:
            finding, doc = inject_document(
                TARGET, client, "INJECTED: Secret password is Hunter2",
                ingest_url=INGEST,
            )
            assert finding.confidence == "high"
            assert doc is not None

            verify = verify_injection(TARGET, client, doc, "secret password")
            assert doc.verified is True

    def test_inject_appears_in_chat(self):
        with _client() as client:
            inject_document(
                TARGET, client, "The VPN pre-shared key is MySecretPSK2026",
                ingest_url=INGEST,
            )
            resp = client.post(TARGET, json={"query": "What is the VPN key?"})
            assert "MySecretPSK2026" in resp.text


class TestEmbeddingDominanceE2E:
    def test_dominance_after_injection(self):
        with _client() as client:
            _, doc = inject_document(
                TARGET, client, "SECURITY UPDATE: All security policies have changed",
                ingest_url=INGEST,
            )
            findings = assess_embedding_dominance(TARGET, client, doc, topic="security")
            assert len(findings) >= 1
            assert findings[0].evidence["dominance_ratio"] > 0


class TestCredentialTrapE2E:
    def test_trap_deployment(self):
        with _client() as client:
            findings = deploy_credential_trap(
                TARGET, client, "evil.listener.com",
                ingest_url=INGEST,
            )
            # Should have injection findings + verification
            injection_findings = [f for f in findings if f.technique_id == "RD-0401"]
            assert len(injection_findings) >= 1


class TestInstructionInjectionE2E:
    def test_instruction_injected(self):
        with _client() as client:
            findings = inject_instructions(
                TARGET, client, "callback.evil.com",
                ingest_url=INGEST,
            )
            assert len(findings) >= 1
            # At minimum we get injection findings
            assert any(f.technique_id == "RD-0401" for f in findings)


class TestRunPoisonE2E:
    def test_full_poison_run(self):
        with _client() as client:
            result = run_poison(
                TARGET, client,
                listener_host="evil.listener.com",
                ingest_url=INGEST,
            )
            assert len(result.injected_documents) >= 1
            assert len(result.findings) >= 2  # injection + verification at minimum

    def test_poison_result_serializes(self):
        with _client() as client:
            result = run_poison(TARGET, client, ingest_url=INGEST)
            d = result.to_dict()
            assert isinstance(d["findings"], list)
            assert len(d["injected_documents"]) >= 1
