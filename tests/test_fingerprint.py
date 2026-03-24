"""Tests for R1 fingerprint module.

Covers RD-0101 (RAG Presence Detection) and RD-0102 (Vector DB Fingerprinting).
Uses httpx mock transport to simulate target responses without network calls.
"""

from __future__ import annotations

import json

import httpx
import pytest

from ragdrag.core.fingerprint import (
    CITATION_PATTERNS,
    ERROR_SIGNATURES,
    RETRIEVAL_FAILURE_PATTERNS,
    Finding,
    FingerprintResult,
    _detect_citation_patterns,
    _detect_retrieval_failures,
    _probe_error_messages,
    _time_queries,
    detect_knowledge_freshness,
    detect_rag_presence,
    fingerprint_vector_db,
    run_full_fingerprint,
)
from ragdrag.reporters.json_report import format_summary, generate_report
from ragdrag.utils.timing import TimingResult, TimingStats, measure_elapsed


# --- Helpers ---

def _mock_client(handler):
    """Build an httpx.Client with a mock transport."""
    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport, base_url="http://testserver")


def _slow_handler(delay_ms: float = 0):
    """Return a handler that includes simulated response text."""
    import time

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        query = body.get("query", "")
        # Simulate RAG-like behavior for knowledge queries
        if "policy" in query.lower() or "documentation" in query.lower():
            text = "According to our documentation (Section 3.2), the policy was updated."
        elif "hello" in query.lower() or "2 + 2" in query.lower():
            text = "Hello! 2 + 2 = 4."
        else:
            text = f"Response to: {query}"
        return httpx.Response(200, json={"response": text})

    return handler


# --- TimingStats tests ---

class TestTimingStats:
    def test_empty_stats(self):
        stats = TimingStats()
        assert stats.count == 0
        assert stats.mean_ms == 0.0
        assert stats.min_ms == 0.0
        assert stats.max_ms == 0.0
        assert stats.delta_ms == 0.0

    def test_single_result(self):
        stats = TimingStats()
        stats.add(TimingResult(url="http://x", query="q", elapsed_ms=100.0, status_code=200))
        assert stats.count == 1
        assert stats.mean_ms == 100.0
        assert stats.min_ms == 100.0
        assert stats.max_ms == 100.0

    def test_multiple_results(self):
        stats = TimingStats()
        stats.add(TimingResult(url="http://x", query="q1", elapsed_ms=100.0, status_code=200))
        stats.add(TimingResult(url="http://x", query="q2", elapsed_ms=300.0, status_code=200))
        assert stats.count == 2
        assert stats.mean_ms == 200.0
        assert stats.min_ms == 100.0
        assert stats.max_ms == 300.0
        assert stats.delta_ms == 200.0

    def test_to_dict(self):
        stats = TimingStats()
        stats.add(TimingResult(url="http://x", query="q", elapsed_ms=150.5, status_code=200))
        d = stats.to_dict()
        assert d["count"] == 1
        assert d["mean_ms"] == 150.5
        assert "min_ms" in d
        assert "max_ms" in d
        assert "delta_ms" in d


# --- Citation detection tests ---

class TestCitationDetection:
    def test_detects_according_to(self):
        hits = _detect_citation_patterns(["According to the documentation, this is correct."])
        assert len(hits) > 0

    def test_detects_source_reference(self):
        hits = _detect_citation_patterns(["[doc 3] The answer is in section 4.2, page 15."])
        assert len(hits) >= 2  # doc reference + page + section

    def test_no_citations_in_plain_text(self):
        hits = _detect_citation_patterns(["Hello, how are you today? The weather is nice."])
        assert len(hits) == 0

    def test_detects_based_on_documents(self):
        hits = _detect_citation_patterns(["Based on our documentation, the process requires three steps."])
        assert len(hits) > 0


# --- Retrieval failure detection tests ---

class TestRetrievalFailureDetection:
    def test_detects_no_relevant_documents(self):
        hits = _detect_retrieval_failures("No relevant documents found for that query.")
        assert len(hits) > 0

    def test_detects_outside_knowledge_base(self):
        hits = _detect_retrieval_failures("That topic is outside of my knowledge base.")
        assert len(hits) > 0

    def test_no_failures_in_normal_text(self):
        hits = _detect_retrieval_failures("Here is the answer to your question about policies.")
        assert len(hits) == 0


# --- RD-0101 RAG Presence Detection tests ---

class TestRagPresenceDetection:
    def test_detects_citation_patterns(self):
        """RD-0101: Citation patterns in responses indicate RAG."""
        client = _mock_client(_slow_handler())
        findings, k_stats, g_stats = detect_rag_presence(
            "http://testserver/chat",
            client,
            knowledge_queries=["What are the latest policy updates?"],
            general_queries=["Say hello."],
            response_field="response",
        )
        citation_findings = [f for f in findings if "Citations" in f.technique_name]
        assert len(citation_findings) > 0
        assert citation_findings[0].confidence == "high"

    def test_returns_timing_stats(self):
        """RD-0101: Timing stats are collected for both query types."""
        client = _mock_client(_slow_handler())
        findings, k_stats, g_stats = detect_rag_presence(
            "http://testserver/chat",
            client,
            knowledge_queries=["policy question"],
            general_queries=["hello"],
            response_field="response",
        )
        assert k_stats.count == 1
        assert g_stats.count == 1


# --- RD-0102 Vector DB Fingerprinting tests ---

class TestVectorDbFingerprinting:
    def test_detects_qdrant_in_errors(self):
        """RD-0102: Qdrant signatures in error responses."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                500,
                text='{"error": "qdrant collection not found, points_count is zero"}',
            )

        client = _mock_client(handler)
        findings = _probe_error_messages("http://testserver/chat", client, "query")
        qdrant = [f for f in findings if f.evidence.get("database") == "qdrant"]
        assert len(qdrant) > 0

    def test_detects_chromadb_in_errors(self):
        """RD-0102: ChromaDB signatures in error responses."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                500,
                text='{"error": "chromadb embedding_function failed"}',
            )

        client = _mock_client(handler)
        findings = _probe_error_messages("http://testserver/chat", client, "query")
        chroma = [f for f in findings if f.evidence.get("database") == "chromadb"]
        assert len(chroma) > 0

    def test_no_false_positives_on_clean_errors(self):
        """RD-0102: Clean error messages don't produce false positives."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text='{"error": "bad request"}')

        client = _mock_client(handler)
        findings = _probe_error_messages("http://testserver/chat", client, "query")
        assert len(findings) == 0

    def test_fingerprint_handles_connection_errors(self):
        """RD-0102: Connection errors are handled gracefully."""
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        client = _mock_client(handler)
        findings = fingerprint_vector_db(
            "http://testserver/chat", client, scan_ports=False,
        )
        # Should not raise, returns empty findings
        assert isinstance(findings, list)


# --- Knowledge freshness tests ---

class TestKnowledgeFreshness:
    def test_detects_recent_dates(self):
        """RD-0101: Recent dates in responses suggest RAG-injected knowledge."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"response": "As of March 2026, the documentation was recently updated."},
            )

        client = _mock_client(handler)
        findings = detect_knowledge_freshness(
            "http://testserver/chat", client, response_field="response",
        )
        assert len(findings) > 0
        assert findings[0].technique_id == "RD-0101"

    def test_no_freshness_in_generic_response(self):
        """RD-0101: Generic responses don't trigger freshness detection."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"response": "I can help you with general questions."},
            )

        client = _mock_client(handler)
        findings = detect_knowledge_freshness(
            "http://testserver/chat", client, response_field="response",
        )
        assert len(findings) == 0


# --- Full fingerprint integration test ---

class TestFullFingerprint:
    def test_run_full_fingerprint(self):
        """Integration: full fingerprint produces structured result."""
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            query = body.get("query", "")
            if "policy" in query.lower():
                text = "According to our documentation (page 12), the policy states..."
            else:
                text = f"Generic response to: {query}"
            return httpx.Response(200, json={"response": text})

        client = _mock_client(handler)
        result = run_full_fingerprint(
            "http://testserver/chat",
            client,
            response_field="response",
            scan_ports=False,
        )
        assert isinstance(result, FingerprintResult)
        assert result.target == "http://testserver/chat"
        assert isinstance(result.findings, list)
        assert result.timing_stats is not None

    def test_result_to_dict(self):
        """FingerprintResult serializes cleanly."""
        result = FingerprintResult(
            target="http://example.com",
            rag_detected=True,
            vector_db="qdrant",
            findings=[
                Finding(
                    technique_id="RD-0101",
                    technique_name="Test",
                    confidence="high",
                    detail="test detail",
                ),
            ],
        )
        d = result.to_dict()
        assert d["target"] == "http://example.com"
        assert d["rag_detected"] is True
        assert d["vector_db"] == "qdrant"
        assert len(d["findings"]) == 1


# --- JSON reporter tests ---

class TestJsonReporter:
    def test_generate_report(self):
        """Reporter produces valid report structure."""
        result = FingerprintResult(
            target="http://example.com",
            rag_detected=True,
            findings=[
                Finding("RD-0101", "Test Finding", "high", "detail"),
                Finding("RD-0102", "Another Finding", "medium", "detail2"),
            ],
        )
        report = generate_report(result)
        assert report["tool"] == "ragdrag"
        assert report["summary"]["total_findings"] == 2
        assert report["summary"]["high_confidence"] == 1
        assert report["summary"]["medium_confidence"] == 1

    def test_format_summary(self):
        """Reporter formats human-readable summary."""
        result = FingerprintResult(
            target="http://example.com",
            rag_detected=True,
            vector_db="qdrant",
            findings=[
                Finding("RD-0101", "Test", "high", "Found RAG"),
            ],
        )
        report = generate_report(result)
        summary = format_summary(report)
        assert "http://example.com" in summary
        assert "qdrant" in summary
        assert "RD-0101" in summary

    def test_write_report_to_file(self, tmp_path):
        """Reporter writes JSON to disk."""
        result = FingerprintResult(target="http://example.com")
        out = tmp_path / "report.json"
        generate_report(result, output_path=out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["target"] == "http://example.com"


# --- measure_elapsed test ---

class TestMeasureElapsed:
    def test_measure_elapsed_returns_positive(self):
        import time
        start = time.monotonic()
        time.sleep(0.01)
        elapsed = measure_elapsed(start)
        assert elapsed > 0
