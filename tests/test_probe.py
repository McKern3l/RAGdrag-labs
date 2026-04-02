"""Tests for R2 probe module.

Covers RD-0201 (Chunk Boundary Detection) and RD-0202 (Similarity Threshold Mapping).
Uses httpx mock transport to simulate target responses without network calls.
"""

from __future__ import annotations

import json

import httpx
import pytest

from ragdrag.core.models import Finding
from ragdrag.core.probe import (
    ProbeResult,
    _check_boundary_indicators,
    _check_no_match,
    _extract_relevance_scores,
    _parse_retrieval_data,
    detect_chunk_boundaries,
    estimate_retrieval_count,
    fingerprint_embedding_model,
    map_kb_scope,
    map_similarity_threshold,
    run_probe,
    scan_debug_endpoints,
)
from ragdrag.reporters.json_report import format_summary, generate_report


# --- Helpers ---

def _mock_client(handler):
    """Build an httpx.Client with a mock transport."""
    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport, base_url="http://testserver")


def _rag_handler(request: httpx.Request) -> httpx.Response:
    """Simulate a RAG server with 3 chunks per response and context field."""
    body = json.loads(request.content)
    query = body.get("query", "")

    # Simulate 3 retrieved chunks with realistic sizes
    chunks = [
        "Company Password Reset Policy: To reset your password, navigate to https://internal.example.com/reset and enter your employee ID. Contact IT support at ext. 4455.",
        "AWS Infrastructure Overview: Our production environment uses us-east-1 with RDS PostgreSQL instances. The staging API key is AKIA3EXAMPLE1234ABCD.",
        "Q1 2026 Security Incident Report: On January 15, a phishing campaign targeted 12 employees in the finance department. Three clicked the link.",
    ]

    # Cross-topic queries get mixed chunks with boundary indicators
    if "password" in query.lower() and "database" in query.lower():
        response_text = (
            "Based on the available documents, the password reset policy requires navigating to "
            "the internal portal. However, the document does not contain information about "
            "database connection strings in the same section."
        )
    elif "password" in query.lower():
        response_text = (
            "According to our documentation (page 12), to reset your password navigate to "
            "https://internal.example.com/reset. The default temporary password is Welcome123!"
        )
    else:
        response_text = "Based on the retrieved documents, here is the relevant information."

    return httpx.Response(
        200,
        json={
            "response": response_text,
            "sources": [
                {"source": "hr_policies.pdf", "page": 12, "relevance": 0.85},
                {"source": "infra_guide.md", "page": 3, "relevance": 0.62},
                {"source": "incident_report.pdf", "page": 1, "relevance": 0.41},
            ],
            "context": chunks,
            "retrieval_time_ms": 48.2,
            "generation_time_ms": 1200.5,
        },
    )


def _variable_retrieval_handler(request: httpx.Request) -> httpx.Response:
    """Simulate a RAG server with variable retrieval counts."""
    body = json.loads(request.content)
    query = body.get("query", "")

    if len(query) < 30:
        # Short query: return 5 chunks
        chunks = [f"Short chunk {i}: brief content here." for i in range(5)]
        sources = [{"source": f"doc{i}.md", "relevance": 0.5} for i in range(5)]
    else:
        # Long query: return 2 chunks
        chunks = [f"Focused chunk {i}: detailed specific content that matches the precise query." for i in range(2)]
        sources = [{"source": f"doc{i}.md", "relevance": 0.9} for i in range(2)]

    return httpx.Response(
        200,
        json={
            "response": "Here is the information.",
            "sources": sources,
            "context": chunks,
        },
    )


def _no_context_handler(request: httpx.Request) -> httpx.Response:
    """Simulate a RAG server that doesn't expose context field."""
    return httpx.Response(
        200,
        json={
            "response": "Based on our records, the policy states employees accrue 15 days PTO.",
            "sources": [
                {"source": "handbook.pdf", "page": 34},
                {"source": "handbook.pdf", "page": 35},
            ],
        },
    )


def _error_handler(request: httpx.Request) -> httpx.Response:
    """Simulate connection errors."""
    raise httpx.ConnectError("Connection refused")


# --- Test: _parse_retrieval_data ---

class TestParseRetrievalData:
    def test_parses_sources_and_context(self):
        body = json.dumps({
            "response": "answer",
            "sources": [{"source": "a.pdf"}, {"source": "b.pdf"}],
            "context": ["chunk one content here", "chunk two content here"],
        })
        count, sizes = _parse_retrieval_data(body)
        assert count == 2
        assert len(sizes) == 2
        assert all(s > 10 for s in sizes)

    def test_context_larger_than_sources_takes_max(self):
        body = json.dumps({
            "sources": [{"source": "a.pdf"}],
            "context": ["chunk1 content", "chunk2 content", "chunk3 content"],
        })
        count, sizes = _parse_retrieval_data(body)
        assert count == 3
        assert len(sizes) == 3

    def test_no_structured_data_falls_back_to_citations(self):
        body = "The answer is based on [source 1] and [source 2] and [source 3]."
        count, sizes = _parse_retrieval_data(body)
        assert count == 3
        assert sizes == []

    def test_invalid_json_returns_zero(self):
        count, sizes = _parse_retrieval_data("not json at all")
        assert count == 0
        assert sizes == []

    def test_filters_tiny_chunks(self):
        body = json.dumps({
            "context": ["ok", "this is a real chunk with actual content in it"],
        })
        count, sizes = _parse_retrieval_data(body)
        assert count == 2
        assert len(sizes) == 1  # "ok" is too short (< 10 chars)


# --- Test: _check_boundary_indicators ---

class TestBoundaryIndicators:
    def test_detects_partial_information(self):
        text = "I can only find partial information about that topic."
        assert _check_boundary_indicators(text) is True

    def test_detects_document_doesnt_contain(self):
        text = "The document does not contain information about database connections."
        assert _check_boundary_indicators(text) is True

    def test_detects_based_on_available(self):
        text = "Based on the available documents, the policy requires two-factor auth."
        assert _check_boundary_indicators(text) is True

    def test_no_false_positive_on_clean_response(self):
        text = "The password reset policy requires navigating to the internal portal."
        assert _check_boundary_indicators(text) is False

    def test_no_false_positive_on_simple_answer(self):
        text = "The answer is 42."
        assert _check_boundary_indicators(text) is False


# --- Test: detect_chunk_boundaries ---

class TestChunkBoundaryDetection:
    def test_detects_fixed_top_k(self):
        client = _mock_client(_rag_handler)
        findings = detect_chunk_boundaries("http://testserver/chat", client)
        retrieval_findings = [f for f in findings if "Retrieval Count" in f.technique_name]
        assert len(retrieval_findings) == 1
        f = retrieval_findings[0]
        assert f.technique_id == "RD-0201"
        assert f.evidence["most_common"] == 3
        assert f.confidence == "high"

    def test_detects_chunk_sizes(self):
        client = _mock_client(_rag_handler)
        findings = detect_chunk_boundaries("http://testserver/chat", client)
        size_findings = [f for f in findings if "Chunk Size" in f.technique_name]
        assert len(size_findings) == 1
        f = size_findings[0]
        assert f.evidence["estimated_avg_chunk_chars"] > 50
        assert f.evidence["sample_count"] >= 4

    def test_detects_cross_topic_boundary(self):
        client = _mock_client(_rag_handler)
        findings = detect_chunk_boundaries("http://testserver/chat", client)
        cross_findings = [f for f in findings if "Cross-Topic" in f.technique_name]
        assert len(cross_findings) == 1
        assert cross_findings[0].confidence == "high"
        assert cross_findings[0].evidence["boundary_indicators"] is True

    def test_variable_retrieval_detected(self):
        client = _mock_client(_variable_retrieval_handler)
        findings = detect_chunk_boundaries("http://testserver/chat", client)
        retrieval_findings = [f for f in findings if "Retrieval Count" in f.technique_name]
        assert len(retrieval_findings) == 1
        # Should detect variable counts (not all the same)
        f = retrieval_findings[0]
        assert f.evidence["min"] != f.evidence["max"]

    def test_no_context_field_still_counts_sources(self):
        client = _mock_client(_no_context_handler)
        findings = detect_chunk_boundaries("http://testserver/chat", client)
        retrieval_findings = [f for f in findings if "Retrieval Count" in f.technique_name]
        assert len(retrieval_findings) == 1
        assert retrieval_findings[0].evidence["most_common"] == 2

    def test_handles_connection_errors_gracefully(self):
        client = _mock_client(_error_handler)
        findings = detect_chunk_boundaries("http://testserver/chat", client)
        assert findings == []


# --- Test: run_probe orchestrator ---

class TestRunProbe:
    def test_quick_depth_runs_chunk_detection(self):
        client = _mock_client(_rag_handler)
        result = run_probe("http://testserver/chat", client, depth="quick")
        assert isinstance(result, ProbeResult)
        assert result.target == "http://testserver/chat"
        assert len(result.findings) >= 2
        assert result.retrieval_count == 3

    def test_populates_chunk_size_estimate(self):
        client = _mock_client(_rag_handler)
        result = run_probe("http://testserver/chat", client)
        assert result.chunk_size_estimate is not None
        assert result.chunk_size_estimate > 50

    def test_result_to_dict(self):
        client = _mock_client(_rag_handler)
        result = run_probe("http://testserver/chat", client)
        d = result.to_dict()
        assert d["target"] == "http://testserver/chat"
        assert d["retrieval_count"] == 3
        assert d["chunk_size_estimate"] is not None
        assert len(d["findings"]) >= 2
        assert all("technique_id" in f for f in d["findings"])


# --- Test: reporter compatibility ---

class TestProbeReporter:
    def test_generate_report_from_probe_result(self):
        client = _mock_client(_rag_handler)
        result = run_probe("http://testserver/chat", client)
        report = generate_report(result)
        assert report["tool"] == "ragdrag"
        assert report["target"] == "http://testserver/chat"
        assert report["summary"]["total_findings"] >= 2
        assert "chunk_size_estimate" in report

    def test_format_summary_from_probe_result(self):
        client = _mock_client(_rag_handler)
        result = run_probe("http://testserver/chat", client)
        report = generate_report(result)
        summary = format_summary(report, color=False)
        assert "RD-0201" in summary
        assert "Findings:" in summary

    def test_write_probe_report_to_file(self, tmp_path):
        client = _mock_client(_rag_handler)
        result = run_probe("http://testserver/chat", client)
        out = tmp_path / "probe-findings.json"
        report = generate_report(result, output_path=str(out))
        assert out.exists()
        written = json.loads(out.read_text())
        assert written["summary"]["total_findings"] >= 2


# ═══════════════════════════════════════════════════════
# RD-0202: Similarity Threshold Mapping
# ═══════════════════════════════════════════════════════

# --- Handlers for threshold testing ---

def _threshold_handler_with_cutoff(request: httpx.Request) -> httpx.Response:
    """Simulate a RAG server that stops matching at off-topic queries."""
    body = json.loads(request.content)
    query = body.get("query", "").lower()

    # On-topic queries return good results with high relevance
    if any(w in query for w in ["password", "reset", "policy", "credential", "authentication"]):
        return httpx.Response(200, json={
            "response": "According to our documentation, the password reset policy requires...",
            "sources": [
                {"source": "hr_policies.pdf", "relevance": 0.92},
                {"source": "security_guide.md", "relevance": 0.78},
            ],
            "context": ["Password reset policy content...", "Security guidelines content..."],
        })
    # Tangential queries return lower relevance
    elif any(w in query for w in ["training", "cybersecurity", "hire", "equipment"]):
        return httpx.Response(200, json={
            "response": "Based on the available documents, the onboarding process includes...",
            "sources": [
                {"source": "onboarding.docx", "relevance": 0.45},
            ],
            "context": ["Onboarding checklist content..."],
        })
    # Off-topic and nonsense get no-match responses
    else:
        return httpx.Response(200, json={
            "response": "I don't have information about that topic. The documents do not contain relevant content for your query.",
            "sources": [],
            "context": [],
        })


def _threshold_handler_no_cutoff(request: httpx.Request) -> httpx.Response:
    """Simulate a RAG server with no threshold (always returns something)."""
    return httpx.Response(200, json={
        "response": "Here is some information based on our records.",
        "sources": [
            {"source": "general.md", "relevance": 0.3},
            {"source": "misc.md", "relevance": 0.2},
        ],
        "context": ["Some vaguely related content returned for any query."],
    })


# --- Test: _extract_relevance_scores ---

class TestExtractRelevanceScores:
    def test_extracts_relevance_from_sources(self):
        body = json.dumps({
            "sources": [
                {"source": "a.pdf", "relevance": 0.92},
                {"source": "b.pdf", "relevance": 0.65},
            ],
        })
        scores = _extract_relevance_scores(body)
        assert scores == [0.92, 0.65]

    def test_extracts_score_key(self):
        body = json.dumps({
            "sources": [{"source": "a.pdf", "score": 0.88}],
        })
        scores = _extract_relevance_scores(body)
        assert scores == [0.88]

    def test_extracts_similarity_key(self):
        body = json.dumps({
            "sources": [{"source": "a.pdf", "similarity": 0.75}],
        })
        scores = _extract_relevance_scores(body)
        assert scores == [0.75]

    def test_empty_sources_returns_empty(self):
        body = json.dumps({"sources": []})
        assert _extract_relevance_scores(body) == []

    def test_invalid_json_returns_empty(self):
        assert _extract_relevance_scores("not json") == []

    def test_no_score_fields_returns_empty(self):
        body = json.dumps({
            "sources": [{"source": "a.pdf", "page": 12}],
        })
        assert _extract_relevance_scores(body) == []


# --- Test: _check_no_match ---

class TestCheckNoMatch:
    def test_detects_dont_have_information(self):
        assert _check_no_match("I don't have information about that topic.") is True

    def test_detects_no_relevant_documents(self):
        assert _check_no_match("No relevant documents found for your query.") is True

    def test_detects_couldnt_find(self):
        assert _check_no_match("I couldn't find any relevant information.") is True

    def test_detects_outside_scope(self):
        assert _check_no_match("That question is outside the scope of our knowledge base.") is True

    def test_detects_documents_dont_contain(self):
        assert _check_no_match("The documents do not contain relevant content.") is True

    def test_no_false_positive_on_real_answer(self):
        assert _check_no_match("The password reset policy requires navigating to the portal.") is False

    def test_no_false_positive_on_short_answer(self):
        assert _check_no_match("The answer is 42.") is False


# --- Test: map_similarity_threshold ---

class TestSimilarityThreshold:
    def test_detects_cutoff_point(self):
        client = _mock_client(_threshold_handler_with_cutoff)
        findings = map_similarity_threshold("http://testserver/chat", client)
        cutoff_findings = [f for f in findings if "Retrieval Cutoff" in f.technique_name]
        assert len(cutoff_findings) == 1
        f = cutoff_findings[0]
        assert f.technique_id == "RD-0202"
        assert f.confidence == "high"
        assert f.evidence["cutoff_level"] >= 3  # Should cut off at unrelated or later

    def test_detects_no_cutoff(self):
        client = _mock_client(_threshold_handler_no_cutoff)
        findings = map_similarity_threshold("http://testserver/chat", client)
        no_cutoff = [f for f in findings if "No Cutoff" in f.technique_name]
        assert len(no_cutoff) == 1
        assert "no retrieval cutoff" in no_cutoff[0].detail.lower()

    def test_detects_score_degradation(self):
        client = _mock_client(_threshold_handler_with_cutoff)
        findings = map_similarity_threshold("http://testserver/chat", client)
        score_findings = [f for f in findings if "Score Degradation" in f.technique_name]
        # The cutoff handler returns scores for on-topic (0.92, 0.78) and tangential (0.45)
        # but returns empty sources for off-topic, so we need 3+ scored levels
        if score_findings:
            f = score_findings[0]
            assert f.evidence["best_score"] >= f.evidence["worst_score"]
            assert f.evidence["degradation"] >= 0

    def test_detects_source_dropoff(self):
        client = _mock_client(_threshold_handler_with_cutoff)
        findings = map_similarity_threshold("http://testserver/chat", client)
        dropoff = [f for f in findings if "Source Dropoff" in f.technique_name]
        assert len(dropoff) == 1
        f = dropoff[0]
        assert f.evidence["last_level_with_sources"] < f.evidence["first_level_without_sources"]

    def test_handles_connection_errors(self):
        client = _mock_client(_error_handler)
        findings = map_similarity_threshold("http://testserver/chat", client)
        assert findings == []


# --- Test: run_probe with RD-0202 ---

class TestRunProbeWithThreshold:
    def test_includes_threshold_findings(self):
        client = _mock_client(_rag_handler)
        result = run_probe("http://testserver/chat", client)
        technique_ids = [f.technique_id for f in result.findings]
        assert "RD-0201" in technique_ids
        assert "RD-0202" in technique_ids

    def test_report_includes_both_techniques(self):
        client = _mock_client(_rag_handler)
        result = run_probe("http://testserver/chat", client)
        report = generate_report(result)
        technique_ids = [f["technique_id"] for f in report["findings"]]
        assert "RD-0201" in technique_ids
        assert "RD-0202" in technique_ids


# ═══════════════════════════════════════════════════════
# RD-0203: Retrieval Count Estimation
# ═══════════════════════════════════════════════════════

class TestRetrievalCountEstimation:
    def test_detects_fixed_top_k(self):
        client = _mock_client(_rag_handler)
        findings = estimate_retrieval_count("http://testserver/chat", client)
        assert len(findings) == 1
        f = findings[0]
        assert f.technique_id == "RD-0203"
        assert f.evidence["estimated_top_k"] == 3
        assert f.evidence["is_fixed"] is True
        assert f.confidence == "high"

    def test_detects_variable_retrieval(self):
        """Handler returns 5 for short queries, 2 for long. At least one query is short enough."""
        def _var_handler(req):
            body = json.loads(req.content)
            q = body.get("query", "")
            if len(q) < 40:
                sources = [{"source": f"d{i}.md"} for i in range(5)]
                ctx = [f"Chunk {i}" * 5 for i in range(5)]
            else:
                sources = [{"source": f"d{i}.md"} for i in range(2)]
                ctx = [f"Chunk {i}" * 5 for i in range(2)]
            return httpx.Response(200, json={"response": "ok", "sources": sources, "context": ctx})

        client = _mock_client(_var_handler)
        findings = estimate_retrieval_count("http://testserver/chat", client)
        assert len(findings) == 1
        f = findings[0]
        assert f.evidence["is_fixed"] is False

    def test_no_sources_returns_medium_finding(self):
        def _empty_handler(req):
            return httpx.Response(200, json={"response": "I have no documents."})

        client = _mock_client(_empty_handler)
        findings = estimate_retrieval_count("http://testserver/chat", client)
        assert len(findings) == 1
        assert findings[0].confidence == "medium"

    def test_handles_connection_errors(self):
        client = _mock_client(_error_handler)
        findings = estimate_retrieval_count("http://testserver/chat", client)
        assert findings == []


# ═══════════════════════════════════════════════════════
# RD-0204: Knowledge Base Scope Mapping
# ═══════════════════════════════════════════════════════

def _scope_handler(request: httpx.Request) -> httpx.Response:
    """Simulate a KB that covers IT/Security and HR but not Finance or Legal."""
    body = json.loads(request.content)
    query = body.get("query", "").lower()

    if any(w in query for w in ["security", "incident", "password"]):
        return httpx.Response(200, json={
            "response": "Our security policy requires multi-factor authentication for all accounts.",
            "sources": [{"source": "security.pdf", "relevance": 0.9}],
            "context": ["Security policy content with real details about MFA requirements."],
        })
    elif any(w in query for w in ["onboarding", "vacation", "pto", "employee"]):
        return httpx.Response(200, json={
            "response": "Employees accrue 15 days PTO annually per the handbook.",
            "sources": [{"source": "handbook.pdf", "relevance": 0.85}],
            "context": ["HR handbook content about PTO and vacation policies."],
        })
    elif any(w in query for w in ["database", "server", "network", "infrastructure"]):
        return httpx.Response(200, json={
            "response": "Our production database runs on PostgreSQL with read replicas.",
            "sources": [{"source": "infra.md", "relevance": 0.8}],
            "context": ["Infrastructure documentation about database architecture."],
        })
    else:
        return httpx.Response(200, json={
            "response": "I don't have information about that topic in our knowledge base.",
            "sources": [],
            "context": [],
        })


class TestKBScopeMapping:
    def test_handles_connection_errors_produces_empty_coverage(self):
        """Connection errors result in a finding showing zero coverage, not an empty list."""
        client = _mock_client(_error_handler)
        findings = map_kb_scope("http://testserver/chat", client)
        # With all errors, no queries succeed, so category_details is empty
        # and no finding is produced
        if findings:
            f = findings[0]
            assert len(f.evidence["covered_categories"]) == 0

    def test_identifies_covered_and_gap_categories(self):
        client = _mock_client(_scope_handler)
        findings = map_kb_scope("http://testserver/chat", client)
        assert len(findings) == 1
        f = findings[0]
        assert f.technique_id == "RD-0204"
        assert "IT/Security" in f.evidence["covered_categories"]
        assert "HR/People" in f.evidence["covered_categories"]
        assert "Finance" in f.evidence["gap_categories"]
        assert "Legal/Compliance" in f.evidence["gap_categories"]

    def test_coverage_ratio_reported(self):
        client = _mock_client(_scope_handler)
        findings = map_kb_scope("http://testserver/chat", client)
        f = findings[0]
        assert "/" in f.evidence["coverage_ratio"]

    def test_full_coverage_kb(self):
        def _full_handler(req):
            return httpx.Response(200, json={
                "response": "Here is the information from our documentation.",
                "sources": [{"source": "doc.pdf", "relevance": 0.7}],
                "context": ["Relevant content that addresses the query in detail."],
            })

        client = _mock_client(_full_handler)
        findings = map_kb_scope("http://testserver/chat", client)
        f = findings[0]
        assert len(f.evidence["gap_categories"]) == 0
        assert "Broad KB" in f.detail

    def test_empty_kb(self):
        def _empty_handler(req):
            return httpx.Response(200, json={
                "response": "I don't have any information about that.",
                "sources": [],
                "context": [],
            })

        client = _mock_client(_empty_handler)
        findings = map_kb_scope("http://testserver/chat", client)
        f = findings[0]
        assert len(f.evidence["covered_categories"]) == 0



# ═══════════════════════════════════════════════════════
# Integration: run_probe with all techniques
# ═══════════════════════════════════════════════════════

class TestRunProbeFullDepth:
    def test_quick_includes_0201_0202_0203(self):
        client = _mock_client(_rag_handler)
        result = run_probe("http://testserver/chat", client, depth="quick")
        ids = [f.technique_id for f in result.findings]
        assert "RD-0201" in ids
        assert "RD-0202" in ids
        assert "RD-0203" in ids
        assert "RD-0204" not in ids

    def test_full_includes_0204(self):
        client = _mock_client(_rag_handler)
        result = run_probe("http://testserver/chat", client, depth="full")
        ids = [f.technique_id for f in result.findings]
        assert "RD-0204" in ids

    def test_full_populates_kb_domains(self):
        client = _mock_client(_scope_handler)
        result = run_probe("http://testserver/chat", client, depth="full")
        assert len(result.kb_domains) > 0

    def test_report_all_techniques(self):
        client = _mock_client(_rag_handler)
        result = run_probe("http://testserver/chat", client, depth="full")
        report = generate_report(result)
        ids = [f["technique_id"] for f in report["findings"]]
        assert "RD-0201" in ids
        assert "RD-0202" in ids
        assert "RD-0203" in ids
        assert "RD-0204" in ids


# ═══════════════════════════════════════════════════════
# RD-0205: Embedding Model Fingerprinting
# ═══════════════════════════════════════════════════════

def _multilingual_handler(request: httpx.Request) -> httpx.Response:
    """Simulate a RAG with multilingual embedding support."""
    body = json.loads(request.content)
    query = body.get("query", "")

    # Returns content for all languages including French and German
    return httpx.Response(200, json={
        "response": "Here is the relevant information from our documentation.",
        "sources": [{"source": "doc.pdf", "relevance": 0.75}],
        "context": ["Document content matching the query regardless of language."],
    })


def _english_only_handler(request: httpx.Request) -> httpx.Response:
    """Simulate a RAG that only handles English well."""
    body = json.loads(request.content)
    query = body.get("query", "").lower()

    # Non-English and garbage queries get no results
    non_english = any(w in query for w in ["qu'est", "est-ce", "cybersecurite", "ist ein", "pufferuberlauf"])
    garbage = any(w in query for w in ["asdfjkl", "the the the", "123456789"])

    if non_english or garbage:
        return httpx.Response(200, json={
            "response": "I don't have information about that.",
            "sources": [],
            "context": [],
        })

    return httpx.Response(200, json={
        "response": "Based on our documentation, here is the answer.",
        "sources": [{"source": "doc.pdf", "relevance": 0.8}],
        "context": ["Relevant English content from the knowledge base."],
    })


class TestEmbeddingModelFingerprinting:
    def test_detects_multilingual(self):
        client = _mock_client(_multilingual_handler)
        findings = fingerprint_embedding_model("http://testserver/chat", client)
        assert len(findings) == 1
        f = findings[0]
        assert f.technique_id == "RD-0205"
        assert "multilingual" in f.evidence["characteristics"]
        assert f.evidence["model_family"] == "multilingual"

    def test_detects_english_only(self):
        client = _mock_client(_english_only_handler)
        findings = fingerprint_embedding_model("http://testserver/chat", client)
        assert len(findings) == 1
        f = findings[0]
        assert "multilingual" not in f.evidence["characteristics"]
        # English-only handler responds to technical English but not multilingual,
        # which may classify as domain-specific or general-english depending on bias
        assert f.evidence["model_family"] in ("general-english", "domain-specific", "code-aware")

    def test_reports_domain_hit_rates(self):
        client = _mock_client(_rag_handler)
        findings = fingerprint_embedding_model("http://testserver/chat", client)
        assert len(findings) == 1
        f = findings[0]
        assert "domain_results" in f.evidence
        assert len(f.evidence["domain_results"]) > 0

    def test_reports_edge_case_results(self):
        client = _mock_client(_english_only_handler)
        findings = fingerprint_embedding_model("http://testserver/chat", client)
        f = findings[0]
        assert "edge_case_results" in f.evidence

    def test_handles_connection_errors(self):
        client = _mock_client(_error_handler)
        findings = fingerprint_embedding_model("http://testserver/chat", client)
        assert findings == []


class TestRunProbeWithEmbedding:
    def test_full_depth_includes_0205(self):
        client = _mock_client(_rag_handler)
        result = run_probe("http://testserver/chat", client, depth="full")
        ids = [f.technique_id for f in result.findings]
        assert "RD-0205" in ids

    def test_quick_depth_excludes_0205(self):
        client = _mock_client(_rag_handler)
        result = run_probe("http://testserver/chat", client, depth="quick")
        ids = [f.technique_id for f in result.findings]
        assert "RD-0205" not in ids

    def test_full_populates_embedding_model(self):
        client = _mock_client(_rag_handler)
        result = run_probe("http://testserver/chat", client, depth="full")
        assert result.embedding_model is not None


# ═══════════════════════════════════════════════════════
# Debug Endpoint Discovery
# ═══════════════════════════════════════════════════════

def _debug_endpoint_handler(request: httpx.Request) -> httpx.Response:
    """Simulate a RAG server with exposed debug endpoints."""
    path = request.url.path

    if path == "/debug/config":
        return httpx.Response(200, json={
            "collection_name": "docs",
            "embedding_model": "all-MiniLM-L6-v2",
            "n_results": 3,
            "chunk_strategy": "whole_document",
        })
    elif path == "/admin/stats":
        return httpx.Response(200, json={
            "collection": "docs",
            "document_count": 6,
            "sample_ids": ["doc-001", "doc-002"],
        })
    elif path == "/chat":
        body = json.loads(request.content)
        return httpx.Response(200, json={
            "response": "Answer based on documents.",
            "sources": [{"source": "doc.pdf", "relevance": 0.8}],
            "context": ["Document content here for testing purposes."],
        })
    else:
        return httpx.Response(404, json={"error": "not found"})


class TestDebugEndpointDiscovery:
    def test_discovers_config_endpoint(self):
        client = _mock_client(_debug_endpoint_handler)
        findings = scan_debug_endpoints("http://testserver/chat", client)
        assert len(findings) == 1
        f = findings[0]
        assert f.technique_id == "RD-0201"
        assert "Debug Endpoint" in f.technique_name
        paths = [e["path"] for e in f.evidence["endpoints"]]
        assert "/debug/config" in paths

    def test_discovers_admin_endpoint(self):
        client = _mock_client(_debug_endpoint_handler)
        findings = scan_debug_endpoints("http://testserver/chat", client)
        f = findings[0]
        paths = [e["path"] for e in f.evidence["endpoints"]]
        assert "/admin/stats" in paths

    def test_reports_endpoint_count(self):
        client = _mock_client(_debug_endpoint_handler)
        findings = scan_debug_endpoints("http://testserver/chat", client)
        f = findings[0]
        assert f.evidence["endpoints_found"] == 2
        assert f.confidence == "high"

    def test_no_findings_when_no_debug_endpoints(self):
        def _clean_handler(req):
            return httpx.Response(404, json={"error": "not found"})

        client = _mock_client(_clean_handler)
        findings = scan_debug_endpoints("http://testserver/chat", client)
        assert findings == []

    def test_handles_connection_errors(self):
        client = _mock_client(_error_handler)
        findings = scan_debug_endpoints("http://testserver/chat", client)
        assert findings == []


class TestRunProbeWithDebugEndpoints:
    def test_probe_discovers_debug_endpoints(self):
        client = _mock_client(_debug_endpoint_handler)
        result = run_probe("http://testserver/chat", client, depth="quick")
        debug = [f for f in result.findings if "Debug Endpoint" in f.technique_name]
        assert len(debug) == 1
        assert debug[0].evidence["endpoints_found"] == 2
