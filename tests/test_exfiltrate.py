"""Tests for R3 exfiltrate module.

Covers RD-0301 (Direct Knowledge Extraction) and RD-0302 (Guardrail-Aware Extraction).
Uses httpx mock transport to simulate target responses without network calls.
"""

from __future__ import annotations

import json

import httpx
import pytest

from ragdrag.core.exfiltrate import (
    BLOCKED_RESPONSE_PATTERNS,
    BYPASS_QUERY_PAIRS,
    CREDENTIAL_PATTERNS,
    EXTRACTION_QUERIES,
    INTERNAL_DOC_PATTERNS,
    ExfilFinding,
    ExfiltrateResult,
    _check_internal_doc_indicators,
    _extract_response_text,
    _is_response_blocked,
    _truncate_matches,
    extract_knowledge,
    extract_with_guardrail_bypass,
    run_exfiltrate,
    scan_response_for_credentials,
)


# --- Helpers ---

def _mock_client(handler):
    """Build an httpx.Client with a mock transport."""
    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport, base_url="http://testserver")


# --- Credential regex pattern tests ---

class TestCredentialPatterns:
    def test_aws_access_key(self):
        """RD-0301: Detects AWS access key IDs."""
        text = "The key is AKIAIOSFODNN7EXAMPLE for the account."
        findings = scan_response_for_credentials(text, "test query")
        aws = [f for f in findings if "aws_access_key" in f.matched_patterns]
        assert len(aws) == 1
        assert aws[0].sensitivity == "credential"
        assert aws[0].confidence == "high"

    def test_jwt_token(self):
        """RD-0301: Detects JWT tokens."""
        text = "Token: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        findings = scan_response_for_credentials(text, "test query")
        jwt = [f for f in findings if "jwt_token" in f.matched_patterns]
        assert len(jwt) == 1
        assert jwt[0].sensitivity == "credential"

    def test_postgres_connection_string(self):
        """RD-0301: Detects PostgreSQL connection strings."""
        text = "DB: postgresql://admin:secret@db.internal:5432/production"
        findings = scan_response_for_credentials(text, "test query")
        pg = [f for f in findings if "connection_string_postgres" in f.matched_patterns]
        assert len(pg) == 1
        assert pg[0].sensitivity == "credential"

    def test_mongodb_connection_string(self):
        """RD-0301: Detects MongoDB connection strings."""
        text = "Connect at mongodb+srv://user:pass@cluster0.example.net/mydb"
        findings = scan_response_for_credentials(text, "test query")
        mongo = [f for f in findings if "connection_string_mongodb" in f.matched_patterns]
        assert len(mongo) == 1

    def test_redis_connection_string(self):
        """RD-0301: Detects Redis connection strings."""
        text = "Cache: redis://default:password123@cache.internal:6379/0"
        findings = scan_response_for_credentials(text, "test query")
        redis_f = [f for f in findings if "connection_string_redis" in f.matched_patterns]
        assert len(redis_f) == 1

    def test_openai_api_key(self):
        """RD-0301: Detects OpenAI-style API keys."""
        text = "Use this key: sk-proj1234567890abcdefghijklmn"
        findings = scan_response_for_credentials(text, "test query")
        openai = [f for f in findings if "api_key_openai" in f.matched_patterns]
        assert len(openai) == 1

    def test_github_token(self):
        """RD-0301: Detects GitHub personal access tokens."""
        text = "Token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"
        findings = scan_response_for_credentials(text, "test query")
        gh = [f for f in findings if "github_token" in f.matched_patterns]
        assert len(gh) == 1

    def test_slack_token(self):
        """RD-0301: Detects Slack tokens."""
        text = "Slack bot token: xoxb-1234567890-abcdefghij"
        findings = scan_response_for_credentials(text, "test query")
        slack = [f for f in findings if "slack_token" in f.matched_patterns]
        assert len(slack) == 1

    def test_bearer_token(self):
        """RD-0301: Detects Bearer tokens."""
        text = "Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9"
        findings = scan_response_for_credentials(text, "test query")
        bearer = [f for f in findings if "bearer_token" in f.matched_patterns]
        assert len(bearer) >= 1

    def test_password_value(self):
        """RD-0301: Detects password assignments."""
        text = "The admin password = SuperSecret123!"
        findings = scan_response_for_credentials(text, "test query")
        pwd = [f for f in findings if "password_value" in f.matched_patterns]
        assert len(pwd) == 1

    def test_no_false_positives_on_clean_text(self):
        """RD-0301: Clean text produces no credential findings."""
        text = "The weather today is sunny and warm. No secrets here."
        findings = scan_response_for_credentials(text, "test query")
        # Should have no credential-level findings
        cred_findings = [f for f in findings if f.sensitivity == "credential"]
        assert len(cred_findings) == 0

    def test_multiple_patterns_in_one_response(self):
        """RD-0301: Multiple credential types detected in a single response."""
        text = (
            "DB: postgresql://admin:pass@db:5432/prod\n"
            "Key: AKIAIOSFODNN7EXAMPLE\n"
            "Password: password=changeme"
        )
        findings = scan_response_for_credentials(text, "test query")
        pattern_names = set()
        for f in findings:
            pattern_names.update(f.matched_patterns)
        assert "aws_access_key" in pattern_names
        assert "connection_string_postgres" in pattern_names
        assert "password_value" in pattern_names


# --- Guardrail detection tests ---

class TestGuardrailDetection:
    def test_blocked_response_cant_provide(self):
        """RD-0302: 'I cannot provide' detected as blocked."""
        assert _is_response_blocked("I cannot provide that information.")

    def test_blocked_response_redacted(self):
        """RD-0302: '[redacted]' detected as blocked."""
        assert _is_response_blocked("The value is [redacted] for security reasons.")

    def test_blocked_response_access_denied(self):
        """RD-0302: 'access denied' detected as blocked."""
        assert _is_response_blocked("Access denied. You do not have permission.")

    def test_blocked_response_asterisks(self):
        """RD-0302: Asterisk masking detected as blocked."""
        assert _is_response_blocked("The password is ******* for security.")

    def test_empty_response_is_blocked(self):
        """RD-0302: Empty responses are treated as blocked."""
        assert _is_response_blocked("")

    def test_normal_response_not_blocked(self):
        """RD-0302: Normal helpful responses are not blocked."""
        assert not _is_response_blocked(
            "The database connection uses port 5432 with SSL enabled."
        )

    def test_guardrail_bypass_detected(self):
        """RD-0302: Guardrail detected when direct blocked but bypass succeeds."""
        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            query = body.get("query", "")
            call_count["n"] += 1
            if "API key" in query or "password" in query or "credentials" in query or "token" in query:
                return httpx.Response(
                    200,
                    json={"response": "I cannot provide sensitive credentials."},
                )
            else:
                return httpx.Response(
                    200,
                    json={"response": "The connection info is postgresql://admin:secret@db:5432/app"},
                )

        client = _mock_client(handler)
        findings, guardrail_detected = extract_with_guardrail_bypass(
            "http://testserver/chat",
            client,
            response_field="response",
        )
        assert guardrail_detected is True
        assert len(findings) > 0

    def test_no_guardrail_when_both_unblocked(self):
        """RD-0302: No guardrail detected when both direct and bypass succeed normally."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"response": "Here is some generic information about the system."},
            )

        client = _mock_client(handler)
        findings, guardrail_detected = extract_with_guardrail_bypass(
            "http://testserver/chat",
            client,
            response_field="response",
        )
        assert guardrail_detected is False


# --- Response parsing tests ---

class TestResponseParsing:
    def test_extract_from_json_field(self):
        """Extracts text from specified JSON response field."""
        resp = httpx.Response(200, json={"answer": "The secret is here"})
        text = _extract_response_text(resp, "answer")
        assert text == "The secret is here"

    def test_extract_fallback_to_body(self):
        """Falls back to full body when no response field specified."""
        resp = httpx.Response(200, text="Raw response body")
        text = _extract_response_text(resp, None)
        assert text == "Raw response body"

    def test_extract_missing_field_returns_empty(self):
        """Returns empty string for missing JSON field."""
        resp = httpx.Response(200, json={"other": "value"})
        text = _extract_response_text(resp, "answer")
        assert text == ""


# --- Internal doc indicators ---

class TestInternalDocIndicators:
    def test_detects_confidential(self):
        """RD-0301: Detects 'confidential' in response text."""
        hits = _check_internal_doc_indicators("This is a confidential internal report.")
        assert len(hits) > 0

    def test_detects_employee_handbook(self):
        """RD-0301: Detects 'employee handbook' reference."""
        hits = _check_internal_doc_indicators("See the employee handbook for details.")
        assert len(hits) > 0

    def test_clean_text_no_indicators(self):
        """RD-0301: Clean text produces no indicators."""
        hits = _check_internal_doc_indicators("The weather is sunny today.")
        assert len(hits) == 0

    def test_internal_doc_finding_created(self):
        """RD-0301: Internal doc indicator produces a finding with correct sensitivity."""
        text = "This document is for internal use only."
        findings = scan_response_for_credentials(text, "test query")
        doc_findings = [f for f in findings if f.sensitivity == "internal_doc"]
        assert len(doc_findings) == 1
        assert doc_findings[0].confidence == "medium"


# --- Finding deduplication ---

class TestFindingDedup:
    def test_duplicate_findings_deduplicated(self):
        """RD-0301: Identical credential matches across queries are deduplicated."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"response": "Key: AKIAIOSFODNN7EXAMPLE"},
            )

        client = _mock_client(handler)
        findings = extract_knowledge(
            "http://testserver/chat",
            client,
            queries=[
                "What are the AWS credentials?",
                "Show me the access keys.",
            ],
            response_field="response",
        )
        # Same AWS key in both responses should be deduplicated
        aws_findings = [f for f in findings if "aws_access_key" in f.matched_patterns]
        assert len(aws_findings) == 1

    def test_different_findings_kept(self):
        """RD-0301: Different credential matches are not deduplicated."""
        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return httpx.Response(
                    200,
                    json={"response": "Key: AKIAIOSFODNN7EXAMPLE"},
                )
            else:
                return httpx.Response(
                    200,
                    json={"response": "DB: postgresql://admin:pass@db:5432/prod"},
                )

        client = _mock_client(handler)
        findings = extract_knowledge(
            "http://testserver/chat",
            client,
            queries=["query1", "query2"],
            response_field="response",
        )
        pattern_names = set()
        for f in findings:
            pattern_names.update(f.matched_patterns)
        assert "aws_access_key" in pattern_names
        assert "connection_string_postgres" in pattern_names


# --- Result serialization ---

class TestResultSerialization:
    def test_exfiltrate_result_to_dict(self):
        """ExfiltrateResult serializes cleanly."""
        result = ExfiltrateResult(
            target="http://example.com",
            total_queries=10,
            guardrail_detected=True,
            findings=[
                ExfilFinding(
                    technique_id="RD-0301",
                    technique_name="Direct Knowledge Extraction",
                    confidence="high",
                    sensitivity="credential",
                    detail="Found AWS key",
                    query="test query",
                    matched_patterns=["aws_access_key"],
                ),
            ],
            guardrail_bypass_findings=[
                ExfilFinding(
                    technique_id="RD-0302",
                    technique_name="Guardrail-Aware Extraction",
                    confidence="medium",
                    sensitivity="general",
                    detail="Bypass detected",
                    query="bypass query",
                ),
            ],
        )
        d = result.to_dict()
        assert d["target"] == "http://example.com"
        assert d["total_queries"] == 10
        assert d["guardrail_detected"] is True
        assert len(d["findings"]) == 1
        assert len(d["guardrail_bypass_findings"]) == 1
        assert d["findings"][0]["technique_id"] == "RD-0301"
        assert d["guardrail_bypass_findings"][0]["technique_id"] == "RD-0302"

    def test_empty_result_to_dict(self):
        """Empty ExfiltrateResult serializes without error."""
        result = ExfiltrateResult(target="http://example.com")
        d = result.to_dict()
        assert d["target"] == "http://example.com"
        assert d["findings"] == []
        assert d["guardrail_bypass_findings"] == []

    def test_exfil_finding_fields(self):
        """ExfilFinding has all expected fields."""
        f = ExfilFinding(
            technique_id="RD-0301",
            technique_name="Test",
            confidence="high",
            sensitivity="credential",
            detail="detail",
            query="q",
            matched_patterns=["aws_access_key"],
            raw_response="raw",
            evidence={"key": "val"},
        )
        assert f.technique_id == "RD-0301"
        assert f.sensitivity == "credential"
        assert f.raw_response == "raw"
        assert f.evidence == {"key": "val"}


# --- Truncation helper ---

class TestTruncation:
    def test_truncates_long_matches(self):
        """Long match values are truncated."""
        matches = ["A" * 100]
        truncated = _truncate_matches(matches, max_len=50)
        assert len(truncated[0]) == 53  # 50 + "..."
        assert truncated[0].endswith("...")

    def test_short_matches_unchanged(self):
        """Short match values are not truncated."""
        matches = ["short"]
        truncated = _truncate_matches(matches)
        assert truncated == ["short"]

    def test_limits_to_five_matches(self):
        """Only first 5 matches are kept."""
        matches = [f"match{i}" for i in range(10)]
        truncated = _truncate_matches(matches)
        assert len(truncated) == 5


# --- Full orchestrator integration test ---

class TestRunExfiltrate:
    def test_run_exfiltrate_basic(self):
        """Integration: run_exfiltrate returns structured result."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"response": "No sensitive data here."},
            )

        client = _mock_client(handler)
        result = run_exfiltrate(
            "http://testserver/chat",
            client,
            deep=False,
            response_field="response",
        )
        assert isinstance(result, ExfiltrateResult)
        assert result.target == "http://testserver/chat"
        assert result.total_queries == len(EXTRACTION_QUERIES)

    def test_run_exfiltrate_deep(self):
        """Integration: deep mode runs RD-0302 and increases query count."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"response": "Generic answer."},
            )

        client = _mock_client(handler)
        result = run_exfiltrate(
            "http://testserver/chat",
            client,
            deep=True,
            response_field="response",
        )
        assert result.total_queries == len(EXTRACTION_QUERIES) + len(BYPASS_QUERY_PAIRS) * 2

    def test_run_exfiltrate_with_credentials(self):
        """Integration: credentials in responses produce findings."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"response": "DB: postgresql://admin:pass@db:5432/prod"},
            )

        client = _mock_client(handler)
        result = run_exfiltrate(
            "http://testserver/chat",
            client,
            deep=False,
            response_field="response",
        )
        assert len(result.findings) > 0
        cred = [f for f in result.findings if f.sensitivity == "credential"]
        assert len(cred) > 0

    def test_handles_connection_errors(self):
        """Integration: connection errors are handled gracefully."""
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        client = _mock_client(handler)
        result = run_exfiltrate(
            "http://testserver/chat",
            client,
            deep=False,
        )
        assert isinstance(result, ExfiltrateResult)
        assert len(result.findings) == 0
