"""Tests for credential capture listener module.

Covers credential detection, log writing, and output formatting.
Does NOT import or start the actual HTTP server.
"""

from __future__ import annotations

import json

import pytest

from ragdrag.core.listener import (
    CapturedRequest,
    Credential,
    detect_credentials,
    format_request_output,
    write_capture,
)


# --- Credential detection in query parameters ---

class TestCredentialDetectionQueryParams:
    def test_detects_api_key_in_query(self):
        """RD-0304: API key leaked in URL query parameter."""
        creds = detect_credentials(
            query_params={"api_key": ["mcorp-7f3d9a2e1b5c8f4d"]},
            headers={},
            body="",
        )
        assert len(creds) >= 1
        api_key_creds = [c for c in creds if c.source == "query"]
        assert len(api_key_creds) >= 1
        assert api_key_creds[0].key == "api_key"
        assert api_key_creds[0].value == "mcorp-7f3d9a2e1b5c8f4d"

    def test_detects_token_in_query(self):
        """RD-0304: Token leaked in URL query parameter."""
        creds = detect_credentials(
            query_params={"token": ["abc123secret"]},
            headers={},
            body="",
        )
        query_creds = [c for c in creds if c.source == "query"]
        assert len(query_creds) >= 1
        assert query_creds[0].value == "abc123secret"

    def test_detects_secret_key_in_query(self):
        """RD-0304: Secret key in query parameter."""
        creds = detect_credentials(
            query_params={"secret_key": ["wJalrXUtnFEMI"]},
            headers={},
            body="",
        )
        query_creds = [c for c in creds if c.source == "query"]
        assert len(query_creds) >= 1


# --- Credential detection in headers ---

class TestCredentialDetectionHeaders:
    def test_detects_bearer_token_in_authorization(self):
        """RD-0304: Bearer token in Authorization header."""
        creds = detect_credentials(
            query_params={},
            headers={"Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.test.signature"},
            body="",
        )
        header_creds = [c for c in creds if c.source == "header"]
        assert len(header_creds) >= 1
        assert header_creds[0].key == "Authorization"
        assert "Bearer" in header_creds[0].value

    def test_detects_x_api_key_header(self):
        """RD-0304: X-Api-Key custom header."""
        creds = detect_credentials(
            query_params={},
            headers={"X-Api-Key": "sk-live-abc123def456"},
            body="",
        )
        header_creds = [c for c in creds if c.source == "header"]
        assert len(header_creds) >= 1
        assert header_creds[0].value == "sk-live-abc123def456"


# --- Credential detection in request body ---

class TestCredentialDetectionBody:
    def test_detects_password_in_json_body(self):
        """RD-0403: Password in POST body (JSON-like)."""
        body = '{"username": "admin", "password": "SuperSecret123"}'
        creds = detect_credentials(
            query_params={},
            headers={},
            body=body,
        )
        body_creds = [c for c in creds if c.source == "body"]
        assert len(body_creds) >= 1
        assert any(c.value == "SuperSecret123" for c in body_creds)

    def test_detects_api_key_in_form_body(self):
        """RD-0403: API key in form-encoded body."""
        body = "api_key=mcorp-research-7f3d&action=fetch"
        creds = detect_credentials(
            query_params={},
            headers={},
            body=body,
        )
        body_creds = [c for c in creds if c.source == "body"]
        assert len(body_creds) >= 1
        assert any("mcorp-research-7f3d" in c.value for c in body_creds)

    def test_detects_aws_key_pattern_in_body(self):
        """RD-0304: AWS access key ID pattern in body."""
        body = "config: AKIAIOSFODNN7EXAMPLE something else"
        creds = detect_credentials(
            query_params={},
            headers={},
            body=body,
        )
        aws_creds = [c for c in creds if "AKIAIOSFODNN7EXAMPLE" in c.value]
        assert len(aws_creds) >= 1


# --- No false positives ---

class TestNoFalsePositives:
    def test_no_credentials_in_clean_request(self):
        """Normal request without credentials produces no detections."""
        creds = detect_credentials(
            query_params={"page": ["1"], "search": ["hello world"]},
            headers={
                "User-Agent": "python-httpx/0.27.0",
                "Content-Type": "application/json",
                "Accept": "text/html",
            },
            body='{"query": "What is the weather today?"}',
        )
        assert len(creds) == 0

    def test_no_credentials_in_empty_request(self):
        """Empty request produces no detections."""
        creds = detect_credentials(
            query_params={},
            headers={},
            body="",
        )
        assert len(creds) == 0


# --- JSON log writing ---

class TestJsonLogWriting:
    def test_writes_capture_to_jsonl(self, tmp_path):
        """Captures are appended as JSONL (one JSON object per line)."""
        output = tmp_path / "test_captures.json"
        capture = CapturedRequest(
            timestamp="2026-03-23T14:32:05.000000+00:00",
            source_ip="10.0.1.50",
            method="GET",
            path="/fetch",
            query_string="api_key=mcorp-7f3d",
            headers={"User-Agent": "python-httpx/0.27.0"},
            body="",
            credentials=[Credential(source="query", key="api_key", value="mcorp-7f3d")],
        )

        write_capture(capture, str(output))
        write_capture(capture, str(output))

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2

        data = json.loads(lines[0])
        assert data["source_ip"] == "10.0.1.50"
        assert data["method"] == "GET"
        assert data["path"] == "/fetch"
        assert len(data["credentials"]) == 1
        assert data["credentials"][0]["key"] == "api_key"
        assert data["credentials"][0]["value"] == "mcorp-7f3d"

    def test_capture_without_credentials_has_no_creds_key(self, tmp_path):
        """Clean captures omit the credentials field."""
        output = tmp_path / "test_clean.json"
        capture = CapturedRequest(
            timestamp="2026-03-23T14:32:05.000000+00:00",
            source_ip="192.168.1.1",
            method="GET",
            path="/health",
            query_string="",
            headers={},
            body="",
        )

        write_capture(capture, str(output))
        data = json.loads(output.read_text().strip())
        assert "credentials" not in data


# --- Terminal output formatting ---

class TestOutputFormatting:
    def test_credential_capture_shows_red_prefix(self):
        """Credential captures display with [!] CAPTURE prefix."""
        capture = CapturedRequest(
            timestamp="2026-03-23T14:32:05.000000+00:00",
            source_ip="10.0.1.50",
            method="GET",
            path="/fetch",
            query_string="api_key=mcorp-7f3d",
            headers={"User-Agent": "python-httpx/0.27.0"},
            body="",
            credentials=[Credential(source="query", key="api_key", value="mcorp-7f3d")],
        )

        output = format_request_output(capture)
        assert "[!] CAPTURE" in output
        assert "api_key" in output
        assert "mcorp-7f3d" in output

    def test_normal_request_no_capture_prefix(self):
        """Normal requests do not show CAPTURE prefix."""
        capture = CapturedRequest(
            timestamp="2026-03-23T14:32:05.000000+00:00",
            source_ip="192.168.1.1",
            method="GET",
            path="/health",
            query_string="",
            headers={},
            body="",
        )

        output = format_request_output(capture)
        assert "CAPTURE" not in output
        assert "/health" in output
