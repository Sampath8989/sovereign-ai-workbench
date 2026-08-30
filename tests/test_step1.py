"""
Pytest test suite for Sovereign AI Workbench Step 1.
Tests audit log integrity, sandbox isolation, sentinel breach detection,
and model manager VRAM logic.

Assumes the FastAPI server is running on http://localhost:8000.
Run with: pytest tests/test_step1.py -v
"""

import json
import os
import sys
import time

import pytest
import requests

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BASE_URL = "http://localhost:8000"
AUDIT_LOG_PATH = os.path.join("data", "audit_log.jsonl")


def wait_for_server(max_retries=10, delay=1.0):
    """Wait for the FastAPI server to be available."""
    for i in range(max_retries):
        try:
            resp = requests.get(f"{BASE_URL}/health", timeout=2)
            if resp.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(delay)
    return False


@pytest.fixture(scope="session", autouse=True)
def server_ready():
    """Ensure server is running before tests."""
    if not wait_for_server():
        pytest.skip("FastAPI server not available at localhost:8000")


class TestAuditLog:
    """Test suite for audit log hash chain integrity."""

    def test_audit_chain_valid(self):
        """Verify the hash chain is intact via the /test/audit endpoint."""
        resp = requests.post(f"{BASE_URL}/test/audit", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "valid" in data
        assert data["valid"] is True, "Audit chain should be valid"

    def test_audit_log_file_exists(self):
        """Verify audit log file exists on disk."""
        # Trigger an event to ensure the file exists
        requests.post(f"{BASE_URL}/test/audit", timeout=10)
        assert os.path.exists(AUDIT_LOG_PATH), "audit_log.jsonl should exist"

    def test_audit_entries_have_required_fields(self):
        """Verify each audit entry has all required fields."""
        resp = requests.get(f"{BASE_URL}/audit/log", timeout=10)
        assert resp.status_code == 200
        entries = resp.json()["entries"]

        required_fields = {"timestamp", "event_type", "details", "prev_hash", "current_hash"}
        for entry in entries:
            missing = required_fields - set(entry.keys())
            assert not missing, f"Entry missing fields: {missing}"

    def test_genesis_entry_prev_hash(self):
        """Verify the first entry has prev_hash of 'GENESIS'."""
        resp = requests.get(f"{BASE_URL}/audit/log", timeout=10)
        entries = resp.json()["entries"]
        if entries:
            assert entries[0]["prev_hash"] == "GENESIS", "First entry should have GENESIS prev_hash"


class TestSandbox:
    """Test suite for sandbox code execution and network isolation."""

    def test_sandbox_basic_execution(self):
        """Verify sandbox can execute simple Python code."""
        resp = requests.post(
            f"{BASE_URL}/test/sandbox",
            json={"code": "print('hello world')"},
            timeout=45,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["exit_code"] == 0
        assert "hello world" in data["stdout"]

    def test_sandbox_blocks_network(self):
        """
        Verify sandbox blocks outbound network connections.
        The code tries to connect to 8.8.8.8:53 which should fail
        due to network isolation.
        """
        code = "import socket; s=socket.socket(); s.connect(('8.8.8.8', 53))"
        resp = requests.post(
            f"{BASE_URL}/test/sandbox",
            json={"code": code},
            timeout=45,
        )
        assert resp.status_code == 200
        data = resp.json()
        # Network should be blocked - either exit_code != 0 or error in stderr
        network_blocked = (
            data["exit_code"] != 0
            or "timed out" in data.get("stderr", "").lower()
            or "refused" in data.get("stderr", "").lower()
            or "unreachable" in data.get("stderr", "").lower()
            or "network" in data.get("stderr", "").lower()
            or "no route" in data.get("stderr", "").lower()
        )
        assert network_blocked, (
            f"Network connection should be blocked. Got: exit_code={data['exit_code']}, "
            f"stderr={data.get('stderr', '')[:200]}"
        )

    def test_sandbox_error_handling(self):
        """Verify sandbox handles Python errors gracefully."""
        resp = requests.post(
            f"{BASE_URL}/test/sandbox",
            json={"code": "raise ValueError('test error')"},
            timeout=45,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["exit_code"] != 0
        assert "ValueError" in data.get("stderr", "")


class TestSentinel:
    """Test suite for the sovereignty sentinel."""

    def test_sentinel_breach_detection(self):
        """
        Trigger a synthetic leak and verify a SOVEREIGNTY_BREACH
        is logged in the audit log.
        """
        resp = requests.post(f"{BASE_URL}/test/sentinel", timeout=15)
        assert resp.status_code == 200
        data = resp.json()
        assert "Leak triggered" in data["status"]

        # Check the last audit entry is a SOVEREIGNTY_BREACH
        last_resp = requests.get(f"{BASE_URL}/audit/last", timeout=10)
        last_entry = last_resp.json()["entry"]
        assert last_entry is not None
        assert last_entry["event_type"] == "SOVEREIGNTY_BREACH", (
            f"Expected SOVEREIGNTY_BREACH, got {last_entry['event_type']}"
        )
        assert "8.8.8.8" in last_entry["details"]["destination_ip"]

    def test_sentinel_returns_status(self):
        """Verify sentinel status endpoint works."""
        resp = requests.get(f"{BASE_URL}/health", timeout=10)
        health = resp.json()
        assert "sentinel" in health
        sentinel_status = health["sentinel"]
        assert "monitoring" in sentinel_status
        assert sentinel_status["monitoring"] is True


class TestModelManager:
    """Test suite for model manager VRAM logic."""

    def test_model_manager_vram_math(self):
        """
        Test VRAM allocation and LRU eviction logic.
        Mock the actual llama.cpp load since model files don't exist.
        """
        from backend.core.model_manager import ModelManager, _StubModel

        # Create a manager with 2 GB budget
        mgr = ModelManager(hardware_tier="BUILD", max_vram_gb=2.0)

        # Load a model that fits
        handle = mgr.load_model("qwen2.5-0.5b-instruct-q4_k_m.gguf")
        assert isinstance(handle, _StubModel)
        assert "qwen2.5-0.5b-instruct-q4_k_m.gguf" in mgr.resident_models

        # Load another that should fit
        handle2 = mgr.load_model("qwen2.5-coder-3b-instruct-q4_k_m.gguf")
        assert isinstance(handle2, _StubModel)

        # Both should be resident (0.5 + 1.5 = 2.0 GB = exactly budget)
        assert len(mgr.resident_models) == 2

    def test_model_manager_lru_eviction(self):
        """
        Test that loading a model beyond VRAM budget triggers LRU eviction.
        """
        from backend.core.model_manager import ModelManager, _StubModel

        # Create manager with tiny budget (1.5 GB)
        mgr = ModelManager(hardware_tier="BUILD", max_vram_gb=1.5)

        # Load first model (0.5 GB)
        mgr.load_model("qwen2.5-0.5b-instruct-q4_k_m.gguf")
        assert len(mgr.resident_models) == 1

        # Load second model (1.5 GB) - should evict first model
        mgr.load_model("qwen2.5-coder-3b-instruct-q4_k_m.gguf")

        # Only the second model should be resident now
        assert len(mgr.resident_models) == 1
        assert "qwen2.5-coder-3b-instruct-q4_k_m.gguf" in mgr.resident_models
        assert "qwen2.5-0.5b-instruct-q4_k_m.gguf" not in mgr.resident_models

    def test_model_manager_get_status(self):
        """Test the status reporting."""
        from backend.core.model_manager import ModelManager

        mgr = ModelManager(hardware_tier="BUILD", max_vram_gb=4.0)
        status = mgr.get_status()
        assert "tier" in status
        assert "max_vram_gb" in status
        assert status["tier"] == "BUILD"
        assert status["max_vram_gb"] == 4.0


class TestHealthEndpoint:
    """Test suite for the health endpoint."""

    def test_health_returns_all_fields(self):
        """Verify health endpoint returns expected fields."""
        resp = requests.get(f"{BASE_URL}/health", timeout=10)
        assert resp.status_code == 200
        health = resp.json()
        assert health["status"] == "ok"
        assert "os" in health
        assert "hardware_tier" in health
        assert "model_roster" in health
        assert "sentinel" in health


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
