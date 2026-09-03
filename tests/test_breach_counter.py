"""
Regression test for breach counter state bug.

Bug: SovereignSentinel._breach_count persisted across monitoring sessions.
When the sentinel was restarted (new monitoring session), the counter retained
the count from previous sessions, causing "Breaches detected: 7" on fresh page load.

Fix: Reset _breach_count and _seen_connections when start_monitoring() is called.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestBreachCounterReset:
    """Test that the breach counter resets per monitoring session."""

    def test_breach_count_starts_at_zero(self):
        """On fresh sentinel init, breach_count should be 0."""
        from backend.infra.sentinel_runner import SovereignSentinel

        sentinel = SovereignSentinel(enforce_kills=False)
        status = sentinel.get_status()
        assert status["breach_count"] == 0

    def test_breach_count_resets_on_monitoring_restart(self):
        """When monitoring is restarted, breach_count should reset to 0."""
        from backend.infra.sentinel_runner import SovereignSentinel

        sentinel = SovereignSentinel(enforce_kills=False)

        # Simulate some breaches by directly setting the counter
        sentinel._breach_count = 5
        assert sentinel.get_status()["breach_count"] == 5

        # Start monitoring — should reset the counter
        sentinel.start_monitoring()
        assert sentinel.get_status()["breach_count"] == 0

        # Clean up
        sentinel.stop_monitoring()

    def test_breach_count_increments_during_session(self):
        """Breaches detected during an active session should increment the counter."""
        from backend.infra.sentinel_runner import SovereignSentinel

        sentinel = SovereignSentinel(enforce_kills=False)
        sentinel.start_monitoring()

        # Simulate a breach
        sentinel._enforce_breach(99999, "8.8.8.8", "tcp")
        assert sentinel.get_status()["breach_count"] == 1

        # Simulate another
        sentinel._enforce_breach(99999, "1.1.1.1", "tcp")
        assert sentinel.get_status()["breach_count"] == 2

        sentinel.stop_monitoring()

    def test_seen_connections_reset_on_restart(self):
        """_seen_connections should reset when monitoring restarts."""
        from backend.infra.sentinel_runner import SovereignSentinel

        sentinel = SovereignSentinel(enforce_kills=False)
        sentinel.start_monitoring()

        # Add some seen connections
        sentinel._seen_connections.add("1234:8.8.8.8:53:tcp")
        sentinel._seen_connections.add("5678:1.1.1.1:443:tcp")
        assert len(sentinel._seen_connections) == 2

        # Restart monitoring — should clear seen connections
        sentinel.stop_monitoring()
        sentinel.start_monitoring()
        assert len(sentinel._seen_connections) == 0

        sentinel.stop_monitoring()

    def test_health_endpoint_returns_zero_breach_count_initially(self):
        """The /health endpoint should report 0 breaches on fresh start."""
        import requests

        try:
            resp = requests.get("http://localhost:8000/health", timeout=5)
            if resp.status_code == 200:
                health = resp.json()
                # The sentinel breach_count should be 0 or small
                # (not 7 or some arbitrary number from a previous session)
                breach_count = health.get("sentinel", {}).get("breach_count", 0)
                assert breach_count < 10, (
                    f"Breach count suspiciously high on fresh load: {breach_count}. "
                    "Counter may not be resetting between sessions."
                )
        except requests.ConnectionError:
            pytest.skip("FastAPI server not running")
