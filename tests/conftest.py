"""
Pytest configuration: isolates AuditLogger singleton state between test functions.

The AuditLogger uses a module-level singleton writer thread whose _next_sequence
counter persists across test functions. Different tests write to different log
files, but verify_chain() on each file expects sequences starting at 1 — causing
false sequence_gap / valid=False when the singleton has advanced from prior tests.

This fixture resets the singleton to a completely fresh state before EACH test
function, ensuring full isolation.
"""

import os
import sys
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True, scope="function")
def _reset_audit_singleton():
    """Reset the AuditLogger singleton writer before every test function."""
    from backend.core.audit_log import AuditLogger

    # Reset singleton writer thread to fresh state (_next_sequence=1)
    AuditLogger._reset_for_testing()

    yield

    # No teardown needed — the next test's fixture will reset again
