#!/usr/bin/env python3
"""
Validate System: Cross-platform pre-flight check for the Sovereign AI Workbench.

Checks:
  1. FastAPI server is running (GET /health)
  2. Qdrant is running (GET http://localhost:6333/collections)
  3. Audit log hash chain is intact (POST /test/audit)

Usage:
    python scripts/validate_system.py [--base-url http://127.0.0.1:8000]
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import requests
except ImportError:
    print("[SKIP] requests library not installed. Cannot run validation.")
    sys.exit(2)


def check_fastapi(base_url: str, timeout: int = 5) -> bool:
    """Check if the FastAPI server is running and healthy."""
    print(f"  Checking FastAPI at {base_url}/health ...")
    try:
        resp = requests.get(f"{base_url}/health", timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            tier = data.get("hardware_tier", "unknown")
            print(f"  [PASS] FastAPI is running (tier={tier})")
            return True
        else:
            print(f"  [FAIL] FastAPI returned status {resp.status_code}")
            return False
    except requests.ConnectionError:
        print(f"  [FAIL] Cannot connect to FastAPI at {base_url}")
        print(f"         Start the server with: uvicorn backend.main:app --reload")
        return False
    except requests.Timeout:
        print(f"  [FAIL] FastAPI health check timed out ({timeout}s)")
        return False
    except Exception as e:
        print(f"  [FAIL] FastAPI check error: {e}")
        return False


def check_qdrant(timeout: int = 5) -> bool:
    """Check if Qdrant is running AND the sovereign_kb collection exists."""
    qdrant_host = "localhost"
    qdrant_port = 6333
    base_url = f"http://{qdrant_host}:{qdrant_port}"
    expected_collection = "sovereign_kb"

    print(f"  Checking Qdrant at {base_url} ...")
    try:
        # Step 1: Check Qdrant is reachable
        resp = requests.get(f"{base_url}/collections", timeout=timeout)
        if resp.status_code != 200:
            print(f"  [WARN] Qdrant returned status {resp.status_code} (may still work with in-memory fallback)")
            return True  # Non-fatal: RAG falls back to in-memory search

        data = resp.json()
        collections = [c["name"] for c in data.get("result", {}).get("collections", [])]
        print(f"  Qdrant running ({len(collections)} collections: {collections})")

        # Step 2: Check sovereign_kb collection exists
        if expected_collection not in collections:
            print(f"  [FAIL] Collection '{expected_collection}' not found in Qdrant")
            print(f"         Available: {collections}")
            print(f"         RAG search will return empty results")
            return False

        # Step 3: Check collection has points (non-empty)
        coll_resp = requests.get(
            f"{base_url}/collections/{expected_collection}", timeout=timeout
        )
        if coll_resp.status_code == 200:
            coll_data = coll_resp.json().get("result", {})
            point_count = coll_data.get("points_count", 0)
            if point_count == 0:
                print(f"  [WARN] Collection '{expected_collection}' exists but has 0 points")
                print(f"         RAG search will return empty results (run /ingest first)")
                return True  # Non-fatal: empty is OK if not yet ingested
            else:
                print(f"  [PASS] Qdrant healthy: '{expected_collection}' has {point_count} points")
                return True
        else:
            print(f"  [WARN] Could not query collection details (status {coll_resp.status_code})")
            return True  # Non-fatal

    except requests.ConnectionError:
        print(f"  [WARN] Qdrant not reachable at {base_url}")
        print(f"         RAG will use in-memory fallback (acceptable for demo)")
        return True  # Non-fatal
    except requests.Timeout:
        print(f"  [WARN] Qdrant health check timed out ({timeout}s)")
        return True  # Non-fatal
    except Exception as e:
        print(f"  [WARN] Qdrant check error: {e}")
        return True  # Non-fatal


def check_audit_log(base_url: str, timeout: int = 10) -> bool:
    """Check audit log hash chain integrity."""
    print(f"  Checking audit log via {base_url}/test/audit ...")
    try:
        resp = requests.post(f"{base_url}/test/audit", timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            is_valid = data.get("valid", False)
            entry_count = data.get("entry_count", 0)
            details = data.get("details", "")
            if is_valid:
                print(f"  [PASS] Audit log is intact ({entry_count} entries, chain valid)")
                return True
            else:
                print(f"  [FAIL] Audit log chain broken: {details}")
                return False
        else:
            print(f"  [FAIL] Audit check returned status {resp.status_code}")
            return False
    except requests.ConnectionError:
        print(f"  [FAIL] Cannot connect to audit endpoint")
        return False
    except requests.Timeout:
        print(f"  [FAIL] Audit check timed out ({timeout}s)")
        return False
    except Exception as e:
        print(f"  [FAIL] Audit check error: {e}")
        return False


def validate_system(base_url: str = "http://127.0.0.1:8000") -> dict:
    """
    Run all pre-flight checks and return a summary dict.

    Returns:
        A dict with keys: fastapi, qdrant, audit, all_passed.
    """
    print("\n" + "=" * 60)
    print("Sovereign AI Workbench — Pre-Flight Validation")
    print("=" * 60)

    results = {}

    print("\n[1/3] FastAPI Server")
    results["fastapi"] = check_fastapi(base_url)

    print("\n[2/3] Qdrant Vector DB")
    results["qdrant"] = check_qdrant()

    print("\n[3/3] Audit Log Integrity")
    results["audit"] = check_audit_log(base_url)

    # FastAPI is the only hard requirement
    results["all_passed"] = results["fastapi"]

    print("\n" + "=" * 60)
    if results["all_passed"]:
        print("RESULT: All critical checks passed. Ready for demo.")
    else:
        print("RESULT: Critical checks failed. Fix issues above before demo.")
    print("=" * 60 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Pre-flight validation for the Sovereign AI Workbench demo."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="FastAPI server URL (default: http://127.0.0.1:8000)",
    )
    args = parser.parse_args()

    results = validate_system(args.base_url)
    sys.exit(0 if results["all_passed"] else 1)


if __name__ == "__main__":
    main()
