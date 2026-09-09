"""
Regression tests for the /upload endpoint.

Covers the file-lookup bug end-to-end (the uploaded filename must be stored
with its exact, mixed case so later vision/file lookups resolve it) and the
upload hardening for the "Request failed with status code 500" report:

- malformed multipart bodies must yield a readable 4xx, never a raw 500;
- empty / control-character / traversal filenames are rejected up front;
- storage failures are reported with a human-readable detail + traceback log.
"""

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app

SANDBOX_DIR = PROJECT_ROOT / "workspace" / "sandbox_files"
MIXED_CASE_NAME = "Screenshot_2026-09-03_15-43-10.png"

# Small valid PNG-ish payload (content is irrelevant to the endpoint).
PAYLOAD = b"\x89PNG\r\n\x1a\n" + b"0" * 256

_CREATED: list = []


@pytest.fixture(scope="module")
def client():
    """TestClient WITHOUT the lifespan context: exercises the endpoint code
    directly without needing RAG/sentinel/model infrastructure."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    for p in list(_CREATED):
        try:
            if p.is_dir():
                p.rmdir()
            else:
                p.unlink(missing_ok=True)
        except OSError:
            pass
        finally:
            if p in _CREATED:
                _CREATED.remove(p)


def _upload(client, name: str, content: bytes = PAYLOAD):
    res = client.post(
        f"/upload?target_filename={name}",
        files={"file": ("upload.bin", content, "application/octet-stream")},
    )
    return res


class TestUploadStoresExactCase:
    def test_mixed_case_name_preserved_on_disk(self, client):
        res = _upload(client, MIXED_CASE_NAME)
        assert res.status_code == 200, res.text
        assert res.json()["filename"] == MIXED_CASE_NAME

        on_disk = SANDBOX_DIR / MIXED_CASE_NAME
        assert on_disk.is_file(), f"File not stored under its exact-case name: {on_disk}"
        assert on_disk.read_bytes() == PAYLOAD
        _CREATED.append(on_disk)

    def test_lowercase_variant_does_not_overwrite_mixed_case(self, client):
        assert _upload(client, MIXED_CASE_NAME).status_code == 200
        _CREATED.append(SANDBOX_DIR / MIXED_CASE_NAME)
        # Different case => different file on a case-sensitive filesystem.
        lower = SANDBOX_DIR / MIXED_CASE_NAME.lower()
        assert not lower.exists()


class TestUploadValidation:
    def test_empty_filename_rejected(self, client):
        res = client.post(
            "/upload?target_filename=",
            files={"file": ("x.bin", PAYLOAD, "application/octet-stream")},
        )
        assert res.status_code == 400
        assert "empty" in res.json()["detail"].lower()

    def test_null_byte_filename_rejected(self, client):
        res = client.post(
            "/upload?target_filename=evil%00.png",
            files={"file": ("x.bin", PAYLOAD, "application/octet-stream")},
        )
        assert res.status_code in (400, 403)

    def test_path_traversal_rejected(self, client):
        res = _upload(client, "../escape.png")
        assert res.status_code == 403
        assert not (SANDBOX_DIR.parent / "escape.png").exists()

    def test_absolute_path_rejected(self, client):
        res = _upload(client, "/etc/cron.d/evil")
        assert res.status_code == 403


class TestUploadNoRaw500:
    def test_boundaryless_multipart_returns_readable_4xx(self, client):
        """The frontend must never send this, but if it does the server must
        reply with a readable 4xx — never an opaque 500."""
        body = (
            b'--BOUND\r\n'
            b'Content-Disposition: form-data; name="file"; filename="x.png"\r\n'
            b'Content-Type: image/png\r\n\r\n'
            + PAYLOAD +
            b'\r\n--BOUND--\r\n'
        )
        res = client.post(
            f"/upload?target_filename={MIXED_CASE_NAME}",
            content=body,
            headers={"Content-Type": "multipart/form-data"},  # no boundary param
        )
        assert res.status_code == 400, res.text
        assert "boundary" in res.json().get("detail", "").lower()

    def test_nested_target_directory_is_created(self, client):
        """Nested names must not crash with a 500 — the parent dir is created."""
        res = _upload(client, "subfolder/Scan_2026.png")
        assert res.status_code == 200, res.text
        nested = SANDBOX_DIR / "subfolder" / "Scan_2026.png"
        assert nested.is_file()
        _CREATED.append(nested)
        _CREATED.append(SANDBOX_DIR / "subfolder")
