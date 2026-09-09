"""
Test suite for Phase 1: Session Persistence in Sovereign AI Workbench.

Verifies:
1. SQLite session store saves every conversation turn, model used, traces, and deliverables.
2. Conversations survive process restart (re-opening store on same SQLite file).
3. Auto-generated titles from first user message.
4. Fast API endpoints: GET /sessions, GET /sessions/{id}, POST /sessions, DELETE /sessions/{id}.
5. /chat endpoint session tracking across multiple turns.
"""

import os
import tempfile
import pytest
from fastapi.testclient import TestClient

from backend.core.session_store import SessionStore
from backend.main import app, session_store


class TestSessionStoreUnit:
    """Unit tests for SQLite SessionStore."""

    def test_session_creation_and_retrieval(self, tmp_path):
        db_file = str(tmp_path / "test_sessions.db")
        store = SessionStore(db_file)

        sess = store.create_session(title="Turbine Analysis")
        sid = sess["id"]
        assert sid is not None
        assert sess["title"] == "Turbine Analysis"
        assert sess["message_count"] == 0

        # Add user message
        m1 = store.add_message(sid, "user", "What is the operating tolerance of Turbine #4?")
        assert m1["id"] is not None
        assert m1["role"] == "user"

        # Add assistant message with deliverables and trace
        m2 = store.add_message(
            sid,
            "assistant",
            "Turbine #4 tolerance is +/- 0.05mm.",
            model_used="llama-3.2-3b-instruct-q4_k_m.gguf",
            trace=["Retriever: Found 1 source", "Verifier: Grounding check PASSED"],
            deliverables=["turbine_report.docx", "vibration_data.xlsx"],
        )
        assert m2["id"] is not None
        assert m2["role"] == "assistant"

        loaded = store.get_session(sid)
        assert loaded is not None
        assert loaded["title"] == "Turbine Analysis"
        assert loaded["message_count"] == 2
        assert len(loaded["messages"]) == 2
        assert loaded["messages"][0]["content"] == "What is the operating tolerance of Turbine #4?"
        assert loaded["messages"][1]["deliverables"] == ["turbine_report.docx", "vibration_data.xlsx"]
        assert loaded["messages"][1]["model_used"] == "llama-3.2-3b-instruct-q4_k_m.gguf"
        assert len(loaded["messages"][1]["trace"]) == 2

    def test_persistence_survives_app_restart(self, tmp_path):
        """Simulate app shutdown and restart by dropping store instance and creating a new one."""
        db_file = str(tmp_path / "restart_test.db")

        # Session 1 in first instance
        store1 = SessionStore(db_file)
        sess1 = store1.create_session(session_id="session-alpha", title="Alpha Project")
        store1.add_message("session-alpha", "user", "Step 1: Check heat exchanger")
        store1.add_message(
            "session-alpha",
            "assistant",
            "Heat exchanger efficiency is 94%.",
            model_used="qwen2.5-coder-3b-instruct-q4_k_m.gguf",
            deliverables=["heat_map.xlsx"],
        )

        # "RESTART": drop store1 entirely
        del store1

        # Re-open store on same file
        store2 = SessionStore(db_file)
        loaded = store2.get_session("session-alpha")
        assert loaded is not None, "Session must survive restart"
        assert loaded["title"] == "Alpha Project"
        assert loaded["message_count"] == 2
        assert loaded["messages"][0]["content"] == "Step 1: Check heat exchanger"
        assert loaded["messages"][1]["deliverables"] == ["heat_map.xlsx"]
        assert loaded["messages"][1]["model_used"] == "qwen2.5-coder-3b-instruct-q4_k_m.gguf"

    def test_auto_generated_title_from_prompt(self, tmp_path):
        db_file = str(tmp_path / "title_test.db")
        store = SessionStore(db_file)

        # Message added to non-existent session auto-creates session with derived title
        sid = "auto-sess-1"
        store.add_message(sid, "user", "Calculate maximum allowable stress on boiler plate 12")
        sess = store.get_session(sid)
        assert sess is not None
        assert sess["title"] == "Calculate maximum allowable stress on boiler..."

    def test_delete_session_cascades(self, tmp_path):
        db_file = str(tmp_path / "del_test.db")
        store = SessionStore(db_file)

        s = store.create_session()
        sid = s["id"]
        store.add_message(sid, "user", "Hello")
        store.add_message(sid, "assistant", "Hi there")

        assert store.get_session(sid) is not None
        assert store.delete_session(sid) is True
        assert store.get_session(sid) is None

        # Verify messages table is cleaned up
        with store._get_connection() as conn:
            cnt = conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (sid,)).fetchone()[0]
            assert cnt == 0


class TestSessionApiEndpoints:
    """Integration tests for FastAPI session endpoints."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        test_db = str(tmp_path / "api_sessions.db")
        test_store = SessionStore(test_db)
        monkeypatch.setattr("backend.main.session_store", test_store)
        with TestClient(app) as c:
            yield c

    def test_chat_creates_and_updates_session(self, client):
        # 1. First turn without session_id -> auto-creates session
        resp1 = client.post("/chat", json={"prompt": "Hello"})
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert "session_id" in data1
        sid = data1["session_id"]
        assert sid is not None

        # 2. Second turn with same session_id
        resp2 = client.post("/chat", json={"prompt": "What can you do?", "session_id": sid})
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["session_id"] == sid

        # 3. GET /sessions lists the session
        list_resp = client.get("/sessions")
        assert list_resp.status_code == 200
        sessions = list_resp.json()
        assert len(sessions) >= 1
        found = next((s for s in sessions if s["id"] == sid), None)
        assert found is not None
        assert found["message_count"] == 4  # 2 user + 2 assistant

        # 4. GET /sessions/{id} returns full conversation
        detail_resp = client.get(f"/sessions/{sid}")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["id"] == sid
        assert len(detail["messages"]) == 4
        assert detail["messages"][0]["role"] == "user"
        assert detail["messages"][0]["content"] == "Hello"
        assert detail["messages"][1]["role"] == "assistant"
        assert detail["messages"][2]["role"] == "user"
        assert detail["messages"][2]["content"] == "What can you do?"
        assert detail["messages"][3]["role"] == "assistant"

    def test_explicit_session_lifecycle(self, client):
        # Create session explicitly
        c_resp = client.post("/sessions", json={"title": "Custom Investigation"})
        assert c_resp.status_code == 200
        sid = c_resp.json()["id"]

        # Rename session
        p_resp = client.patch(f"/sessions/{sid}", json={"title": "Renamed Investigation"})
        assert p_resp.status_code == 200

        # Confirm rename
        g_resp = client.get(f"/sessions/{sid}")
        assert g_resp.status_code == 200
        assert g_resp.json()["title"] == "Renamed Investigation"

        # Delete session
        d_resp = client.delete(f"/sessions/{sid}")
        assert d_resp.status_code == 200

        # Confirm 404
        assert client.get(f"/sessions/{sid}").status_code == 404
