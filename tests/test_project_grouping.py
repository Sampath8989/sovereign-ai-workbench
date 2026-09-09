"""
Test suite for Phase 2: Projects Grouping and Foreign Key Persistence.
Verifies project containers, chat counts, standalone vs project sessions,
and persistence surviving application/process restarts.
"""

import os
import shutil
import tempfile
import uuid
import pytest
from fastapi.testclient import TestClient

from backend.core.session_store import SessionStore
from backend.main import app


@pytest.fixture
def temp_db_store():
    """Create an isolated temporary SQLite database for testing."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_projects.db")
    store = SessionStore(db_path=db_path)
    yield store, db_path
    shutil.rmtree(tmp_dir, ignore_errors=True)


class TestProjectGroupingUnit:
    """Unit tests for project containers and foreign-key relationships."""

    def test_create_and_list_projects(self, temp_db_store):
        store, _ = temp_db_store
        p1 = store.create_project(name="Pipeline Inspection", description="Corrosion tests")
        p2 = store.create_project(name="Turbine Maintenance", description="Vibration logs")

        assert p1["name"] == "Pipeline Inspection"
        assert p2["name"] == "Turbine Maintenance"

        projects = store.list_projects()
        assert len(projects) == 2
        names = [p["name"] for p in projects]
        assert "Pipeline Inspection" in names
        assert "Turbine Maintenance" in names
        assert projects[0]["session_count"] == 0

    def test_session_inside_project_vs_standalone(self, temp_db_store):
        store, _ = temp_db_store
        proj = store.create_project(name="Project Alpha")

        # Standalone session
        s_standalone = store.create_session(title="Standalone Chat", project_id=None)
        # Project session
        s_proj1 = store.create_session(title="Project Chat 1", project_id=proj["id"])
        s_proj2 = store.create_session(title="Project Chat 2", project_id=proj["id"])

        # Check project session count
        projects = store.list_projects()
        proj_summary = next(p for p in projects if p["id"] == proj["id"])
        assert proj_summary["session_count"] == 2

        # Check filtering in list_sessions
        all_sessions = store.list_sessions(project_id=None)
        assert len(all_sessions) == 3

        proj_sessions = store.list_sessions(project_id=proj["id"])
        assert len(proj_sessions) == 2
        proj_titles = [s["title"] for s in proj_sessions]
        assert "Project Chat 1" in proj_titles
        assert "Project Chat 2" in proj_titles

        standalone_sessions = store.list_sessions(project_id="standalone")
        assert len(standalone_sessions) == 1
        assert standalone_sessions[0]["title"] == "Standalone Chat"

    def test_assign_session_to_project(self, temp_db_store):
        store, _ = temp_db_store
        proj = store.create_project(name="Target Project")
        sess = store.create_session(title="Initially Standalone")

        assert sess["project_id"] is None

        # Move to project
        ok = store.assign_session_to_project(sess["id"], proj["id"])
        assert ok is True

        retrieved = store.get_session(sess["id"])
        assert retrieved["project_id"] == proj["id"]

        # Move back to standalone
        ok = store.assign_session_to_project(sess["id"], None)
        assert ok is True

        retrieved = store.get_session(sess["id"])
        assert retrieved["project_id"] is None

    def test_delete_project_preserves_sessions_as_standalone(self, temp_db_store):
        store, _ = temp_db_store
        proj = store.create_project(name="Temporary Project")
        sess = store.create_session(title="Protected Chat", project_id=proj["id"])

        # Delete project
        ok = store.delete_project(proj["id"])
        assert ok is True

        # Project should not exist
        assert store.get_project(proj["id"]) is None

        # Session should still exist, but project_id must be None (standalone)
        retrieved = store.get_session(sess["id"])
        assert retrieved is not None
        assert retrieved["project_id"] is None
        assert retrieved["title"] == "Protected Chat"

    def test_persistence_survives_app_restart(self, temp_db_store):
        store1, db_path = temp_db_store

        # 1. Create project and sessions in store1
        p = store1.create_project(name="Surviving Project", description="Must survive process exit")
        s1 = store1.create_session(title="Surviving Session 1", project_id=p["id"])
        store1.add_message(s1["id"], "user", "What is the operating pressure?")
        store1.add_message(s1["id"], "assistant", "Nominal operating pressure is 150 PSI.")

        # 2. Simulate complete application exit / process restart by abandoning store1
        del store1

        # 3. Instantiate fresh new SessionStore pointing to the same SQLite file
        store2 = SessionStore(db_path=db_path)

        projects = store2.list_projects()
        assert len(projects) == 1
        assert projects[0]["id"] == p["id"]
        assert projects[0]["name"] == "Surviving Project"
        assert projects[0]["session_count"] == 1

        sess = store2.get_session(s1["id"])
        assert sess is not None
        assert sess["project_id"] == p["id"]
        assert len(sess["messages"]) == 2
        assert sess["messages"][0]["content"] == "What is the operating pressure?"
        assert sess["messages"][1]["content"] == "Nominal operating pressure is 150 PSI."


class TestProjectApiEndpoints:
    """Integration tests for FastAPI /projects and project assignment endpoints."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        self.client = TestClient(app)

    def test_project_api_lifecycle(self):
        # 1. Create project
        resp = self.client.post("/projects", json={"name": "API Engineering Project", "description": "API Test"})
        assert resp.status_code == 200
        proj_data = resp.json()
        proj_id = proj_data["id"]
        assert proj_data["name"] == "API Engineering Project"

        # 2. List projects
        resp = self.client.get("/projects")
        assert resp.status_code == 200
        projects = resp.json()
        match = next((p for p in projects if p["id"] == proj_id), None)
        assert match is not None
        assert match["session_count"] == 0

        # 3. Create session inside project via /sessions
        resp = self.client.post("/sessions", json={"title": "API Session in Proj", "project_id": proj_id})
        assert resp.status_code == 200
        sess_id = resp.json()["id"]

        # Check updated count in /projects
        resp = self.client.get("/projects")
        match = next((p for p in resp.json() if p["id"] == proj_id), None)
        assert match["session_count"] == 1

        # 4. Get single project
        resp = self.client.get(f"/projects/{proj_id}")
        assert resp.status_code == 200
        proj_detail = resp.json()
        assert len(proj_detail["sessions"]) == 1
        assert proj_detail["sessions"][0]["id"] == sess_id

        # 5. Move session to standalone via PATCH /sessions/{id}/project
        resp = self.client.patch(f"/sessions/{sess_id}/project", json={"project_id": None})
        assert resp.status_code == 200

        # Verify session is now standalone
        resp = self.client.get(f"/sessions/{sess_id}")
        assert resp.json()["project_id"] is None

        # 6. Delete project
        resp = self.client.delete(f"/projects/{proj_id}")
        assert resp.status_code == 200

        # Verify project is gone
        resp = self.client.get(f"/projects/{proj_id}")
        assert resp.status_code == 404

        # Clean up session
        self.client.delete(f"/sessions/{sess_id}")
