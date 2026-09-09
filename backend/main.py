"""
Sovereign AI Workbench - Main FastAPI Application.
Exposes health, sandbox testing, sentinel testing, audit verification,
ingestion, and agent chat endpoints.
"""

import os
import platform
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Query, UploadFile, File, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from typing import Optional
from backend.config import get_tier, get_max_vram_gb, get_model_roster, get_available_models
from backend.core.audit_log import AuditLogger, verify_chain
from backend.core.auth import get_role
from backend.core.model_manager import ModelManager, ModelInferenceError, get_model_manager
from backend.core.sandbox_manager import SandboxManager
from backend.infra.sentinel_runner import SovereignSentinel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Log directory / file: full tracebacks (logged with exc_info=True on
# upload/generation failures) must survive console scrollback so a "Request
# failed with status code 500" can always be traced back to its real cause.
_LOGFILE_DIR = Path(__file__).resolve().parent.parent / "workspace" / "logs"
_LOGFILE_PATH = _LOGFILE_DIR / "backend.log"


def _setup_file_logging() -> None:
    """
    Mirror INFO+ records (including full tracebacks) to a rotating log file.

    Stderr alone is not enough: uvicorn output is ephemeral and users pasting
    a bare 500 need a persistent record of the underlying exception. Safe to
    call on import and idempotent under ``uvicorn --reload`` re-imports.
    """
    from logging.handlers import RotatingFileHandler
    try:
        _LOGFILE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning(f"File logging disabled: cannot create {_LOGFILE_DIR}: {e}")
        return
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, RotatingFileHandler) and h.baseFilename == str(_LOGFILE_PATH):
            return  # already attached (reload)
    try:
        file_handler = RotatingFileHandler(
            _LOGFILE_PATH,
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        root.addHandler(file_handler)
        logger.info(f"File logging enabled: {_LOGFILE_PATH}")
    except OSError as e:
        logger.warning(f"File logging disabled: cannot open {_LOGFILE_PATH}: {e}")


_setup_file_logging()

# Maximum upload size: 20 MB (generous for P&ID scans, handwritten notes, nameplate photos)
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB

import uuid
from backend.core.session_store import get_session_store, SessionStore

# Module-level singletons (initialized at startup)
model_manager: ModelManager = None
sandbox_manager: SandboxManager = None
sentinel: SovereignSentinel = None
audit_logger: AuditLogger = None
rag = None
session_store: SessionStore = get_session_store()


def _ingest_directory_internal(dir_path: Path, rag_instance) -> tuple:
    """Helper to process and ingest files from a directory (including subdirectories) into RAG."""
    from backend.ingestion.pdf_processor import process_pdf
    from backend.ingestion.email_processor import process_email
    from backend.ingestion.chunker import chunk_text

    all_chunks = []
    files_processed = 0

    target_files = [p for p in sorted(dir_path.rglob("*")) if p.is_file() and not p.name.startswith(".")]
    for file_path in target_files:
        try:
            if file_path.suffix.lower() == '.pdf':
                chunks = process_pdf(str(file_path))
            elif file_path.suffix.lower() in ['.msg', '.eml']:
                chunks = process_email(str(file_path))
            elif file_path.suffix.lower() == '.txt':
                text = file_path.read_text(encoding='utf-8', errors='ignore')
                metadata = {
                    "source": file_path.name,
                    "doc_type": "Text",
                    "page": 1,
                }
                chunks = chunk_text(text, metadata)
            else:
                continue

            all_chunks.extend(chunks)
            files_processed += 1
            logger.info(f"Processed {file_path.name}: {len(chunks)} chunks")
        except Exception as e:
            logger.warning(f"Failed to process {file_path.name}: {e}")

    chunks_added = rag_instance.ingest(all_chunks) if all_chunks else 0
    return files_processed, chunks_added


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup, clean up on shutdown."""
    global model_manager, sandbox_manager, sentinel, audit_logger, rag

    logger.info("Starting Sovereign AI Workbench...")

    audit_logger = AuditLogger()
    audit_logger.log_event(
        "SYSTEM_STARTUP",
        {
            "os": platform.system(),
            "tier": get_tier(),
            "max_vram_gb": get_max_vram_gb(),
        },
    )

    model_manager = get_model_manager()
    sandbox_manager = SandboxManager()
    sentinel = SovereignSentinel()

    # Initialize RAG system
    try:
        from backend.tools.rag_search import get_rag
        rag = get_rag()
        if rag.get_status()["total_chunks"] == 0:
            kb_root = Path(__file__).parent.parent / "data" / "knowledge_base"
            if kb_root.exists():
                logger.info("Auto-ingesting default knowledge base on startup...")
                files_cnt, chunks_cnt = _ingest_directory_internal(kb_root, rag)
                logger.info(f"Auto-ingested default KB: {files_cnt} files, {chunks_cnt} chunks")
        logger.info(f"RAG system initialized: {rag.get_status()}")
    except Exception as e:
        logger.warning(f"RAG system initialization failed: {e}")

    # Ensure workspace/outputs/ exists
    output_dir = Path(__file__).parent.parent / "workspace" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Start the egress sentinel in the background
    sentinel.start_monitoring()

    logger.info("Sovereign AI Workbench is ready.")
    yield

    # Shutdown
    logger.info("Shutting down Sovereign AI Workbench...")
    sentinel.stop_monitoring()
    model_manager.unload_all()
    audit_logger.log_event("SYSTEM_SHUTDOWN", {})
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Sovereign AI Workbench",
    description="Air-gapped AI workbench with sovereignty enforcement",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------- Request Models ----------


class CodeExecutionRequest(BaseModel):
    code: str


class GenerateRequest(BaseModel):
    model_name: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7


class ChatRequest(BaseModel):
    prompt: str
    model: Optional[str] = "auto"
    session_id: Optional[str] = None
    project_id: Optional[str] = None
    image_path: Optional[str] = None


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None
    project_id: Optional[str] = None


class UpdateSessionRequest(BaseModel):
    title: str


class CreateProjectRequest(BaseModel):
    name: str
    description: Optional[str] = ""


class UpdateProjectRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class AssignProjectRequest(BaseModel):
    project_id: Optional[str] = None


class IngestRequest(BaseModel):
    directory: str


# ---------- Endpoints ----------


@app.get("/health")
async def health():
    """Return system health status."""
    mm = model_manager or get_model_manager()
    return {
        "status": "ok",
        "os": platform.system(),
        "hardware_tier": get_tier(),
        "max_vram_gb": get_max_vram_gb(),
        "model_roster": get_model_roster(),
        "available_models": get_available_models(),
        "resident_models": mm.get_status() if mm else {},
        "sentinel": sentinel.get_status() if sentinel else {},
    }


@app.get("/models")
async def get_models_endpoint():
    """Return all available local models and the auto routing option."""
    models = get_available_models()
    mm = model_manager or get_model_manager()
    resident = list(mm.resident_models.keys()) if mm else []
    active = resident[-1] if resident else "auto"
    return {
        "models": models,
        "resident_models": resident,
        "default": "auto",
        "active": active,
    }


@app.post("/chat")
async def chat_endpoint(req: ChatRequest, role: str = Depends(get_role)):
    """
    Agent chat endpoint. Invokes the LangGraph ReWOO orchestrator:
    plan -> execute -> synthesize.

    The ``role`` query parameter controls RBAC filtering on retrieved sources.
    """
    try:
        from backend.agents.graph import app as graph_app
        from backend.agents.executor import reset_artifact_tracking

        # Fresh per-request artifact isolation: exactly one deliverable file per
        # filename per generation request.
        reset_artifact_tracking()

        session_id = req.session_id or str(uuid.uuid4())
        session_store.add_message(session_id, "user", req.prompt, project_id=req.project_id)

        result = graph_app.invoke({
            "input": req.prompt,
            "role": role,
            "selected_model": req.model or "auto",
            "image_path": req.image_path,
        })

        # Build trace from graph state for the frontend AgentTrace component
        from backend.agents.planner import is_direct_response
        trace = []
        context = result.get("context", {})
        plan = result.get("plan", [])
        is_direct = plan and is_direct_response(plan)

        if is_direct:
            trace.append("Planner: Detected greeting — direct response")
            trace.append("Synthesizer: Generated response")
        else:
            if plan:
                trace.append(f"Planner: Decomposed into {len(plan)} step(s)")
            for k, v in sorted(context.items()):
                if k.endswith("_tool"):
                    step_num = k.split("_")[1]
                    action = context.get(f"step_{step_num}_action", "")
                    tool = v
                    trace.append(f"Executor: {tool}.{action}()")
            retrieved = result.get("retrieved_sources", [])
            if retrieved:
                trace.append(f"Retriever: Found {len(retrieved)} source(s) from knowledge base")
            else:
                trace.append("Retriever: No matching sources found")
            verification = result.get("verification", {})
            if verification:
                grounded = verification.get("grounded", False)
                trace.append(f"Verifier: Grounding check {'PASSED' if grounded else 'incomplete (no sources)'}")
            trace.append("Synthesizer: Generated final response")

        from backend.config import get_coder_model
        model_used = result.get("model_used", get_coder_model())
        deliverables = list(result.get("deliverables") or [])
        deliverables_lower = {d.lower() for d in deliverables}
        for k, v in sorted(context.items()):
            if k.endswith("_result") and isinstance(v, str):
                from pathlib import Path as _P
                try:
                    p = _P(v)
                    # Only real, on-disk files are deliverables — error strings
                    # that merely end in ".docx" are not artifacts.
                    if p.suffix.lower() in ('.docx', '.pptx', '.xlsx', '.pdf', '.txt', '.csv') and p.is_file():
                        # Case-insensitive dedup: "Report.docx" == "report.docx"
                        if p.name.lower() not in deliverables_lower:
                            deliverables_lower.add(p.name.lower())
                            deliverables.append(p.name)
                except Exception:
                    pass

        # ------------------------------------------------------------------
        # Response-formatting boundary: strip ALL internal execution-trace
        # tokens and reasoning blocks before the response reaches the client.
        # This is the authoritative sanitization point — never left to the
        # model to "not mention".
        # ------------------------------------------------------------------
        from backend.core.output_sanitizer import strip_internal_trace_tokens
        final_response = strip_internal_trace_tokens(result.get("output", ""))

        session_store.add_message(
            session_id,
            "assistant",
            final_response,
            model_used=model_used,
            trace=trace,
            deliverables=deliverables,
        )

        return {
            "response": final_response,
            "model_used": model_used,
            "trace": trace,
            "deliverables": deliverables,
            "session_id": session_id,
        }
    except ModelInferenceError as e:
        # Typed inference failure (CUDA OOM, model unusable...): return a
        # structured, displayable error instead of a bare 500, and log the
        # full traceback.
        logger.error(f"Model inference failed ({e.code}): {e.message}", exc_info=True)
        status_code = 400 if e.code == "image_attach_failed" else 503
        raise HTTPException(
            status_code=status_code,
            detail={"error": e.code, "detail": e.message},
        )
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def list_sessions_endpoint(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    project_id: Optional[str] = None,
):
    """List stored sessions ordered by most recently updated."""
    return session_store.list_sessions(limit=limit, offset=offset, project_id=project_id)


@app.post("/sessions")
async def create_session_endpoint(req: Optional[CreateSessionRequest] = None):
    """Create a new conversation session."""
    title = req.title if req else None
    project_id = req.project_id if req else None
    return session_store.create_session(title=title, project_id=project_id)


@app.get("/sessions/{session_id}")
async def get_session_endpoint(session_id: str):
    """Retrieve a session with all its messages and deliverables."""
    sess = session_store.get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    return sess


@app.delete("/sessions/{session_id}")
async def delete_session_endpoint(session_id: str):
    """Delete a session and all its messages."""
    success = session_store.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@app.patch("/sessions/{session_id}")
async def update_session_endpoint(session_id: str, req: UpdateSessionRequest):
    """Update a session title."""
    success = session_store.update_session_title(session_id, req.title)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "updated", "session_id": session_id, "title": req.title}


@app.patch("/sessions/{session_id}/project")
async def assign_session_project_endpoint(session_id: str, req: AssignProjectRequest):
    """Assign or move a conversation to a project, or set to standalone (null)."""
    success = session_store.assign_session_to_project(session_id, req.project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session or target project not found")
    return {"status": "updated", "session_id": session_id, "project_id": req.project_id}


# ---------- Project Endpoints ----------


@app.get("/projects")
async def list_projects_endpoint():
    """List all projects with chat counts."""
    return session_store.list_projects()


@app.post("/projects")
async def create_project_endpoint(req: CreateProjectRequest):
    """Create a new project grouping container."""
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Project name cannot be empty")
    return session_store.create_project(name=req.name.strip(), description=req.description or "")


@app.get("/projects/{project_id}")
async def get_project_endpoint(project_id: str):
    """Retrieve project details along with member conversations."""
    proj = session_store.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    return proj


@app.patch("/projects/{project_id}")
async def update_project_endpoint(project_id: str, req: UpdateProjectRequest):
    """Update project name or description."""
    success = session_store.update_project(project_id, name=req.name, description=req.description)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "updated", "project_id": project_id}


@app.delete("/projects/{project_id}")
async def delete_project_endpoint(project_id: str):
    """Delete a project. Existing member conversations are preserved as standalone."""
    success = session_store.delete_project(project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "deleted", "project_id": project_id}


@app.post("/test/sandbox")
async def test_sandbox(req: CodeExecutionRequest):
    """Execute code in the sandbox and return output."""
    if not sandbox_manager:
        raise HTTPException(status_code=503, detail="Sandbox manager not initialized")

    result = sandbox_manager.execute_code(req.code)
    return result


@app.post("/test/sentinel")
async def test_sentinel():
    """
    Sovereignty self-check. This is a verification/self-test action, NOT a
    breach event: it is idempotent and non-mutating with respect to the breach
    counter and never logs a SOVEREIGNTY_BREACH. It reports pass/fail based on
    whether an outbound probe to an external IP was blocked (pass) or reached
    (fail).
    """
    if not sentinel:
        raise HTTPException(status_code=503, detail="Sentinel not initialized")

    result = sentinel.trigger_synthetic_leak()
    passed = result.get("status") == "blocked"
    if passed:
        message = "Sovereignty self-test passed: outbound traffic blocked by kernel rules."
    else:
        pid = result.get("initiating_pid", "unknown")
        pname = result.get("initiating_process_name", "python")
        target = result.get("target", "8.8.8.8:53")
        message = (
            f"Sovereignty self-test failed: synthetic outbound probe from PID {pid} ({pname}) "
            f"reached {target}. Root cause: {result.get('root_cause')}"
        )
    return {
        "status": "test_completed",
        "passed": passed,
        "message": message,
        "detail": result,
    }


@app.post("/ingest")
async def ingest_endpoint(req: IngestRequest):
    """
    Ingest documents from a directory into the knowledge base.
    Supports .pdf, .msg, .eml, and .txt files.
    """
    if not rag:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        from pathlib import Path
        import sys

        dir_path = Path(req.directory)
        if not dir_path.exists():
            raise HTTPException(status_code=400, detail=f"Directory not found: {req.directory}")

        files_processed, chunks_added = _ingest_directory_internal(dir_path, rag)

        # Log to audit trail
        audit_logger.log_event(
            "KNOWLEDGE_INGESTION",
            {
                "directory": str(dir_path),
                "files_processed": files_processed,
                "chunks_added": chunks_added,
            },
        )

        return {
            "status": "Ingestion complete",
            "files_processed": files_processed,
            "chunks_added": chunks_added,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/test/audit")
async def test_audit():
    """Verify the audit log hash chain integrity."""
    result = verify_chain()
    return result


@app.post("/models/load")
async def load_model_endpoint(model_name: str):
    """Load a model into GPU memory."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    try:
        model_manager.load_model(model_name)
        return {"status": "loaded", "model": model_name, "manager": model_manager.get_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/pin")
async def pin_model_endpoint(model_name: str):
    """Pin a model into resident GPU memory if within tier VRAM budget."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    try:
        res = model_manager.pin_model(model_name)
        return res
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/unpin")
async def unpin_model_endpoint():
    """Unpin the current model, permitting LRU eviction."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    try:
        res = model_manager.unpin_model()
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
async def generate_endpoint(req: GenerateRequest):
    """Generate text using a loaded model."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    try:
        output = model_manager.generate(
            req.model_name,
            req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        return {"model": req.model_name, "output": output}
    except ModelInferenceError as e:
        logger.error(f"Model inference failed ({e.code}): {e.message}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={"error": e.code, "detail": e.message},
        )
    except Exception as e:
        logger.error(f"Generation endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download")
async def download_file(filename: str = Query(..., description="Name of the file to download")):
    """
    Download a generated file from workspace/outputs/.
    """
    # Input validation: reject null bytes and control characters before any path ops
    if "\x00" in filename or any(ord(c) < 0x20 and c not in '\t\n\r' for c in filename):
        raise HTTPException(status_code=403, detail="Filename contains invalid characters")

    output_dir = Path(__file__).parent.parent / "workspace" / "outputs"
    file_path = (output_dir / filename).resolve()

    # Security: ensure the resolved path is within the output directory
    if not str(file_path).startswith(str(output_dir.resolve())):
        raise HTTPException(status_code=403, detail="Access denied: path traversal detected")

    if not file_path.exists():
        # Case-insensitive fallback for Linux filesystems
        target_name_lower = filename.lower()
        matched_file = None
        for child in output_dir.iterdir():
            if child.is_file() and child.name.lower() == target_name_lower:
                matched_file = child
                break
        if matched_file:
            file_path = matched_file
        else:
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    # Determine MIME type
    suffix = file_path.suffix.lower()
    mime_map = {
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".csv": "text/csv",
    }
    media_type = mime_map.get(suffix, "application/octet-stream")

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type=media_type,
    )


@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...), target_filename: str = Query(..., description="Target filename in sandbox_files/")):
    """
    Upload a file to workspace/sandbox_files/.
    Enforces a 20 MB upload size limit via Content-Length pre-check and
    incremental streaming to prevent disk exhaustion from large uploads.
    """
    # Input validation: reject empty/control-character filenames before any
    # path operation. Keep the client's original (mixed-case) name verbatim so
    # later lookups match the real file on disk exactly.
    if not target_filename or not target_filename.strip():
        raise HTTPException(status_code=400, detail="Filename must not be empty")
    if "\x00" in target_filename or any(ord(c) < 0x20 and c not in "\t\n\r" for c in target_filename):
        raise HTTPException(status_code=403, detail="Filename contains invalid characters")

    sandbox_dir = Path(__file__).parent.parent / "workspace" / "sandbox_files"
    try:
        sandbox_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Upload failed: cannot create sandbox directory {sandbox_dir}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Upload directory is not available on the server")
    if not sandbox_dir.is_dir():
        raise HTTPException(status_code=500, detail="Upload path is not a directory")

    target_path = (sandbox_dir / target_filename).resolve()
    sandbox_root = str(sandbox_dir.resolve()) + os.sep

    # Containment check: the resolved path must live strictly inside the
    # sandbox directory (reject ../, absolute paths, prefix-sibling dirs).
    if not str(target_path).startswith(sandbox_root):
        raise HTTPException(status_code=403, detail="Access denied: path traversal detected")

    # Ensure the target's parent directory exists (nested names like
    # "subfolder/scan.png") so write_bytes never fails with a raw 500.
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Upload failed: cannot create parent for {target_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Upload directory is not writable on the server")
    # --- Upload size enforcement ---
    # 1) Fast-reject via Content-Length header (client may lie, so we also
    #    check during streaming below).
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_size = int(content_length)
            if declared_size > MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload too large: {declared_size} bytes exceeds the {MAX_UPLOAD_SIZE_BYTES} byte limit."
                )
        except (ValueError, TypeError):
            pass  # Malformed header; fall through to streaming check

    # 2) Stream-read in chunks and enforce the limit even if the client
    #    lied about Content-Length or omitted it entirely.
    try:
        chunks = []
        total_size = 0
        chunk_size = 1024 * 1024  # 1 MB chunks
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > MAX_UPLOAD_SIZE_BYTES:
                # Reject immediately — do not write partial data to disk
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload too large: exceeds the {MAX_UPLOAD_SIZE_BYTES} byte limit."
                )
            chunks.append(chunk)

        content = b"".join(chunks)
        target_path.write_bytes(content)
        logger.info(f"File uploaded: {target_path} ({len(content)} bytes)")
        return {
            "status": "File uploaded",
            "filename": target_filename,
            "path": f"workspace/sandbox_files/{target_filename}",
            "size": len(content),
        }
    except HTTPException:
        raise
    except Exception as e:
        # Never surface a raw 500 without context: log the full traceback for
        # diagnosis and return a readable detail the client can display.
        logger.error(f"Upload failed for {target_filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store {target_filename}: {e}",
        )


@app.get("/audit/log")
async def get_audit_log():
    """Read all audit log entries."""
    if not audit_logger:
        raise HTTPException(status_code=503, detail="Audit logger not initialized")

    return {"entries": audit_logger.read_all_entries()}


@app.get("/audit/last")
async def get_last_audit_entry():
    """Get the most recent audit log entry."""
    if not audit_logger:
        raise HTTPException(status_code=503, detail="Audit logger not initialized")

    entry = audit_logger.get_last_entry()
    return {"entry": entry}


@app.get("/benchmark")
async def benchmark_endpoint():
    """
    Run the pre-demo benchmarking script and return accuracy metrics.

    If ``docs/benchmark_results.json`` already exists, returns its contents.
    Otherwise executes the benchmark inline.
    """
    try:
        from pathlib import Path
        import json

        results_path = Path(__file__).parent.parent / "docs" / "benchmark_results.json"

        # Return cached results if available
        if results_path.exists():
            return json.loads(results_path.read_text(encoding="utf-8"))

        # Run benchmark inline
        from scripts.benchmark_accuracy import run_benchmark
        metrics = run_benchmark()
        return metrics
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
