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

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.config import get_tier, get_max_vram_gb, get_model_roster
from backend.core.audit_log import AuditLogger, verify_chain
from backend.core.model_manager import ModelManager
from backend.core.sandbox_manager import SandboxManager
from backend.infra.sentinel_runner import SovereignSentinel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Maximum upload size: 20 MB (generous for P&ID scans, handwritten notes, nameplate photos)
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB

# Module-level singletons (initialized at startup)
model_manager: ModelManager = None
sandbox_manager: SandboxManager = None
sentinel: SovereignSentinel = None
audit_logger: AuditLogger = None
rag = None


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

    model_manager = ModelManager()
    sandbox_manager = SandboxManager()
    sentinel = SovereignSentinel()

    # Initialize RAG system
    try:
        from backend.tools.rag_search import get_rag
        rag = get_rag()
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


class IngestRequest(BaseModel):
    directory: str


# ---------- Endpoints ----------


@app.get("/health")
async def health():
    """Return system health status."""
    return {
        "status": "ok",
        "os": platform.system(),
        "hardware_tier": get_tier(),
        "max_vram_gb": get_max_vram_gb(),
        "model_roster": get_model_roster(),
        "resident_models": model_manager.get_status() if model_manager else {},
        "sentinel": sentinel.get_status() if sentinel else {},
    }


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Agent chat endpoint. Invokes the LangGraph ReWOO orchestrator:
    plan -> execute -> synthesize.
    """
    try:
        from backend.agents.graph import app as graph_app

        result = graph_app.invoke({"input": req.prompt})
        return {"response": result.get("output", "")}
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/test/sandbox")
async def test_sandbox(req: CodeExecutionRequest):
    """Execute code in the sandbox and return output."""
    if not sandbox_manager:
        raise HTTPException(status_code=503, detail="Sandbox manager not initialized")

    result = sandbox_manager.execute_code(req.code)
    return result


@app.post("/test/sentinel")
async def test_sentinel():
    """Trigger a synthetic network leak to test the sentinel."""
    if not sentinel:
        raise HTTPException(status_code=503, detail="Sentinel not initialized")

    result = sentinel.trigger_synthetic_leak()
    return {"status": "Leak triggered, check audit log", "detail": result}


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

        # Add parent directory to sys.path for imports
        parent = str(Path(__file__).parent.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)

        from backend.ingestion.pdf_processor import process_pdf
        from backend.ingestion.email_processor import process_email
        from backend.ingestion.chunker import chunk_text

        all_chunks = []
        files_processed = 0

        for file_path in dir_path.iterdir():
            if file_path.is_file():
                try:
                    if file_path.suffix.lower() == '.pdf':
                        chunks = process_pdf(str(file_path))
                    elif file_path.suffix.lower() in ['.msg', '.eml']:
                        chunks = process_email(str(file_path))
                    elif file_path.suffix.lower() == '.txt':
                        # Process plain text files
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

        # Ingest into RAG
        chunks_added = rag.ingest(all_chunks)

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
    except Exception as e:
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
    # Input validation
    if "\x00" in target_filename:
        raise HTTPException(status_code=403, detail="Filename contains invalid characters")

    sandbox_dir = Path(__file__).parent.parent / "workspace" / "sandbox_files"
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    target_path = (sandbox_dir / target_filename).resolve()

    # Containment check
    if not str(target_path).startswith(str(sandbox_dir.resolve())):
        raise HTTPException(status_code=403, detail="Access denied: path traversal detected")

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
            "path": f"workspace/sandbox_files/{target_filename}",
            "size": len(content),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
