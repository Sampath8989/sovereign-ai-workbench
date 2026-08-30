"""
Sovereign AI Workbench - Main FastAPI Application.
Exposes health, sandbox testing, sentinel testing, audit verification,
and agent chat endpoints.
"""

import os
import platform
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
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

# Module-level singletons (initialized at startup)
model_manager: ModelManager = None
sandbox_manager: SandboxManager = None
sentinel: SovereignSentinel = None
audit_logger: AuditLogger = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup, clean up on shutdown."""
    global model_manager, sandbox_manager, sentinel, audit_logger

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
