# Sovereign AI Workbench

A production-ready, air-gapped AI workbench with sovereignty enforcement, hybrid RAG, multi-step agent orchestration, and deliverable synthesis tools.

> **Fully Windows compatible** — runs on Windows 10/11 with Python 3.10+, Node.js 18+, and optional NVIDIA GPU.

---

## Table of Contents

- [What It Is](#what-it-is)
- [Architecture Overview](#architecture-overview)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation (Windows)](#installation-windows)
- [Running the Workbench](#running-the-workbench)
- [Frontend UI](#frontend-ui)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Docker Setup (Optional)](#docker-setup-optional)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## What It Is

Sovereign AI Workbench is a **locally-hosted AI assistant** that runs entirely on your machine with **no external network access**. It combines:

- **Local LLM inference** via llama.cpp (GGUF models) with a MockLLM fallback for testing
- **Multi-step agent orchestration** using LangGraph's ReWOO pattern
- **Hybrid RAG** (BM25 sparse + Qdrant dense vectors) for knowledge retrieval
- **Deliverable synthesis** — generates Word, PowerPoint, Excel files from natural language
- **Network sovereignty enforcement** — monitors and kills unauthorized network connections
- **Tamper-evident audit logging** with SHA-256 hash chaining

If no GPU or GGUF model files are present, the system runs in **MockLLM mode** — deterministic responses allow full pipeline testing without hardware.

---

## Architecture Overview

| Layer | Components | Purpose |
|-------|-----------|---------|
| **Step 1 — Inference & Sandbox** | Model Manager, gVisor Sandbox, eBPF Sentinel, Audit Log | VRAM-aware model hot-swapping, kernel-level code sandboxing, network egress enforcement, tamper-evident logging |
| **Step 2 — Agent Orchestration** | LangGraph ReWOO, Planner, Executor, Router, File I/O | Multi-step task decomposition, plan→execute→synthesize pipeline, semantic task routing |
| **Step 3 — Knowledge & Verification** | Hybrid RAG (BM25 + Qdrant), Citation Tagger, Chain-of-Thought Verifier | Hybrid sparse+dense search, source citation tagging, claim grounding verification |
| **Step 4 — Deliverable Synthesis** | Doc Generator, PPT Generator, Spreadsheet Generator/Analyzer, Symbolic Calculator | Word/PowerPoint/Excel generation, xlsx analysis, SymPy math engine, file download endpoint |

---

## How It Works

1. **User sends a prompt** (e.g., "Create a Word document with Q4 report")
2. **Planner** (MockLLM) decomposes the prompt into a JSON plan of steps
3. **Executor** dispatches each step to the appropriate tool:
   - `doc_generator` → creates `.docx`
   - `ppt_generator` → creates `.pptx`
   - `spreadsheet_generator` → creates `.xlsx`
   - `calculator` → solves math expressions via SymPy
   - `file_io` → reads/writes sandboxed files
   - `code` → executes Python in Docker sandbox
4. **Retriever** searches the knowledge base via hybrid RAG
5. **Verifier** checks that generated claims are grounded in sources
6. **Synthesizer** produces the final answer with citation tags

For **greetings and simple chat**, the pipeline short-circuits directly to a conversational response.

---

## Prerequisites

### Required
- **Python 3.10, 3.11, or 3.12** — [Download from python.org](https://www.python.org/downloads/)
  - ✅ Check "Add Python to PATH" during installation
- **Node.js 18+ (LTS)** — [Download from nodejs.org](https://nodejs.org/)
- **Git** — [Download from git-scm.com](https://git-scm.com/)

### Optional (for full features)
- **NVIDIA GPU** with CUDA support — for local LLM inference
- **Docker Desktop** — for sandboxed code execution
- **Qdrant** — for persistent vector search (falls back to in-memory)

---

## Installation (Windows)

### Step 1: Clone the Repository

```bash
git clone <your-repo-url> sovereign-ai-workbench
cd sovereign-ai-workbench
```

### Step 2: Run the Setup Script

Double-click **`setup_windows.bat`** or run in Command Prompt / PowerShell:

```cmd
setup_windows.bat
```

This will:
1. Verify Python and Node.js are installed
2. Create a Python virtual environment (`venv/`)
3. Install all Python backend dependencies (`requirements.txt`)
4. Install all frontend dependencies (`frontend/package.json`)

### Step 3: (Optional) Place GGUF Model Files

Download GGUF models and place them in the `models/` directory:

```
models/
├── qwen2.5-coder-7b-instruct-q3_k_m.gguf    (Code tasks)
├── deepseek-r1-7b.gguf                        (Reasoning/math)
├── phi4-14b.gguf                              (Deep synthesis)
├── llava-7b.gguf                              (Vision/OCR)
├── qwen2.5-7b-instruct-q3_k_m.gguf           (General chat)
└── ... (see config.py for full roster)
```

> **Without model files**, the system runs in MockLLM mode — all features work with deterministic test responses.

### Step 4: (Optional) Configure Environment

Create or edit `.env` in the project root:

```env
# Hardware tier: BUILD (4GB VRAM) or DEMO (8GB+ VRAM)
HARDWARE_TIER=BUILD

# Force MockLLM even if models exist (useful for testing)
USE_MOCK_LLM=true

# Sentinel enforcement (kills network-breaching processes)
SENTINEL_ENFORCE=false

# Qdrant connection (optional — falls back to in-memory)
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Embedder (1 = mock for fast startup, 0 = real sentence-transformers)
USE_MOCK_EMBEDDER=1
```

---

## Running the Workbench

### Option A: Use the Batch Script (Recommended)

Double-click **`start_windows.bat`** — this opens two separate terminal windows:
- **Backend** on `http://localhost:8000`
- **Frontend** on `http://localhost:5173`

### Option B: Manual Start

Open **two separate** Command Prompt / PowerShell windows:

**Terminal 1 — Backend:**
```cmd
cd sovereign-ai-workbench
venv\Scripts\activate
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Frontend:**
```cmd
cd sovereign-ai-workbench\frontend
npm run dev
```

### Option C: Backend Only (No UI)

```cmd
cd sovereign-ai-workbench
venv\Scripts\activate
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Verify It's Running

```bash
curl http://localhost:8000/health
```

Or open `http://localhost:8000/health` in your browser.

---

## Frontend UI

The React frontend provides a chat interface at `http://localhost:5173` with:

- **Chat canvas** — send prompts and view responses
- **Model selector** — choose Auto (intelligent routing) or a specific model
- **Model status** — monitor loaded models and VRAM usage
- **Agent trace** — view the plan→execute→retrieve→synthesize pipeline steps
- **Deliverable downloads** — download generated Word/PowerPoint/Excel files
- **Sovereignty monitor** — view network breach detection status

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health status (OS, tier, models, sentinel) |
| `GET` | `/models` | List all available models and routing options |
| `POST` | `/chat` | Agent chat — plan, execute, retrieve, synthesize |
| `GET` | `/download?filename=` | Download a generated file from `workspace/outputs/` |
| `POST` | `/upload` | Upload a file to `workspace/sandbox_files/` |
| `POST` | `/test/sandbox` | Execute code in Docker sandbox |
| `POST` | `/test/sentinel` | Trigger synthetic network leak test |
| `POST` | `/test/audit` | Verify audit log hash chain integrity |
| `POST` | `/ingest` | Ingest documents into knowledge base |
| `GET` | `/audit/log` | Read all audit log entries |
| `GET` | `/audit/last` | Get most recent audit entry |
| `POST` | `/models/load` | Load a model into GPU memory |
| `POST` | `/generate` | Generate text with a loaded model |
| `GET` | `/benchmark` | Run accuracy benchmark |

### Example: Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Create a word document named report.docx with title Q4 Report and content Revenue increased 15%."}'
```

### Example: Download File

```bash
curl -O "http://localhost:8000/download?filename=report.docx"
```

---

## Project Structure

```
sovereign-ai-workbench/
├── backend/
│   ├── main.py                     # FastAPI server + all endpoints
│   ├── config.py                   # Hardware tier detection, model roster
│   ├── core/
│   │   ├── audit_log.py            # Tamper-evident SHA-256 hash-chained log
│   │   ├── model_manager.py        # VRAM-aware model hot-swap + MockLLM
│   │   ├── sandbox_manager.py      # Docker code execution sandbox
│   │   ├── router.py               # Semantic task router (CODE/FILE/VISION/TEXT)
│   │   └── auth.py                 # 2-role RBAC (engineer/manager)
│   ├── infra/
│   │   ├── sentinel_runner.py      # Network egress sentinel (psutil/eBPF)
│   │   └── egress_trace.c          # BPF egress trace (future BCC integration)
│   ├── agents/
│   │   ├── graph.py                # LangGraph state machine (plan→execute→synthesize)
│   │   ├── planner.py              # ReWOO task decomposer
│   │   ├── executor.py             # Tool dispatcher
│   │   └── verifier.py             # Citation grounding verifier
│   ├── tools/
│   │   ├── path_safety.py          # Shared path traversal containment
│   │   ├── calculator.py           # SymPy symbolic math (safe parser)
│   │   ├── doc_generator.py        # Word .docx generation
│   │   ├── ppt_generator.py        # PowerPoint .pptx generation
│   │   ├── spreadsheet_generator.py # Excel .xlsx generation
│   │   ├── spreadsheet_analyzer.py  # Excel .xlsx reading
│   │   ├── file_io.py              # Sandboxed file read/write
│   │   ├── rag_search.py           # Hybrid BM25 + Qdrant RAG
│   │   ├── citation_tagger.py      # Source citation tagging
│   │   ├── confidence_helpers.py   # Low-confidence warning helpers
│   │   ├── pid_extractor.py        # P&ID topology extraction
│   │   ├── handwriting_triage.py   # Handwriting OCR
│   │   └── photo_analyzer.py       # Equipment nameplate analysis
│   └── ingestion/
│       ├── pdf_processor.py        # PDF text extraction
│       ├── email_processor.py      # .msg/.eml email processing
│       └── chunker.py              # Text chunking for RAG
├── frontend/
│   ├── src/
│   │   ├── App.tsx                 # Main app component
│   │   ├── components/
│   │   │   ├── ChatCanvas.tsx      # Chat interface
│   │   │   ├── ModelSelector.tsx   # Model selection dropdown
│   │   │   └── ModelStatus.tsx     # VRAM/model status display
│   │   └── hooks/
│   │       └── useApi.ts           # API client hooks
│   ├── package.json                # Frontend dependencies
│   └── vite.config.ts              # Vite build config
├── workspace/
│   ├── sandbox_files/              # Agent file I/O sandbox
│   └── outputs/                    # Generated deliverables
├── data/
│   ├── knowledge_base/             # Ingested documents
│   ├── audit_log.jsonl             # Tamper-evident audit log
│   └── audit_checkpoints.jsonl     # Audit log checkpoints
├── models/                         # GGUF model files
├── tests/
│   ├── conftest.py                 # Pytest fixtures
│   ├── test_step1.py               # Inference/sandbox/sentinel tests
│   ├── test_step2.py               # Agent orchestration tests
│   ├── test_step3.py               # RAG/citations/verifier tests
│   ├── test_step4.py               # Deliverable synthesis tests
│   └── ...                         # Adversarial audit tests
├── docs/                           # Benchmark results, demo scripts
├── setup_windows.bat               # Windows one-click setup
├── start_windows.bat               # Windows one-click start
├── requirements.txt                # Python dependencies
├── docker-compose.yml              # Docker infrastructure
├── Dockerfile.agent                # Agent container image
└── .gitignore
```

---

## Running Tests

### Quick Test

```cmd
venv\Scripts\activate
pytest tests/ -v
```

### Step-by-Step Test Suites

```cmd
REM Step 4: Deliverable synthesis tools (35-test adversarial audit)
pytest tests/test_step4.py -v

REM Step 3: RAG, citations, verifier
pytest tests/test_step3.py -v

REM Step 2: Agent orchestration
pytest tests/test_adversarial_step2_full.py -v

REM Step 1: Inference, sandbox, sentinel (24 tests)
pytest tests/test_adversarial_step1.py -v

REM Planner routing regression tests
pytest tests/test_planner_routing.py -v

REM Model selection and auto-routing tests
pytest tests/test_model_selection.py -v

REM Breach counter regression test
pytest tests/test_breach_counter.py -v
```

### Frontend Tests

```cmd
cd frontend
npm test
```

---

## Docker Setup (Optional)

For production-grade sandboxed code execution with network isolation:

### Prerequisites
- [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)

### Start Infrastructure

```cmd
docker-compose up -d
```

This starts:
- **PostgreSQL** (port 5432) — metadata and configuration
- **Qdrant** (port 6333) — vector database for RAG
- **Agent** (port 8000) — the FastAPI backend in an isolated network

The `sovereign-net` Docker network is **internal** — containers can talk to each other but have **no route to the public internet**.

### Security Note

On Windows, Docker Desktop uses WSL2 or Hyper-V. The egress sentinel's iptables enforcement is Linux-only; on Windows it operates in **log-only mode** (psutil monitoring without kernel-level blocking).

---

## Configuration

### Hardware Tiers

| Tier | VRAM | Use Case |
|------|------|----------|
| `BUILD` | 4 GB | Development, testing, small models |
| `DEMO` | 8 GB+ | Full demo with larger models |

Set via `HARDWARE_TIER` env var or let the system auto-detect via `nvidia-smi`.

### Auto Model Routing

When `model=auto`, the system selects the best model based on task intent:

| Task Type | Model | Trigger Keywords |
|-----------|-------|-----------------|
| Reasoning/Math | DeepSeek R1 7B | calculate, solve, equation, step-by-step |
| Code/Deliverables | Qwen 2.5 Coder 7B | code, script, docx, pptx, xlsx |
| Vision/OCR | LLaVA 7B | image, scan, photo, diagram, P&ID |
| Deep Synthesis | Phi-4 14B | architecture, comprehensive, deep dive |
| General Chat | Qwen 2.5 7B Instruct | default fallback |

### RBAC Roles

| Role | Access |
|------|--------|
| `engineer` | All collections except `financials_restricted` |
| `manager` | Full access to all collections |

Pass `?role=manager` to `/chat` to access restricted content.

---

## Troubleshooting

### "Python is not installed or not in your PATH"

1. Install Python from [python.org](https://www.python.org/downloads/)
2. **Check "Add Python to PATH"** during installation
3. Restart your terminal

### "Node.js is not installed or not in your PATH"

1. Install Node.js LTS from [nodejs.org](https://nodejs.org/)
2. Restart your terminal

### "llama_cpp not installed" warning

This is normal if you don't have a GPU or GGUF models. The system runs in MockLLM mode. To use real models:

```cmd
pip install llama-cpp-python
```

For CUDA GPU support:
```cmd
set CMAKE_ARGS=-DGGML_CUDA=on
pip install llama-cpp-python --force-reinstall
```

### Port 8000 already in use

```cmd
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Frontend shows "Failed to fetch"

Ensure the backend is running on port 8000. The frontend proxies API calls to `http://localhost:8000`.

### Docker not available

The system works without Docker. Code execution falls back to a subprocess stub. All other features (chat, RAG, document generation, calculator) work fully without Docker.

### Tests fail with "signal.SIGALRM"

This is a Linux-only signal. The code now handles Windows with threading-based timeouts. If you see this error, make sure you're running the latest code.

---

## Security Features

- **Path traversal containment** — All file operations are sandboxed to `workspace/`
- **Code injection prevention** — Calculator uses safe expression parsing with restricted locals
- **DoS protection** — SymPy solve calls have a 5-second timeout
- **Input validation** — Filename length limits, null-byte rejection, XML sanitization
- **Network sovereignty** — Sentinel monitors and kills unauthorized network connections
- **Audit logging** — SHA-256 hash-chained JSONL with truncation detection
- **RBAC** — Role-based access control for knowledge base queries

---

## License

[Add your license here]
