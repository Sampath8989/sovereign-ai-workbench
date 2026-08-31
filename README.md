# Sovereign AI Workbench

A production-ready, air-gapped AI workbench with sovereignty enforcement, hybrid RAG, multi-step agent orchestration, and deliverable synthesis tools.

## Architecture Overview

| Layer | Components | Purpose |
|-------|-----------|---------|
| **Step 1 — Inference & Sandbox** | Model Manager, gVisor Sandbox, eBPF Sentinel, Audit Log | VRAM-aware model hot-swapping, kernel-level code sandboxing, network egress enforcement, tamper-evident logging |
| **Step 2 — Agent Orchestration** | LangGraph ReWOO, Planner, Executor, Router, File I/O | Multi-step task decomposition, plan→execute→synthesize pipeline, semantic task routing |
| **Step 3 — Knowledge & Verification** | Hybrid RAG (BM25 + Qdrant), Citation Tagger, Chain-of-Thought Verifier | Hybrid sparse+dense search, source citation tagging, claim grounding verification |
| **Step 4 — Deliverable Synthesis** | Doc Generator, PPT Generator, Spreadsheet Generator/Analyzer, Symbolic Calculator | Word/PowerPoint/Excel generation, xlsx analysis, SymPy math engine, file download endpoint |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn backend.main:app --reload

# Health check
curl http://localhost:8000/health

# Chat with the agent
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Create a word document named report.docx with title Q4 Report and content Revenue increased 15%."}'

# Download a generated file
curl -O "http://localhost:8000/download?filename=report.docx"
```

## Deliverable Synthesis Tools (Step 4)

The agent can generate production deliverables from natural language prompts:

| Tool | Trigger Keywords | Output |
|------|-----------------|--------|
| **Word Generator** | "word document", "docx", "approval note" | `.docx` with heading + body |
| **PowerPoint Generator** | "powerpoint", "slides", "pptx" | `.pptx` with title + bullet points |
| **Spreadsheet Generator** | "spreadsheet", "xlsx", "excel" | `.xlsx` from 2D data |
| **Spreadsheet Analyzer** | (reads existing xlsx) | 2D cell data from specified range |
| **Symbolic Calculator** | "calculate", "solve", "math", "equation" | Step-by-step math solution |

### Security Hardening

All Step 4 tools passed a 35-test adversarial QA audit:

- **Path traversal containment** — All generators and the analyzer reject filenames that resolve outside their intended directory (`workspace/outputs/` or `workspace/sandbox_files/`). Shared `safe_resolve_output_path()` utility enforces this uniformly.
- **Code injection prevention** — The calculator uses `sympy.parsing.sympy_parser.parse_expr()` with a restricted locals dict (no Python builtins). A regex pre-filter blocks `__import__`, `exec`, `eval`, `os.*`, `sys.*`, `open()`, etc.
- **DoS protection** — Sympy `solve()` calls are wrapped in a 5-second timeout via `signal.SIGALRM`. Complex expressions return a clean error instead of hanging.
- **Input validation** — Filename length limits (200 chars), null-byte rejection, XML control-character sanitization, non-serializable cell value coercion.
- **Download endpoint safety** — `/download` validates filenames before path operations, returns 403 for traversal/null-bytes, 404 for missing files, correct MIME types for all formats.

## Security Layers

The system uses **two independent security layers** for network isolation:

1. **Primary: Docker Network Isolation** (`network=none` / `internal: true`) — structural fail-closed at the kernel netns level
2. **Secondary: Sentinel Detection** (psutil polling + optional eBPF kprobes) — detects and kills offending processes

## Running Tests

```bash
# Step 4: Deliverable synthesis tools (35-test adversarial audit)
pytest tests/test_step4.py -v

# Step 3: RAG, citations, verifier
pytest tests/test_step3.py -v

# Step 2: Agent orchestration
pytest tests/test_adversarial_step2_full.py -v

# Step 1: Inference, sandbox, sentinel (24 tests)
pytest tests/test_adversarial_step1.py -v
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health status |
| `POST` | `/chat` | Agent chat — plan, execute, synthesize |
| `GET` | `/download?filename=` | Download a generated file |
| `POST` | `/test/sandbox` | Execute code in gVisor sandbox |
| `POST` | `/test/sentinel` | Trigger synthetic network leak |
| `POST` | `/test/audit` | Verify audit log hash chain |
| `POST` | `/ingest` | Ingest documents into knowledge base |
| `GET` | `/audit/log` | Read all audit log entries |
| `POST` | `/models/load` | Load a model into GPU memory |
| `POST` | `/generate` | Generate text with a loaded model |

## Project Structure

```
sovereign-ai-workbench/
├── backend/
│   ├── core/
│   │   ├── audit_log.py            # Tamper-evident SHA-256 hash-chained log
│   │   ├── model_manager.py        # VRAM-aware model hot-swap + MockLLM
│   │   └── sandbox_manager.py      # gVisor code execution sandbox
│   ├── infra/
│   │   └── sentinel_runner.py      # eBPF/psutil egress sentinel
│   ├── ingestion/
│   │   ├── pdf_processor.py        # PDF text extraction
│   │   ├── email_processor.py      # .msg/.eml email processing
│   │   └── chunker.py              # Text chunking for RAG
│   ├── agents/
│   │   ├── graph.py                # LangGraph state machine (plan→execute→retrieve→synthesize)
│   │   ├── planner.py              # ReWOO task decomposer
│   │   ├── executor.py             # Tool dispatcher (file_io, llm, code, calculator, generators)
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
│   │   └── citation_tagger.py      # Source citation tagging
│   └── main.py                     # FastAPI server + /download endpoint
├── workspace/
│   ├── sandbox_files/              # Agent file I/O sandbox
│   └── outputs/                    # Generated deliverables
├── data/knowledge_base/            # Ingested documents
├── tests/
│   ├── test_step4.py               # Step 4 tool tests
│   ├── test_step3.py               # Step 3 RAG/verifier tests
│   ├── test_adversarial_step2_full.py # Step 2 adversarial audit
│   └── test_adversarial_step1.py   # Step 1 adversarial audit (24 tests)
├── models/                         # GGUF model files
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## Test Results

| Step | Suite | Tests | Status |
|------|-------|-------|--------|
| Step 1 | Adversarial audit (hot-swap, sentinel, sandbox, audit log) | 24 | ✅ All passing |
| Step 2 | Adversarial audit (ReWOO, router, file I/O) | 29 | ✅ All passing |
| Step 3 | RAG, citations, CoT verifier | 26 | ✅ All passing |
| Step 4 | Deliverable synthesis tools (35-test adversarial audit) | 35 | ✅ All passing |
| **Total** | | **114** | ✅ |

## Key Technical Decisions

- **ReWOO pattern** — Plans are generated upfront without intermediate observations, reducing LLM calls
- **Hybrid RAG** — BM25 sparse search + Qdrant dense vectors with reciprocal rank fusion
- **MockLLM fallback** — Deterministic responses when model files are absent, enables full pipeline testing without GPU
- **Cross-platform paths** — All file operations use `pathlib.Path` for Linux/Windows compatibility
- **UUID-based filenames** — MockLLM generates unique output filenames per request to prevent concurrent write collisions
