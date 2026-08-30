# Sovereign AI Workbench

A sovereign, air-gapped agentic AI system with four security components under test.

## Architecture

| Component | File | Purpose |
|-----------|------|---------|
| VRAM-Tiered Model Hot-Swapping | `backend/core/model_manager.py` | Loads, evicts, and generates from GGUF models with VRAM-aware scheduling |
| eBPF Egress Sentinel | `backend/infra/sentinel_runner.py` + `egress_trace.c` | Active enforcement of network sovereignty via psutil polling + optional BPF kprobes |
| gVisor Code Sandbox | Docker `runsc` runtime | Kernel-level sandboxing for untrusted code execution |
| Tamper-Evident Audit Log | `backend/core/audit_log.py` | SHA-256 hash-chained JSONL logger with truncation detection |

## Security Layers

The system uses **two independent security layers** for network isolation:

1. **Primary: Docker Network Isolation** (`network=none` / `internal: true`) — structural fail-closed at the kernel netns level, independent of any userspace process
2. **Secondary: Sentinel Detection** (psutil polling + SIGKILL) — detects and kills offending processes, but is a secondary layer that depends on the sentinel process being running

## Running Tests

```bash
# Full 24-test adversarial audit suite
python3 -m pytest tests/test_adversarial_step1.py -v

# Individual component tests
python3 -m pytest tests/test_adversarial_step1.py::TestHotSwap -v
python3 -m pytest tests/test_adversarial_step1.py::TestEgressSentinel -v
python3 -m pytest tests/test_adversarial_step1.py::TestGVisorSandbox -v
python3 -m pytest tests/test_adversarial_step1.py::TestAuditLog -v
```

## Test Results

**24/24 PASS** — verified stable across multiple consecutive runs.

| # | Component | Test | Result |
|---|-----------|------|--------|
| 1 | Hot-Swap | Concurrent tier requests | PASS |
| 2 | Hot-Swap | 50 load/evict cycles | PASS |
| 3 | Hot-Swap | Oversized model rejection | PASS |
| 4 | Hot-Swap | Free vs total VRAM query | PASS |
| 5 | Hot-Swap | Corrupt model recovery | PASS |
| 6 | Hot-Swap | Wall-clock swap cycle | PASS |
| 7 | eBPF Sentinel | IPv6 outbound connection | PASS |
| 8 | eBPF Sentinel | DNS exfiltration (UDP/53) | PASS |
| 9 | eBPF Sentinel | Fail-closed on sentinel crash | PASS |
| 10 | eBPF Sentinel | Bytes before kill | PASS |
| 11 | eBPF Sentinel | IP literal egress bypass | PASS |
| 12 | eBPF Sentinel | Forked child egress hook | PASS |
| 13 | eBPF Sentinel | 20 concurrent egress | PASS |
| 14 | gVisor Sandbox | /proc host info leakage | PASS |
| 15 | gVisor Sandbox | Raw socket (SOCK_RAW) | PASS |
| 16 | gVisor Sandbox | Fork bomb containment | PASS |
| 17 | gVisor Sandbox | Memory exhaustion loop | PASS |
| 18 | gVisor Sandbox | Session bleed | PASS |
| 19 | gVisor Sandbox | Oversized file write | PASS |
| 20 | Audit Log | Delete last 5 entries | PASS |
| 21 | Audit Log | Concurrent logging integrity | PASS |
| 22 | Audit Log | kill-9 gap detection | PASS |
| 23 | Audit Log | Malformed entries | PASS |
| 24 | Compound | Sentinel crash + resource exhaustion | PASS |

## Key Findings

### Audit Log Truncation Detection (Fixed)
- `CHECKPOINT_INTERVAL` reduced from 10 to 1 (checkpoint every entry)
- `verify_chain()` now sets `valid=False` on any tampering signal
- Tail deletion of even 1 entry is now detected

### gVisor /proc Classification
- gVisor exposes container-scoped information (hostname, kernel version) in `/proc`
- All leaked info is **container-scoped**, not host-scoped
- Container hostname differs from host, `/proc/version` shows synthetic gVisor kernel

### eBPF Program Loading
- BCC 0.29.1 installed but incompatible with kernel 7.0.11 headers (17 compile errors)
- Docker network isolation provides the structural defense instead
- Sentinel psutil polling works as secondary detection layer

### Test Isolation
- AuditLogger singleton `_next_sequence` counter resets between tests via `conftest.py` autouse fixture
- Each test gets unique log file via pytest `tmp_path` fixture

## Project Structure

```
sovereign-ai-workbench/
├── backend/
│   ├── core/
│   │   ├── audit_log.py          # Tamper-evident audit log
│   │   ├── model_manager.py      # VRAM-aware model hot-swapping
│   │   └── router.py             # Semantic task router
│   ├── infra/
│   │   ├── sentinel_runner.py    # eBPF/psutil egress sentinel
│   │   └── egress_trace.c        # BPF kprobe program
│   ├── agents/
│   │   ├── graph.py              # LangGraph state machine
│   │   ├── planner.py            # Plan generation
│   │   └── executor.py           # Step execution
│   ├── tools/
│   │   └── file_io.py            # Sandboxed file I/O
│   └── main.py                   # FastAPI server
├── tests/
│   ├── conftest.py               # AuditLogger singleton reset fixture
│   ├── test_adversarial_step1.py # Full 24-test adversarial audit
│   └── ...
├── docker-compose.yml
├── Dockerfile.agent
├── requirements.txt
└── README.md
```
