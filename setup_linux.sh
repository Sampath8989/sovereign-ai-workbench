#!/usr/bin/env bash
# ==============================================================================
# Sovereign AI Workbench - Automated Linux & Pop!_OS Setup and Launch Script
# Compatible with Pop!_OS 22.04 / 24.04 LTS, Ubuntu, Debian, and Linux systems
# ==============================================================================

set -e

# ANSI Color Codes
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Defaults
SKIP_MODELS=false
MODEL_CHOICE="recommended"
NO_START=false
PREFER_3B=true

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --skip-models) SKIP_MODELS=true ;;
        --model) MODEL_CHOICE="$2"; shift ;;
        --no-start) NO_START=true ;;
        --prefer-3b) PREFER_3B=true ;;
        -h|--help)
            echo "Usage: ./setup_linux.sh [OPTIONS]"
            echo "Options:"
            echo "  --skip-models         Skip downloading/checking model weights"
            echo "  --model <name>        Model preset to install (default: recommended)"
            echo "                        Choices: recommended, 3b-suite, all-3b, llama-3b-q4, coder-3b-q4, general-3b-q4, fallback-0.5b, all"
            echo "  --no-start            Do not launch the workbench after setup"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo -e "${CYAN}${BOLD}===================================================${NC}"
echo -e "${CYAN}${BOLD}   Sovereign AI Workbench - Linux / Pop!_OS Setup  ${NC}"
echo -e "${CYAN}${BOLD}===================================================${NC}\n"

# ------------------------------------------------------------------------------
# 1. OS & Hardware Detection
# ------------------------------------------------------------------------------
echo -e "${CYAN}[1/6] Detecting Operating System & Hardware...${NC}"

OS_NAME="Linux"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_NAME="${PRETTY_NAME:-$NAME}"
fi
echo -e "  ${GREEN}✓${NC} Operating System: ${BOLD}${OS_NAME}${NC}"

TOTAL_RAM=$(free -h | awk '/^Mem:/ {print $2}')
CPU_MODEL=$(lscpu | grep "Model name" | sed 's/Model name:[ \t]*//' | head -n 1)
echo -e "  ${GREEN}✓${NC} CPU: ${CPU_MODEL} (${TOTAL_RAM} RAM)"

# NVIDIA GPU Detection
HAS_NVIDIA=false
VRAM_GB=0
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    TOTAL_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    if [ -n "$TOTAL_VRAM_MB" ]; then
        HAS_NVIDIA=true
        VRAM_GB=$(awk "BEGIN {printf \"%.1f\", $TOTAL_VRAM_MB/1024}")
        echo -e "  ${GREEN}✓${NC} GPU: ${BOLD}${GPU_NAME}${NC} (${VRAM_GB} GB VRAM detected)"
    fi
fi

if [ "$HAS_NVIDIA" = false ]; then
    echo -e "  ${YELLOW}!${NC} No NVIDIA GPU detected with nvidia-smi. Workbench will use CPU inference & MockLLM."
fi

# ------------------------------------------------------------------------------
# 2. Check System Prerequisites
# ------------------------------------------------------------------------------
echo -e "\n${CYAN}[2/6] Verifying System Prerequisites...${NC}"

MISSING_DEPS=()

# Check Git
if command -v git &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} git is installed"
else
    MISSING_DEPS+=("git")
fi

# Check Python 3
PYTHON_BIN=""
for py in python3.12 python3.11 python3.10 python3; do
    if command -v $py &> /dev/null; then
        PY_VER=$($py -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        PY_MAJOR=$($py -c "import sys; print(sys.version_info.major)")
        PY_MINOR=$($py -c "import sys; print(sys.version_info.minor)")
        if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 10 ]; then
            PYTHON_BIN=$(command -v $py)
            echo -e "  ${GREEN}✓${NC} Python ${PY_VER} found at ${PYTHON_BIN}"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    MISSING_DEPS+=("python3" "python3-venv" "python3-pip")
fi

# Check Node.js and npm
if command -v node &> /dev/null && command -v npm &> /dev/null; then
    NODE_VER=$(node -v)
    echo -e "  ${GREEN}✓${NC} Node.js ${NODE_VER} & npm are installed"
else
    MISSING_DEPS+=("nodejs" "npm")
fi

# Check build tools
if ! command -v gcc &> /dev/null || ! command -v g++ &> /dev/null; then
    MISSING_DEPS+=("build-essential")
fi

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo -e "\n${RED}[!] Missing required packages: ${MISSING_DEPS[*]}${NC}"
    echo -e "Please install them using your package manager, for example on Pop!_OS / Ubuntu / Debian:"
    echo -e "${YELLOW}  sudo apt update && sudo apt install -y ${MISSING_DEPS[*]}${NC}"
    exit 1
fi

# ------------------------------------------------------------------------------
# 3. Python Virtual Environment Setup
# ------------------------------------------------------------------------------
echo -e "\n${CYAN}[3/6] Setting up Python Virtual Environment...${NC}"

if [ ! -d "venv" ]; then
    echo -e "  [*] Creating virtual environment using ${PYTHON_BIN}..."
    $PYTHON_BIN -m venv venv
    echo -e "  ${GREEN}✓${NC} Virtual environment created in ./venv"
else
    echo -e "  ${GREEN}✓${NC} Virtual environment already exists."
fi

VENV_PY="./venv/bin/python"
VENV_PIP="./venv/bin/pip"

echo -e "  [*] Upgrading pip, setuptools, wheel..."
$VENV_PIP install --upgrade pip setuptools wheel --quiet

# ------------------------------------------------------------------------------
# 4. Backend Dependencies & Hardware-Optimized llama-cpp-python
# ------------------------------------------------------------------------------
echo -e "\n${CYAN}[4/6] Installing Backend Dependencies...${NC}"

# Check if llama-cpp-python is installed
if ! $VENV_PY -c "import llama_cpp" &> /dev/null; then
    if [ "$HAS_NVIDIA" = true ]; then
        echo -e "  [*] Installing llama-cpp-python with CUDA support..."
        $VENV_PIP install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 || \
        $VENV_PIP install llama-cpp-python
    else
        echo -e "  [*] Installing standard llama-cpp-python for CPU..."
        $VENV_PIP install llama-cpp-python
    fi
else
    echo -e "  ${GREEN}✓${NC} llama-cpp-python is already installed."
fi

$VENV_PIP install -r requirements.txt huggingface_hub --quiet
echo -e "  ${GREEN}✓${NC} Backend Python dependencies installed."

# ------------------------------------------------------------------------------
# 5. Frontend Setup & Environment Configuration
# ------------------------------------------------------------------------------
echo -e "\n${CYAN}[5/6] Configuring Frontend & Environment...${NC}"

if [ -d "frontend" ]; then
    echo -e "  [*] Installing frontend dependencies (npm install)..."
    (cd frontend && npm install --quiet)
    echo -e "  ${GREEN}✓${NC} Frontend dependencies ready."
fi

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "  ${GREEN}✓${NC} Created .env from .env.example"
    else
        cat << 'ENVEOF' > .env
# Sovereign AI Workbench - Environment Configuration
HARDWARE_TIER=BUILD
PREFER_3B_MODELS=true
USE_MOCK_LLM=false
SENTINEL_ENFORCE=false
QDRANT_HOST=localhost
QDRANT_PORT=6333
USE_MOCK_EMBEDDER=1
LOG_LEVEL=INFO
ENVEOF
        echo -e "  ${GREEN}✓${NC} Generated initial .env configuration."
    fi
fi

# Ensure 4GB VRAM optimizations are present in .env
if [ "$HAS_NVIDIA" = true ] && (( $(echo "$VRAM_GB < 6.0" | bc -l 2>/dev/null || echo 1) )); then
    if ! grep -q "PREFER_3B_MODELS" .env; then
        echo "PREFER_3B_MODELS=true" >> .env
    fi
    echo -e "  ${GREEN}✓${NC} Configured 3B Model tier preference for 4GB VRAM GPU."
fi

# Ensure workspace and models directories exist
mkdir -p models workspace/outputs workspace/sandbox_files data/knowledge_base

# ------------------------------------------------------------------------------
# 6. Model Weights Setup (Optimized for 3B Models on Pop!_OS)
# ------------------------------------------------------------------------------
echo -e "\n${CYAN}[6/6] Checking Local Models (3B Focus)...${NC}"

if [ "$SKIP_MODELS" = false ]; then
    $VENV_PY scripts/download_models.py --model "$MODEL_CHOICE"
else
    echo -e "  [*] Skipping model download per --skip-models flag."
fi

echo -e "\n${GREEN}${BOLD}===================================================${NC}"
echo -e "${GREEN}${BOLD}   Setup Completed Successfully on Linux Pop!_OS!  ${NC}"
echo -e "${GREEN}${BOLD}===================================================${NC}\n"

if [ "$NO_START" = false ]; then
    echo -e "${CYAN}[*] Starting Sovereign AI Workbench via ./start_linux.sh ...${NC}\n"
    exec ./start_linux.sh
else
    echo -e "To start the workbench at any time, run:"
    echo -e "  ${BOLD}./start_linux.sh${NC}\n"
fi
