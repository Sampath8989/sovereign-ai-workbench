#!/usr/bin/env bash
# ==============================================================================
# Sovereign AI Workbench - Linux & Pop!_OS Launcher
# Starts FastAPI backend (port 8000) and React frontend (port 5173) concurrently.
# Clean shutdown with Ctrl+C stops all background processes.
# ==============================================================================

set -m

# ANSI Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${CYAN}${BOLD}===================================================${NC}"
echo -e "${CYAN}${BOLD}   Starting Sovereign AI Workbench (Linux/Pop!_OS) ${NC}"
echo -e "${CYAN}${BOLD}===================================================${NC}\n"

# Verify virtual environment
if [ ! -f "venv/bin/python" ]; then
    echo -e "${RED}[!] Virtual environment not found. Run ./setup_linux.sh first.${NC}"
    exit 1
fi

# Load environment configuration
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Clean up stale processes on ports 8000 and 5173 if requested or lingering
clean_port() {
    local port=$1
    local pids=$(lsof -ti :$port 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo -e "${YELLOW}[!] Port $port is in use by PID(s): $pids. Cleaning up...${NC}"
        kill -9 $pids 2>/dev/null || true
        sleep 1
    fi
}

clean_port 8000
clean_port 5173

# Process Tracking
BACKEND_PID=0
FRONTEND_PID=0

cleanup() {
    echo -e "\n${YELLOW}${BOLD}[*] Gracefully stopping Sovereign AI Workbench...${NC}"
    if [ "$BACKEND_PID" -ne 0 ]; then
        pkill -P "$BACKEND_PID" 2>/dev/null || true
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ "$FRONTEND_PID" -ne 0 ]; then
        pkill -P "$FRONTEND_PID" 2>/dev/null || true
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    sleep 1
    # Fallback force-kill if still alive
    if [ "$BACKEND_PID" -ne 0 ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        kill -9 "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ "$FRONTEND_PID" -ne 0 ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        kill -9 "$FRONTEND_PID" 2>/dev/null || true
    fi
    clean_port 8000
    clean_port 5173
    echo -e "${GREEN}${BOLD}[✓] Workbench services stopped.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# 1. Start Backend
echo -e "${CYAN}[1/2] Launching Backend API on http://localhost:8000 ...${NC}"
./venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for Backend to become healthy
echo -e "      Waiting for backend to initialize..."
MAX_WAIT=20
WAIT_COUNT=0
BACKEND_ONLINE=false

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:8000/health" > /dev/null 2>&1; then
        BACKEND_ONLINE=true
        break
    fi
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
done

if [ "$BACKEND_ONLINE" = true ]; then
    echo -e "  ${GREEN}✓${NC} Backend is online: ${BOLD}http://localhost:8000/health${NC}"
else
    echo -e "  ${YELLOW}!${NC} Backend started (PID $BACKEND_PID), continuing..."
fi

# 2. Start Frontend
echo -e "\n${CYAN}[2/2] Launching Frontend UI on http://localhost:5173 ...${NC}"
if [ -d "frontend" ]; then
    (cd frontend && npm run dev -- --host 0.0.0.0 --port 5173) &
    FRONTEND_PID=$!
else
    echo -e "  ${YELLOW}!${NC} frontend/ directory not found. Skipping frontend UI launch."
fi

# 3. Open Browser if GUI is active
sleep 2
echo -e "\n${GREEN}${BOLD}===================================================${NC}"
echo -e "${GREEN}${BOLD}   Sovereign AI Workbench is Running!              ${NC}"
echo -e "${GREEN}${BOLD}   • Frontend UI: ${CYAN}http://localhost:5173${NC}"
echo -e "${GREEN}${BOLD}   • Backend API: ${CYAN}http://localhost:8000${NC}"
echo -e "${GREEN}${BOLD}   • API Docs:    ${CYAN}http://localhost:8000/docs${NC}"
echo -e "${GREEN}${BOLD}   Press [Ctrl+C] to stop all services.            ${NC}"
echo -e "${GREEN}${BOLD}===================================================${NC}\n"

if [ -n "$DISPLAY" ] || [ -n "$WAYLAND_DISPLAY" ]; then
    if command -v xdg-open &> /dev/null; then
        xdg-open "http://localhost:5173" > /dev/null 2>&1 || true
    fi
fi

# Wait on background processes
wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
