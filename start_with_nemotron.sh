#!/bin/bash
# =============================================================================
# Glid Surge Optimization - Full Stack Startup with Nemotron 49B
# Optimized for GX10 (DGX Spark) with Grace Blackwell GB10
# =============================================================================
# 
# Quick Start:
#   ./start_with_nemotron.sh demo   # NIM + API + Dashboard (recommended)
#   ./start_with_nemotron.sh nim    # Only NIM server
#   ./start_with_nemotron.sh llm    # Only vLLM server  
#
# =============================================================================

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       Glid Surge Optimization - GX10 Deployment              ║"
echo "║       Nemotron 49B + GNN + Dashboard                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

MODEL_PATH="/home/asus/Desktop/Nemotron 49B"
PROJECT_DIR="/home/asus/Desktop/Glid_Surge_Optimization"
MODE="${1:-demo}"

# ---------- Port helpers (avoid conflicts with existing servers) ----------
port_in_use() {
    local port="$1"
    # Prefer ss if available
    if command -v ss >/dev/null 2>&1; then
        ss -ltn 2>/dev/null | awk '{print $4}' | grep -E "[:.]${port}$" -q
        return $?
    fi
    # Fallback to lsof
    if command -v lsof >/dev/null 2>&1; then
        lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
        return $?
    fi
    # Last resort: /dev/tcp (can be slow / not always enabled)
    (echo >/dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1 && return 0 || return 1
}

pick_free_port() {
    local start_port="$1"
    local max_tries="${2:-50}"
    local p="$start_port"
    local i=0
    while [ $i -lt $max_tries ]; do
        if ! port_in_use "$p"; then
            echo "$p"
            return 0
        fi
        p=$((p + 1))
        i=$((i + 1))
    done
    echo "$start_port"
    return 1
}

configure_ports() {
    # Defaults (match docs), but auto-shift if already taken (e.g. existing uvicorn/api or frontend)
    # Prefer putting NIM on 8001 (common to reserve 8000 for local Python APIs)
    NIM_PORT="${NIM_PORT:-$(pick_free_port 8001 100)}"
    # Demo FastAPI backend defaults to 8002 (auto-shifts if taken)
    API_PORT="${API_PORT:-$(pick_free_port 8002 100)}"
    DASHBOARD_PORT="${DASHBOARD_PORT:-$(pick_free_port 3000 50)}"

    export NIM_PORT API_PORT DASHBOARD_PORT

    # Env vars consumed by Next.js route handlers
    export API_BASE_URL="http://localhost:${API_PORT}"
    export LLM_DIRECT_URL="http://localhost:${NIM_PORT}"
    export GLID_API_BASE_URL="http://localhost:${API_PORT}"
    export NEXT_PUBLIC_GLID_API_BASE_URL="http://localhost:${API_PORT}"
}

# Always configure ports early so we can print consistent URLs
configure_ports

# Load .env if exists
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(cat "$PROJECT_DIR/.env" | grep -v '^#' | xargs)
fi

check_local_model() {
    echo -e "${BLUE}[1/5]${NC} Checking local Nemotron 49B model..."
    if [ -d "$MODEL_PATH" ]; then
        MODEL_SIZE=$(du -sh "$MODEL_PATH" | cut -f1)
        echo -e "${GREEN}✓${NC} Model found at: $MODEL_PATH ($MODEL_SIZE)"
    else
        echo -e "${RED}✗${NC} Model not found at: $MODEL_PATH"
        echo "  Please ensure the model is downloaded to the Desktop/Nemotron 49B folder"
        exit 1
    fi
}

# Only require local model files for non-NIM modes
if [[ "$MODE" == "all" || "$MODE" == "llm" || "$MODE" == "native" ]]; then
    check_local_model
else
    echo -e "${BLUE}[1/5]${NC} Local model directory not required for NIM mode"
fi

# Check GPU availability
echo -e "\n${BLUE}[2/5]${NC} Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU query failed")
    echo -e "${GREEN}✓${NC} GPU detected: $GPU_INFO"
else
    echo -e "${YELLOW}!${NC} nvidia-smi not found - GPU status unknown"
fi

# Function to start demo with NIM (maximum performance)
start_demo() {
    echo -e "\n${CYAN}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  DEMO MODE - Using NVIDIA NIM for Maximum Performance${NC}"
    echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
    
    # Check NGC API Key
    if [ -z "$NGC_API_KEY" ]; then
        echo -e "\n${RED}✗${NC} NGC_API_KEY not set!"
        echo "  Run ./setup_ngc.sh first, or:"
        echo "  export NGC_API_KEY=<your-key>"
        exit 1
    fi
    echo -e "${GREEN}✓${NC} NGC_API_KEY configured"
    
    cd "$PROJECT_DIR"
    
    echo -e "\n${BLUE}[3/5]${NC} Starting Nemotron 49B via NVIDIA NIM..."
    echo "  Using TensorRT-LLM optimizations for Blackwell GB10"
    echo "  This may take 3-8 minutes for initial model load..."
    if [ "$NIM_PORT" != "8001" ]; then
        echo -e "  ${YELLOW}!${NC} Port 8001 is busy; using NIM_PORT=${NIM_PORT} instead"
    fi
    if [ "$API_PORT" != "8002" ]; then
        echo -e "  ${YELLOW}!${NC} Port 8002 is busy; using API_PORT=${API_PORT} instead"
    fi
    if [ "$DASHBOARD_PORT" != "3000" ]; then
        echo -e "  ${YELLOW}!${NC} Port 3000 is busy; using DASHBOARD_PORT=${DASHBOARD_PORT} instead"
    fi
    
    # Start NIM in background
    docker compose -f docker-compose.nim.yml up -d nemotron-nim
    
    echo -e "\n${BLUE}[4/5]${NC} Waiting for NIM server to be ready..."
    
    # Wait for NIM to be ready (longer timeout for initial load)
    MAX_WAIT=600  # 10 minutes
    WAITED=0
    while [ $WAITED -lt $MAX_WAIT ]; do
        if curl -s "http://localhost:${NIM_PORT}/v1/health/ready" > /dev/null 2>&1; then
            echo -e "\n${GREEN}✓${NC} NIM server is ready!"
            break
        fi
        if curl -s "http://localhost:${NIM_PORT}/v1/models" > /dev/null 2>&1; then
            echo -e "\n${GREEN}✓${NC} NIM server is ready!"
            break
        fi
        echo -n "."
        sleep 5
        WAITED=$((WAITED + 5))
    done
    
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo -e "\n${YELLOW}!${NC} NIM server not ready after ${MAX_WAIT}s"
        echo "  Check logs with: docker logs nemotron-nim-server"
        echo "  Continuing with API and Dashboard..."
    fi
    
    echo -e "\n${BLUE}[5/5]${NC} Starting API and Dashboard..."
    
    # Start FastAPI in background (uses port 8001 to avoid conflict with NIM on 8000)
    cd "$PROJECT_DIR"
    uvicorn src.api.server:app --host 0.0.0.0 --port "$API_PORT" &
    API_PID=$!
    echo "  FastAPI started on port ${API_PORT} (PID: $API_PID)"
    
    sleep 3
    
    # Start Next.js dashboard
    PORT="$DASHBOARD_PORT" npm run dev &
    DASHBOARD_PID=$!
    echo "  Dashboard started on port ${DASHBOARD_PORT} (PID: $DASHBOARD_PID)"
    
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  🚀 DEMO READY - Powered by NVIDIA NIM on GX10${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "  Services:"
    echo "  ├── Nemotron 49B (NIM):  http://localhost:${NIM_PORT}"
    echo "  ├── FastAPI Backend:     http://localhost:${API_PORT}"
    echo "  └── Dashboard:           http://localhost:${DASHBOARD_PORT}"
    echo ""
    echo "  Performance Features:"
    echo "  ├── TensorRT-LLM acceleration"
    echo "  ├── Blackwell NVFP4 optimizations"
    echo "  ├── Prefix caching enabled"
    echo "  └── 128GB unified memory utilization"
    echo ""
    echo "  Quick Tests:"
    echo "  curl http://localhost:${NIM_PORT}/v1/models"
    echo "  curl -X POST http://localhost:${NIM_PORT}/v1/chat/completions \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"model\": \"nemotron-49b\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
    echo ""
    echo "  To stop all services:"
    echo "  docker compose -f docker-compose.nim.yml down && kill $API_PID $DASHBOARD_PID"
    echo ""
    
    # Keep script running
    wait
}

# Function to start services with vLLM (development)
start_services() {
    echo -e "\n${BLUE}[3/5]${NC} Starting Nemotron 49B LLM server (vLLM)..."
    echo "  This may take 2-5 minutes for model loading..."
    
    cd "$PROJECT_DIR"
    
    # Start Nemotron in background with docker-compose
    docker compose -f docker-compose.nemotron.yml up -d nemotron-llm
    
    echo -e "\n${BLUE}[4/5]${NC} Waiting for LLM server to be ready..."
    
    # Wait for LLM to be ready (with timeout)
    MAX_WAIT=300  # 5 minutes
    WAITED=0
    while [ $WAITED -lt $MAX_WAIT ]; do
        if curl -s http://localhost:5000/health > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} LLM server is ready!"
            break
        fi
        echo -n "."
        sleep 5
        WAITED=$((WAITED + 5))
    done
    
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo -e "\n${YELLOW}!${NC} LLM server not ready after ${MAX_WAIT}s - continuing anyway"
        echo "  Check logs with: docker logs nemotron-49b-server"
    fi
    
    echo -e "\n${BLUE}[5/5]${NC} Starting API and Dashboard..."
    
    # Start FastAPI in background
    cd "$PROJECT_DIR"
    # Override LLM_BASE_URL for vLLM on port 5000
    export LLM_BASE_URL="http://localhost:5000/v1"
    python -m src.api.server &
    API_PID=$!
    echo "  FastAPI started (PID: $API_PID)"
    
    # Wait for API to be ready
    sleep 3
    
    # Start Next.js dashboard
    npm run dev &
    DASHBOARD_PID=$!
    echo "  Dashboard started (PID: $DASHBOARD_PID)"
    
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  All services started successfully!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "  Services:"
    echo "  ├── Nemotron 49B (vLLM): http://localhost:5000"
    echo "  ├── FastAPI Backend:    http://localhost:8000"
    echo "  └── Dashboard:          http://localhost:3000"
    echo ""
    echo "  API Endpoints:"
    echo "  ├── LLM Chat:          POST http://localhost:8000/llm/chat"
    echo "  ├── Route Compute:     POST http://localhost:8000/routes/compute"
    echo "  └── Surge Predict:     POST http://localhost:8000/predict/surge"
    echo ""
    echo "  Quick Tests:"
    echo "  curl http://localhost:5000/v1/models"
    echo "  curl http://localhost:8000/llm/status"
    echo ""
    echo "  To stop all services:"
    echo "  docker compose -f docker-compose.nemotron.yml down"
    echo "  kill $API_PID $DASHBOARD_PID"
    echo ""
    
    # Keep script running
    wait
}

# Function to start only LLM (for separate deployment)
start_llm_only() {
    echo -e "\n${BLUE}Starting Nemotron 49B LLM server only...${NC}"
    cd "$PROJECT_DIR"
    docker compose -f docker-compose.nemotron.yml up nemotron-llm
}

# Function to run with native vLLM (no Docker)
start_native() {
    echo -e "\n${BLUE}Starting Nemotron 49B with native vLLM...${NC}"
    echo "Requires: pip install vllm==0.9.2"
    echo ""
    
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port 5000 \
        --served-model-name nemotron-49b \
        --tensor-parallel-size 1 \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.90 \
        --enforce-eager \
        --dtype bfloat16 \
        --enable-auto-tool-choice \
        --tool-parser-plugin "$MODEL_PATH/llama_nemotron_toolcall_parser_no_streaming.py" \
        --tool-call-parser llama_nemotron_json
}

# Function to run with NVIDIA NIM (enterprise optimizations)
start_nim() {
    echo -e "\n${BLUE}Starting Nemotron 49B with NVIDIA NIM...${NC}"
    echo "Requires: NGC_API_KEY environment variable"
    echo ""
    
    if [ -z "$NGC_API_KEY" ]; then
        echo -e "${RED}✗${NC} NGC_API_KEY not set!"
        echo "  Get your key from: https://org.ngc.nvidia.com/setup"
        echo "  Then run: export NGC_API_KEY=<your-key>"
        exit 1
    fi
    
    cd "$PROJECT_DIR"
    docker compose -f docker-compose.nim.yml up nemotron-nim
}

# Parse arguments
case "$MODE" in
    demo)
        start_demo
        ;;
    all)
        start_services
        ;;
    llm)
        start_llm_only
        ;;
    native)
        start_native
        ;;
    nim)
        start_nim
        ;;
    help|--help|-h)
        echo ""
        echo "Usage: $0 [demo|all|llm|native|nim]"
        echo ""
        echo "Commands:"
        echo "  demo   - [DEFAULT] Start with NIM + API + Dashboard (recommended for demos)"
        echo "  all    - Start with vLLM + API + Dashboard (development)"
        echo "  llm    - Start only the vLLM server (Docker)"
        echo "  native - Start vLLM natively (no Docker)"
        echo "  nim    - Start only the NIM server"
        echo ""
        echo "For demos, NIM provides maximum performance on GX10 with:"
        echo "  - TensorRT-LLM backend"
        echo "  - Blackwell NVFP4 optimizations"
        echo "  - Higher throughput and lower latency"
        echo ""
        echo "Prerequisites for NIM:"
        echo "  1. Run ./setup_ngc.sh to configure NGC API key"
        echo "  2. Ensure docker is logged into nvcr.io"
        echo ""
        exit 0
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run '$0 help' for usage information."
        exit 1
        ;;
esac

