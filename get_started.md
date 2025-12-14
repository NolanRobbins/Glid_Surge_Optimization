# 🚀 Get Started: Glid Surge Optimization on GX10

This guide explains how to run the full demo using **NVIDIA Nemotron 49B (NIM)** on the DGX Spark (GX10).

## 📋 Prerequisites

- **NVIDIA GX10 (DGX Spark)** system
- **NGC API Key** (for NIM container access)
- Docker & NVIDIA Container Toolkit installed

---

## ⚡ Quick Start

### 1. Setup NGC Credentials (One-time)
```bash
cd /home/asus/Desktop/Glid_Surge_Optimization
./setup_ngc.sh
```
*Follow the prompts to enter your NGC API Key.*

### 2. Launch Demo
```bash
./start_with_nemotron.sh demo
```
*This starts:*
- **Nemotron 49B (NIM)** on port `8000` (First run takes ~3-5 mins for TensorRT optimization)
- **FastAPI Backend** on port `8001`
- **Dashboard** on port `3000`

### 3. Open Dashboard
Visit: **http://localhost:3000**

---

## 🛠️ Verification

Check if services are healthy:

```bash
# Check NIM Model Status
curl http://localhost:8000/v1/models

# Test Chat Completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron-49b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

---

## 🛑 Stopping Services

To stop all containers and processes:

```bash
docker compose -f docker-compose.nim.yml down
pkill -f "uvicorn src.api.server"
pkill -f "next-server"
```

---

## 🔍 Troubleshooting

- **NIM not starting?** Check logs: `docker logs nemotron-nim-server`
- **Port conflicts?** Ensure port `8000` is free.
- **Model loading slow?** First run builds the TensorRT engine (~5 mins). Subsequent runs are instant.

