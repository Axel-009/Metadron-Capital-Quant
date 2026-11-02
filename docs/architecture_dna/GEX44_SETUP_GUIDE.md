# GEX44 Setup Guide — Metadron Capital Production Deployment

> **Server:** Hetzner GEX44 (purchased April 2026)
> **GPU:** NVIDIA RTX 4000 SFF Ada — 20 GB GDDR6 ECC
> **CPU:** Intel Core i5-13500 (14 cores / 20 threads)
> **RAM:** 64 GB DDR4
> **Storage:** 2x 512 GB NVMe
> **Location:** Falkenstein (FSN1)

---

## Phase 1: OS & Base Setup

After Hetzner provisions the server, you'll get root SSH access.

```bash
# 1. Connect to server
ssh root@<your-server-ip>

# 2. Update system
apt update && apt upgrade -y

# 3. Set timezone to US Eastern (market hours)
timedv2ctl set-timezone America/New_York

# 4. Create non-root user
adduser metadron
usermod -aG sudo metadron
su - metadron

# 5. Set up SSH keys (from your local machine)
# ssh-copy-id metadron@<your-server-ip>

# 6. Harden SSH (disable root login, password auth)
sudo sed -i 's/PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl restart sshd

# 7. Firewall
sudo ufw allow 22/tcp       # SSH
sudo ufw allow 80/tcp       # HTTP
sudo ufw allow 443/tcp      # HTTPS
sudo ufw enable
```

---

## Phase 2: NVIDIA Drivers & CUDA

```bash
# 1. Install NVIDIA driver (GEX44 comes with Ubuntu, may have driver pre-installed)
# Check first:
nvidia-smi

# If not installed:
sudo apt install -y nvidia-driver-550 nvidia-utils-550
sudo reboot

# 2. Verify GPU
nvidia-smi
# Should show: NVIDIA RTX 4000 SFF Ada, 20480 MiB

# 3. Install CUDA toolkit (for llama.cpp CUDA build)
sudo apt install -y nvidia-cuda-toolkit
nvcc --version
```

---

## Phase 3: Core Dependencies

```bash
# 1. Python 3.11+ (system or pyenv)
sudo apt install -y python3 python3-pip python3-venv python3-dev

# 2. Node.js 20 LTS (for PM2, Express proxy, React client)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# 3. PM2 (global)
sudo npm install -g pm2

# 4. Build tools (for llama.cpp compilation)
sudo apt install -y build-essential cmake git wget curl

# 5. PostgreSQL
sudo apt install -y postgresql postgresql-contrib
sudo -u postgres createuser metadron
sudo -u postgres createdb metadron_db -O metadron

# 6. Redis
sudo apt install -y redis-server
sudo systemctl enable redis-server

# 7. Nginx
sudo apt install -y nginx certbot python3-certbot-nginx
```

---

## Phase 4: llama.cpp — Build with CUDA

```bash
# 1. Clone llama.cpp
cd /opt
sudo mkdir -p llama.cpp && sudo chown metadron:metadron llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp

# 2. Build with CUDA support
cd /opt/llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)

# 3. Verify CUDA build
./build/bin/llama-cli --version
# Should mention CUDA support
```

---

## Phase 5: Download Qwen2.5 Models

```bash
# 1. Create model directory
mkdir -p /opt/models

# 2. Install huggingface-cli
pip3 install huggingface-hub

# 3. Download Qwen2.5-7B-Instruct GGUF (FP16 — 14 GB)
# For FP16 quality on 20 GB VRAM:
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
  qwen2.5-7b-instruct-fp16.gguf \
  --local-dir /opt/models/

# 4. Download Qwen2.5-Math-7B-Instruct GGUF (Q4_K_M — ~5 GB)
# Quantized to leave room for main model:
huggingface-cli download Qwen/Qwen2.5-Math-7B-Instruct-GGUF \
  qwen2.5-math-7b-instruct-q4_k_m.gguf \
  --local-dir /opt/models/

# 5. Verify downloads
ls -lh /opt/models/*.gguf
# qwen2.5-7b-instruct-fp16.gguf         ~14 GB
# qwen2.5-math-7b-instruct-q4_k_m.gguf  ~5 GB
# TOTAL: ~19 GB — fits in 20 GB VRAM ✓
```

### VRAM Budget

```
RTX 4000 SFF Ada: 20,480 MiB (20 GB)
├── Qwen2.5-7B-Instruct FP16:  ~14,000 MiB
├── Qwen2.5-Math-7B Q4_K_M:    ~5,000 MiB
├── KV cache overhead:          ~1,000 MiB
└── REMAINING:                  ~480 MiB
    └── Tight but workable with llama.cpp LRU model management
```

**Alternative if VRAM is too tight:** Use Q8_0 for the main model instead
of FP16 (~8 GB instead of 14 GB), giving more headroom for KV cache:

```bash
# Q8 alternative (95%+ quality, more VRAM headroom)
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
  qwen2.5-7b-instruct-q8_0.gguf \
  --local-dir /opt/models/
```

---

## Phase 6: llama.cpp Server Configuration

```bash
# Start llama.cpp server with multi-model support
/opt/llama.cpp/build/bin/llama-server \
  --model /opt/models/qwen2.5-7b-instruct-fp16.gguf \
  --alias qwen2.5-7b \
  --host 127.0.0.1 \
  --port 8100 \
  --n-gpu-layers 99 \
  --ctx-size 4096 \
  --parallel 4 \
  --cont-batching \
  --flash-attn

# Key flags:
# --n-gpu-layers 99     → offload ALL layers to GPU
# --ctx-size 4096       → context window (increase if needed, costs VRAM)
# --parallel 4          → handle 4 concurrent requests
# --cont-batching       → continuous batching for throughput
# --flash-attn          → flash attention (reduces VRAM for KV cache)
```

### Multi-Model with LRU (if loading both models)

llama.cpp supports loading multiple models with automatic eviction.
For the GEX44's tight VRAM budget, the safest approach is:

**Option A:** Run ONE model at a time, swap via PM2 restart (simplest):
- Daytime: Qwen2.5-7B FP16 handles everything
- Only swap to Math-7B for overnight batch jobs

**Option B:** Run two llama.cpp instances with split VRAM:
```bash
# Instance 1: General (port 8100) — lower VRAM allocation
/opt/llama.cpp/build/bin/llama-server \
  --model /opt/models/qwen2.5-7b-instruct-q8_0.gguf \
  --port 8100 --n-gpu-layers 99 --ctx-size 2048 --parallel 2

# Instance 2: Math (port 8101) — lower VRAM allocation
/opt/llama.cpp/build/bin/llama-server \
  --model /opt/models/qwen2.5-math-7b-instruct-q4_k_m.gguf \
  --port 8101 --n-gpu-layers 99 --ctx-size 2048 --parallel 2
```

**Recommended: Start with Option A** (single model). The 7B-Instruct handles
sentiment classification AND numerical tasks well enough. Only split if you
find the Math-7B gives measurably better results on factor calculations.

---

## Phase 7: Deploy Metadron Codebase

```bash
# 1. Clone the repo
cd /opt
sudo mkdir -p metadron && sudo chown metadron:metadron metadron
git clone https://github.com/Axel-009/Metadron-Fund-Local.git /opt/metadron

# 2. Python virtual environment
cd /opt/metadron
python3 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt  # or pyproject.toml deps
pip install openbb xgboost hmmlearn scipy numpy pandas rich

# 4. Install Node.js dependencies
cd /opt/metadron/client && npm install && npm run build
cd /opt/metadron/server && npm install
cd /opt/metadron/mirofish/frontend && npm install && npm run build

# 5. Environment variables
cat > /opt/metadron/.env << 'ENVEOF'
# Broker
BROKER_TYPE=paper
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_PAPER_TRADE=true

# OpenBB
OBB_TOKEN=your_openbb_token

# LLM — Local (llama.cpp OpenAI-compatible endpoint)
LOCAL_LLM_BASE_URL=http://127.0.0.1:8100/v1
LOCAL_LLM_MODEL=qwen2.5-7b

# LLM — API (Mimo v2 for complex reasoning)
MIMO_API_KEY=your_mimo_key
MIMO_BASE_URL=https://api.mimo.ai/v1

# Database
DATABASE_URL=postgresql://metadron@localhost/metadron_db

# Ports
ENGINE_API_PORT=8001
PORT=5000
ENVEOF

# 6. Run tests
cd /opt/metadron
source venv/bin/activate
python3 -m pytest tests/ -v
```

---

## Phase 8: PM2 Ecosystem

```bash
# Create ecosystem file
cat > /opt/metadron/ecosystem.config.js << 'EOF'
module.exports = {
  apps: [
    // === LOCAL LLM SERVER (llama.cpp, GPU) ===
    {
      name: "llm-server",
      script: "/opt/llama.cpp/build/bin/llama-server",
      args: [
        "--model", "/opt/models/qwen2.5-7b-instruct-fp16.gguf",
        "--alias", "qwen2.5-7b",
        "--host", "127.0.0.1",
        "--port", "8100",
        "--n-gpu-layers", "99",
        "--ctx-size", "4096",
        "--parallel", "4",
        "--cont-batching",
        "--flash-attn"
      ].join(" "),
      interpreter: "none",
      autorestart: true,
      max_restarts: 10,
      restart_delay: 10000,
      env: {
        CUDA_VISIBLE_DEVICES: "0",
      },
    },

    // === CORE ENGINE (Python, 24/7) ===
    {
      name: "metadron-engine",
      script: "/opt/metadron/venv/bin/python3",
      args: "-u engine/live_loop_orchestrator.py",
      cwd: "/opt/metadron",
      interpreter: "none",
      autorestart: true,
      max_restarts: 50,
      restart_delay: 5000,
      max_memory_restart: "8G",
      env: {
        PYTHONUNBUFFERED: "1",
        VIRTUAL_ENV: "/opt/metadron/venv",
        PATH: "/opt/metadron/venv/bin:/usr/local/bin:/usr/bin:/bin",
      },
    },

    // === FASTAPI BACKEND (port 8001) ===
    {
      name: "metadron-api",
      script: "/opt/metadron/venv/bin/uvicorn",
      args: "app.backend.main:app --host 0.0.0.0 --port 8001 --workers 2",
      cwd: "/opt/metadron",
      interpreter: "none",
      autorestart: true,
      max_memory_restart: "2G",
      env: {
        VIRTUAL_ENV: "/opt/metadron/venv",
        PATH: "/opt/metadron/venv/bin:/usr/local/bin:/usr/bin:/bin",
      },
    },

    // === EXPRESS PROXY (port 5000) ===
    {
      name: "metadron-proxy",
      script: "npx",
      args: "tsx server/index.ts",
      cwd: "/opt/metadron",
      autorestart: true,
      env: {
        PORT: 5000,
        ENGINE_API_PORT: 8001,
      },
    },

    // === MIROFISH BACKEND (port 5001) ===
    {
      name: "mirofish-backend",
      script: "/opt/metadron/venv/bin/python3",
      args: "mirofish/backend/run.py",
      cwd: "/opt/metadron",
      interpreter: "none",
      autorestart: true,
      env: {
        VIRTUAL_ENV: "/opt/metadron/venv",
        PATH: "/opt/metadron/venv/bin:/usr/local/bin:/usr/bin:/bin",
      },
    },

    // === OVERNIGHT LEARNER (cron: weekdays 20:00 ET) ===
    {
      name: "metadron-learner",
      script: "/opt/metadron/venv/bin/python3",
      args: "-u engine/monitoring/learning_loop.py",
      cwd: "/opt/metadron",
      interpreter: "none",
      cron_restart: "0 20 * * 1-5",
      autorestart: false,
      env: {
        VIRTUAL_ENV: "/opt/metadron/venv",
        PATH: "/opt/metadron/venv/bin:/usr/local/bin:/usr/bin:/bin",
      },
    },
  ],
};
EOF

# Start all services
cd /opt/metadron
pm2 start ecosystem.config.js

# Save for reboot persistence
pm2 save
pm2 startup  # follow the instructions it prints

# Verify all running
pm2 status
pm2 logs --lines 20
```

---

## Phase 9: Nginx Reverse Proxy + SSL

```nginx
# /etc/nginx/sites-available/metadron
server {
    listen 80;
    server_name your-domain.com;

    # React client (static build)
    location / {
        root /opt/metadron/client/dist;
        try_files $uri $uri/ /index.html;
    }

    # Express proxy → FastAPI engine
    location /api/ {
        proxy_pass http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }

    # SSE streaming (long-lived connections)
    location /api/stream/ {
        proxy_pass http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 86400s;
    }

    # MiroFish
    location /mirofish/ {
        alias /opt/metadron/mirofish/frontend/dist/;
        try_files $uri $uri/ /mirofish/index.html;
    }

    location /mirofish/api/ {
        proxy_pass http://127.0.0.1:5001/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
# Enable site and SSL
sudo ln -s /etc/nginx/sites-available/metadron /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# SSL (once DNS is pointed to server)
sudo certbot --nginx -d your-domain.com
```

---

## Phase 10: Verification Checklist

```bash
# 1. GPU is working
nvidia-smi
# → RTX 4000 SFF Ada, 20480 MiB

# 2. llama.cpp is serving
curl http://127.0.0.1:8100/health
# → {"status":"ok"}

# 3. Test inference
curl http://127.0.0.1:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "messages": [{"role": "user", "content": "Classify sentiment: Apple beats Q3 earnings estimates by 15%"}],
    "max_tokens": 50
  }'
# → Should return BULLISH classification

# 4. PM2 services
pm2 status
# → llm-server: online
# → metadron-engine: online
# → metadron-api: online
# → metadron-proxy: online
# → mirofish-backend: online
# → metadron-learner: stopped (waits for cron)

# 5. Engine health
curl http://127.0.0.1:8001/health
# → {"status":"ok"}

# 6. Full pipeline test
cd /opt/metadron && source venv/bin/activate
python3 -m pytest tests/ -v

# 7. Website
curl -I https://your-domain.com
# → 200 OK
```

---

## Daily Operations

```bash
# Monitor everything
pm2 monit

# View engine logs
pm2 logs metadron-engine --lines 100

# View LLM server logs
pm2 logs llm-server --lines 50

# Check GPU usage
watch -n 1 nvidia-smi

# Restart after code update
cd /opt/metadron && git pull origin main
pm2 reload metadron-engine metadron-api metadron-proxy

# Full restart
pm2 restart all
```

---

## Overnight Model Swap (Optional, for InvestLM via Air-LLM)

If you want to run InvestLM-65B overnight for deep analysis:

```bash
# Create swap script
cat > /opt/metadron/scripts/overnight_swap.sh << 'SWAPEOF'
#!/bin/bash
# Stop daytime llama.cpp (frees GPU VRAM)
pm2 stop llm-server

# Run InvestLM via Air-LLM (layer-by-layer, uses ~4 GB VRAM)
cd /opt/metadron
source venv/bin/activate
python3 -c "
from engine.ml.bridges.air_llm_bridge import run_overnight_analysis
run_overnight_analysis()
"

# Restart daytime model before market open
pm2 start llm-server
SWAPEOF

chmod +x /opt/metadron/scripts/overnight_swap.sh

# Add to crontab: run at 20:00 ET weekdays
crontab -e
# 0 20 * * 1-5 /opt/metadron/scripts/overnight_swap.sh >> /var/log/metadron-overnight.log 2>&1
```

---

## Cost Summary

```
FIXED MONTHLY:
├── Hetzner GEX44:              EUR 184.00
├── Domain + DNS:               EUR   1.00
├── Backups (Hetzner Storage):  EUR   5.00
└── TOTAL FIXED:                EUR 190.00

VARIABLE (API only):
├── Mimo v2 API (reasoning):    EUR  50-100 (estimated)
├── OpenBB Pro (if needed):     EUR   0-30
├── Alpaca (paper = free):      EUR   0
└── TOTAL VARIABLE:             EUR  50-130

GRAND TOTAL:                    EUR 240-320/month
```

All sentiment classification, quant analysis, and numerical tasks run
locally on Qwen2.5-7B at ZERO marginal cost. Only complex multi-turn
reasoning (12 investor personas) goes to Mimo v2 API.
