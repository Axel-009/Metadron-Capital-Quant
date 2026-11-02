/**
 * Metadron Capital — PM2 process configuration (self-hosted services).
 *
 * The trading algorithm itself runs on QuantConnect (algorithm/). These are the
 * services you run yourself: the engine API, Jarvis and its model servers.
 *
 *   pm2 start ecosystem.config.cjs                      # everything below
 *   pm2 start ecosystem.config.cjs --only jarvis-service
 *   pm2 logs / pm2 monit
 */
const path = require('path');
const ROOT = __dirname;
const log = (n) => ({
  error_file: path.join(ROOT, `logs/pm2/${n}-error.log`),
  out_file: path.join(ROOT, `logs/pm2/${n}-out.log`),
  merge_logs: true,
});

module.exports = {
  apps: [
    // Engine API (FastAPI) — Jarvis reads platform state from here
    {
      name: 'engine-api', script: 'python3',
      args: '-m uvicorn engine.api.server:app --host 0.0.0.0 --port 8001 --log-level info',
      cwd: ROOT, interpreter: 'none',
      env: { PYTHONUNBUFFERED: '1', ENGINE_API_PORT: '8001', METADRON_BROKER_TYPE: 'paper' },
      autorestart: true, max_restarts: 10, restart_delay: 5000, ...log('engine-api'),
    },
    // Llama 3.1-8B model server — Jarvis's primary brain (needs a GPU)
    {
      name: 'llama-model-server', script: 'python3',
      args: '-m uvicorn engine.bridges.llama_model_server:create_app --factory --host 0.0.0.0 --port 8005 --log-level info',
      cwd: ROOT, interpreter: 'none',
      env: { PYTHONUNBUFFERED: '1', LLAMA_PORT: '8005',
             LLAMA_MODEL_PATH: process.env.LLAMA_MODEL_PATH || 'meta-llama/Llama-3.1-8B-Instruct' },
      autorestart: true, max_memory_restart: '8G', max_restarts: 5, restart_delay: 10000, ...log('llama-model-server'),
    },
    // LLM inference bridge — Jarvis fallback when the Llama server is down
    {
      name: 'llm-inference-bridge', script: 'python3',
      args: '-m uvicorn engine.bridges.llm_inference_bridge:create_app --factory --host 0.0.0.0 --port 8002 --log-level info',
      cwd: ROOT, interpreter: 'none',
      env: { PYTHONUNBUFFERED: '1', LLM_BRIDGE_PORT: '8002' },
      autorestart: true, max_restarts: 5, restart_delay: 10000, ...log('llm-inference-bridge'),
    },
    // Jarvis — read-only voice + text assistant (OpenJarvis, vendor/OpenJarvis)
    {
      name: 'jarvis-service', script: 'python3',
      args: '-m uvicorn engine.bridges.jarvis_bridge:create_app --factory --host 0.0.0.0 --port 8006 --log-level info',
      cwd: ROOT, interpreter: 'none',
      env: { PYTHONUNBUFFERED: '1', JARVIS_PORT: '8006', LLAMA_PORT: '8005', LLM_BRIDGE_PORT: '8002', ENGINE_API_PORT: '8001' },
      autorestart: true, max_memory_restart: '4G', max_restarts: 10, restart_delay: 5000, ...log('jarvis-service'),
    },
  ],
};
