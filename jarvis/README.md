# Jarvis — integration and setup

Jarvis is a **read-only** voice + text assistant for the platform, built on
[OpenJarvis](https://github.com/open-jarvis/OpenJarvis) (vendored in `vendor/OpenJarvis`, Apache-2.0).
It can read platform state, answer, speak and *recommend*. It can never trade, change config or
write files: every write action is converted into a recommendation that you approve
(permission guard: `engine/agents/nanoclaw/permission_guard.py`).

## Pieces

| Piece | File | Port |
|-------|------|------|
| Jarvis service (persona, chat, voice, recommendations) | `engine/bridges/jarvis_bridge.py` | 8006 |
| Engine API route `/api/jarvis/*` (status, messages, chat, voice, speak, recommendations) | `engine/api/routers/jarvis.py` | 8001 |
| Primary brain: Llama 3.1-8B server | `engine/bridges/llama_model_server.py` | 8005 |
| Fallback brain: LLM inference bridge | `engine/bridges/llm_inference_bridge.py` | 8002 |
| Permission guard (read / speak / recommend only) | `engine/agents/nanoclaw/permission_guard.py` | — |
| OpenJarvis speech (STT faster-whisper, TTS kokoro / piper) | `vendor/OpenJarvis` | — |
| Dashboard page (React) | `jarvis/ui/open-jarvis.tsx` | — |

## Setup

```bash
pip install -r requirements.txt
pip install -e vendor/OpenJarvis              # speech pipeline (optional — text works without it)
cp .env.example .env                          # fill in keys
export LLAMA_MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct   # needs a GPU; or rely on the bridge
pm2 start ecosystem.config.cjs                # engine-api, llama-model-server, llm-inference-bridge, jarvis-service
curl localhost:8001/api/jarvis/status
```

Without a GPU, skip `llama-model-server`; Jarvis falls back to the inference bridge. Without
`vendor/OpenJarvis` installed, voice is disabled and text chat still works (guarded imports).

## Planned (agent architecture)

Jarvis becomes the front desk for the Claude Managed Agents roster (macro analyst, events analyst,
quant reviewer, risk officer): it routes your questions to them and collects your approvals. Agents
read QuantConnect results through its API; none of them can place orders.
