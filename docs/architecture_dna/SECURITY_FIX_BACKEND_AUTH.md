# OUTSTANDING SECURITY FIX — MUST ADDRESS BEFORE FRONTEND BUILD

> **IMPORTANT**: This is a blocking security issue for the frontend build.
> Do NOT expose the FastAPI backend to any network until this is resolved.

---

## Issue #17: Backend API — Open CORS + Zero Authentication

**File**: `app/backend/main.py:62`

**Problem**:
1. **CORS wide open**: `allow_origins=["*"]` with `allow_credentials=True` — any website
   on the internet can make authenticated cross-origin requests to this API.
2. **No authentication middleware**: API keys are created and stored in the database
   (`app/backend/api/api_keys.py`) but **no middleware validates them on incoming requests**.
   Every endpoint is publicly accessible with no token, no header check, nothing.

**Current risk**: LOW — the backend returns hardcoded stubs (no real data or execution).
Live trading runs through `LiveLoopOrchestrator` via CLI, not the API.

**Future risk**: CRITICAL — once the frontend is built and the API is wired to the actual
engine, anyone who can reach the server could:
- Trigger pipeline runs
- View portfolio positions, NAV, P&L
- Potentially submit or cancel trades
- Access signal intelligence and alpha data

---

## Required Fixes for the Frontend Build

### 1. Restrict CORS to your domain only

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com", "http://localhost:5173"],  # Vite dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 2. Add API key validation middleware

```python
@app.middleware("http")
async def validate_api_key(request: Request, call_next):
    if request.url.path in ("/health", "/docs", "/openapi.json"):
        return await call_next(request)
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        return JSONResponse(status_code=401, content={"detail": "Missing API key"})
    # Validate against hashed keys in database
    db = next(get_db())
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    stored = db.query(ApiKey).filter_by(key_hash=key_hash, is_active=True).first()
    if not stored:
        return JSONResponse(status_code=403, content={"detail": "Invalid API key"})
    return await call_next(request)
```

### 3. Rate limiting (recommended)

```python
# pip install slowapi
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
```

---

## Implementation Checklist

- [ ] Restrict `allow_origins` to production domain + localhost dev
- [ ] Add `X-API-Key` header validation middleware using existing `ApiKey` table
- [ ] Add rate limiting (slowapi or similar) on all endpoints
- [ ] Add HTTPS enforcement (redirect HTTP → HTTPS)
- [ ] Wire hedge fund endpoints to actual engine (currently stubs)
- [ ] Add request logging / audit trail for all trade-related endpoints

---

*Identified during platform audit 2026-03-29*
*Priority: BLOCKING for frontend build — non-blocking for CLI-based live trading*
