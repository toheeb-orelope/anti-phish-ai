# utils/rate_limit.py
import time
from fastapi import Request, HTTPException, status

# -------------------------
# Basic in-memory rate limiting
# -------------------------
RATE_LIMIT = {}
# Allow a higher burst rate to support browser extension scans
MAX_REQUESTS = 60  # allowed requests per window
WINDOW_SECONDS = 60  # per time window (seconds)


async def rate_limiter(request: Request):
    """Simple per-API-key rate limiting (fallback to client IP)."""
    # Prefer API key if provided, so users are isolated.
    api_key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
    identifier = api_key if api_key else (request.client.host if request.client else "unknown")

    now = time.time()

    if identifier not in RATE_LIMIT:
        RATE_LIMIT[identifier] = []

    # keep only requests within the window
    RATE_LIMIT[identifier] = [t for t in RATE_LIMIT[identifier] if now - t < WINDOW_SECONDS]

    if len(RATE_LIMIT[identifier]) >= MAX_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please wait a minute before retrying.",
        )

    RATE_LIMIT[identifier].append(now)
