# utils/rate_limit.py
import time
from fastapi import Request, HTTPException, status

# -------------------------
# Basic in-memory rate limiting
# -------------------------
RATE_LIMIT = {}
MAX_REQUESTS = 10  # allowed requests
WINDOW_SECONDS = 60  # per time window (seconds)


async def rate_limiter(request: Request):
    """Simple per-IP rate limiting using in-memory dict."""
    client_ip = request.client.host
    now = time.time()

    if client_ip not in RATE_LIMIT:
        RATE_LIMIT[client_ip] = []

    # keep only requests within the window
    RATE_LIMIT[client_ip] = [
        t for t in RATE_LIMIT[client_ip] if now - t < WINDOW_SECONDS
    ]

    if len(RATE_LIMIT[client_ip]) >= MAX_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please wait a minute before retrying.",
        )

    RATE_LIMIT[client_ip].append(now)
