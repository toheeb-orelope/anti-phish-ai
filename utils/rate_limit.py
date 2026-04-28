import time
from collections import defaultdict

from fastapi import HTTPException, Request, status

RATE_LIMIT = defaultdict(list)
MAX_REQUESTS = 60
WINDOW_SECONDS = 60


def _get_rate_limit_identifier(request: Request) -> str:
    api_key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
    if api_key:
        return api_key
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


async def rate_limiter(request: Request):
    """Simple per-API-key rate limiting (fallback to client IP)."""
    identifier = _get_rate_limit_identifier(request)
    now = time.time()
    RATE_LIMIT[identifier] = [
        timestamp
        for timestamp in RATE_LIMIT[identifier]
        if now - timestamp < WINDOW_SECONDS
    ]

    if len(RATE_LIMIT[identifier]) >= MAX_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please wait a minute before retrying.",
        )

    RATE_LIMIT[identifier].append(now)
