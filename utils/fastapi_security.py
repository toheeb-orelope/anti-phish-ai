# utils/security.py
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

# -------------------------
# API Key Security
# -------------------------
API_KEY = "secure_api_key_12345"  # <-- replace with env var in production
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


async def verify_api_key(api_key: str = Depends(api_key_header)):
    """Check API key validity for all protected endpoints."""
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid or missing API key."
        )
