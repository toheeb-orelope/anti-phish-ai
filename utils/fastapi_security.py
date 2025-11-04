# utils/security.py
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from .user_store import get_user_by_api_key

# -------------------------
# API Key Security
# -------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


async def verify_api_key(api_key: str = Depends(api_key_header)):
    """Check API key validity for all protected endpoints."""
    user = get_user_by_api_key(api_key)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid or missing API key."
        )
    return user
