# utils/schemas.py
from pydantic import BaseModel
from typing import Optional, List, Dict


# -------------------------
# Input request schema
# -------------------------
class URLRequest(BaseModel):
    url: str
    user_id: Optional[str] = None


# -------------------------
# Output response schema
# -------------------------
class URLResponse(BaseModel):
    url: str
    verdict: str
    confidence: float
    threshold_used: float
    reasons: List[str]
    model_breakdown: Dict[str, float]
