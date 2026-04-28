from typing import Dict, List, Optional

from pydantic import BaseModel


class URLRequest(BaseModel):
    url: str
    user_id: Optional[str] = None


class URLResponse(BaseModel):
    url: str
    verdict: str
    confidence: float
    threshold_used: float
    reasons: List[str]
    model_breakdown: Dict[str, float]


class SignupRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    username: str
    api_key: str
