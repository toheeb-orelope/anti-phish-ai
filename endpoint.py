# main.py
from fastapi import FastAPI, HTTPException, status, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Local imports
from run_xai import run_example, THRESHOLD
from utils.fastapi_security import verify_api_key
from utils.response_schemas import (
    URLResponse,
    URLRequest,
    SignupRequest,
    LoginRequest,
    LoginResponse,
)
from utils.rate_limit import rate_limiter
from utils.user_store import (
    create_user,
    authenticate,
    generate_api_key,
    set_user_api_key,
    get_public_user,
)

# -------------------------
# Initialize App
# -------------------------
app = FastAPI(
    title="Anti-Phish AI Detection API",
    description="Real-time phishing detection using hybrid AI model + Explainable AI.",
    version="1.0.0",
    openapi_tags=[
        {"name": "Prediction", "description": "Phishing detection endpoints"}
    ],
)

# CORS for local browser/extension calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# Root route
# -------------------------
@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    return {
        "message": "🚀 Anti-Phish AI API is running",
        "threshold": THRESHOLD,
        "documentation": "/docs",
    }


# -------------------------
# Prediction route
# -------------------------
@app.post(
    "/predict",
    response_model=URLResponse,
    dependencies=[Depends(verify_api_key), Depends(rate_limiter)],
    status_code=status.HTTP_200_OK,
)
async def predict_url(request_data: URLRequest):
    try:
        result = run_example(request_data.url)
        return URLResponse(
            url=result["url"],
            verdict=result["verdict"],
            confidence=result["confidence"],
            threshold_used=THRESHOLD,
            reasons=result.get("reasons", []),
            model_breakdown=result.get("model_breakdown", {}),
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {e}",
        )


# -------------------------
# Auth routes
# -------------------------
@app.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(payload: SignupRequest):
    try:
        user = create_user(payload.username, payload.password)
        return {"message": "User created", "user": user}
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {e}",
        )


@app.post("/login", response_model=LoginResponse, status_code=status.HTTP_200_OK)
async def login(payload: LoginRequest):
    user = authenticate(payload.username, payload.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )
    key = generate_api_key()
    updated = set_user_api_key(payload.username, key)
    return LoginResponse(username=updated["username"], api_key=updated["api_key"])


@app.get("/me", status_code=status.HTTP_200_OK)
async def me(current_user=Depends(verify_api_key)):
    public_user = (
        get_public_user(current_user["username"])
        if isinstance(current_user, dict)
        else None
    )
    if not public_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    return public_user


# -------------------------
# Server startup
# -------------------------
# if __name__ == "__main__":
# uvicorn.run("endpoint:app", host="0.0.0.0", port=8000, reload=True)
