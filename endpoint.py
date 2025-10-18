# main.py
from fastapi import FastAPI, HTTPException, status, Depends, Request
import uvicorn

# Local imports
from run_xai import run_example, THRESHOLD
from utils.fastapi_security import verify_api_key
from utils.response_schemas import URLResponse, URLRequest
from utils.rate_limit import rate_limiter

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
# Server startup
# -------------------------
# if __name__ == "__main__":
# uvicorn.run("endpoint:app", host="0.0.0.0", port=8000, reload=True)
