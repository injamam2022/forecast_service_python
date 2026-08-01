"""
Enterprise Demand Forecast API — FastAPI replacement for R Plumber forecast_api.R

Endpoints:
  GET  /health
  POST /forecast
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from app.engine import run_forecast

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("forecast_service")

API_KEY = os.getenv("FORECAST_API_KEY", "").strip()

app = FastAPI(
    title="Enterprise Demand Forecast API",
    version="1.0.0",
    description="SAP IBP-grade demand forecasting (Python)",
)


@app.get("/health")
def health():
    return {"status": "OK", "service": "Enterprise Forecast API (Python)", "version": "1.0.0"}


def _check_auth(x_api_key: Optional[str]) -> None:
    if not API_KEY:
        return
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/forecast")
async def forecast(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-KEY"),
):
    _check_auth(x_api_key)
    try:
        payload: Dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not payload.get("timeSeries"):
        raise HTTPException(status_code=400, detail="timeSeries missing")

    try:
        result = run_forecast(payload)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Forecast processing failed")
        return JSONResponse(
            status_code=500,
            content={"error": "Forecast processing failed", "details": str(e)},
        )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("FORECAST_HOST", "0.0.0.0")
    port = int(os.getenv("FORECAST_PORT", "8001"))
    uvicorn.run("main:app", host=host, port=port, reload=False)
