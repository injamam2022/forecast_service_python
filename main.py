"""
Enterprise Demand Forecast API — FastAPI replacement for R Plumber forecast_api.R

Endpoints:
  GET  /health
  GET  /capabilities
  POST /forecast

Security:
  - FORECAST_API_KEY is required. The service refuses to start without it unless
    FORECAST_DEV_MODE=1 (local development only).
  - Request bodies larger than FORECAST_MAX_BODY_MB (default 200) are rejected.
"""

from __future__ import annotations

import hmac
import logging
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from app.engine import run_forecast

load_dotenv()

logging.basicConfig(
    level=os.getenv("FORECAST_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("forecast_service")

API_KEY = os.getenv("FORECAST_API_KEY", "").strip()
DEV_MODE = os.getenv("FORECAST_DEV_MODE", "").strip().lower() in ("1", "true", "yes")
MAX_BODY_BYTES = int(float(os.getenv("FORECAST_MAX_BODY_MB", "200")) * 1024 * 1024)

if not API_KEY and not DEV_MODE:
    raise RuntimeError(
        "FORECAST_API_KEY is not set. Set it in .env (same value as the API's PY_FORECAST_API_KEY), "
        "or set FORECAST_DEV_MODE=1 for local development without a key."
    )
if not API_KEY:
    logger.warning("FORECAST_DEV_MODE: running WITHOUT an API key — do not expose this service.")

app = FastAPI(
    title="Enterprise Demand Forecast API",
    version="1.0.0",
    description="SAP IBP-grade demand forecasting (Python)",
)


@app.get("/health")
def health():
    from app.engine import ENGINE_INFO

    return {
        "status": "OK",
        "service": "Enterprise Forecast API (Python)",
        "version": "1.0.0",
        "engine": ENGINE_INFO,
    }


def _check_auth(x_api_key: Optional[str]) -> None:
    if not API_KEY:
        return
    if not x_api_key or not hmac.compare_digest(str(x_api_key).encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/capabilities")
def capabilities(x_api_key: Optional[str] = Header(default=None, alias="X-API-KEY")):
    """Algorithms this engine runs, so the API can validate profiles before a run."""
    _check_auth(x_api_key)
    from app.engine import ALGORITHM_ALIASES, SUPPORTED_ALGORITHMS
    from app.selection import SUPPORTED_MEASURES

    return {
        "algorithms": SUPPORTED_ALGORITHMS,
        "aliases": ALGORITHM_ALIASES,
        "selectionStrategies": ["best_fit", "stability", "first", "manual", "ensemble_average", "ensemble_median"],
        "errorMeasures": list(SUPPORTED_MEASURES),
    }


def _validate_payload(payload: Any) -> None:
    """Shape check before any work: clear 400s instead of deep stack traces."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")
    series = payload.get("timeSeries")
    if not isinstance(series, list) or not series:
        raise HTTPException(status_code=400, detail="timeSeries missing")
    for i, ts in enumerate(series):
        if not isinstance(ts, dict):
            raise HTTPException(status_code=400, detail=f"timeSeries[{i}] must be an object")
        if not str(ts.get("planningObjectId") or "").strip():
            raise HTTPException(status_code=400, detail=f"timeSeries[{i}].planningObjectId missing")
        if ts.get("series") is not None and not isinstance(ts.get("series"), list):
            raise HTTPException(status_code=400, detail=f"timeSeries[{i}].series must be a list")
    fp = payload.get("forecastPeriods")
    if fp is not None:
        try:
            if int(fp) < 1 or int(fp) > 1000:
                raise ValueError
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="forecastPeriods must be between 1 and 1000")


@app.post("/forecast")
async def forecast(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-KEY"),
):
    _check_auth(x_api_key)
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Request body too large")
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Request body too large")
    try:
        import json

        payload: Dict[str, Any] = json.loads(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    _validate_payload(payload)

    # The forecast itself is CPU-bound: run it in a worker thread so the event loop
    # (and /health) stays responsive during long runs.
    from starlette.concurrency import run_in_threadpool

    try:
        result = await run_in_threadpool(run_forecast, payload)
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

    host = os.getenv("FORECAST_HOST", "127.0.0.1")
    port = int(os.getenv("FORECAST_PORT", "8001"))
    workers = max(1, int(os.getenv("FORECAST_WORKERS", "1")))
    uvicorn.run("main:app", host=host, port=port, reload=False, workers=workers)
