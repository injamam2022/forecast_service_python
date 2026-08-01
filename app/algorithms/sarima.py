"""SARIMA seasonal ARIMA for trend + seasonality."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def forecast_sarima(
    history: np.ndarray,
    horizon: int,
    params: Dict[str, Any],
    seasonal_periods: int = 12,
) -> Tuple[np.ndarray, np.ndarray, float]:
    hist = np.asarray(history, dtype=float).ravel()
    hist = hist[np.isfinite(hist)]
    if hist.size == 0:
        return np.zeros(horizon), np.array([]), 0.0
    if hist.size < 4:
        mean = float(np.mean(hist))
        return np.full(horizon, mean), np.full(hist.size, mean), 0.0

    p = int(params.get("p") if params.get("p") is not None else 1)
    d = int(params.get("d") if params.get("d") is not None else 1)
    q = int(params.get("q") if params.get("q") is not None else 1)
    P = int(params.get("P") if params.get("P") is not None else 1)
    D = int(params.get("D") if params.get("D") is not None else 1)
    Q = int(params.get("Q") if params.get("Q") is not None else 1)
    m = int(params.get("m") or params.get("seasonal_periods") or seasonal_periods)

    # Disable seasonal if not enough history
    use_seasonal = hist.size >= m * 2
    seasonal_order = (P, D, Q, m) if use_seasonal else (0, 0, 0, 0)

    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        model = SARIMAX(
            hist,
            order=(p, d, q),
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)
        fc = np.asarray(fit.forecast(horizon), dtype=float)
        fitted = np.asarray(fit.fittedvalues, dtype=float)[: hist.size]
        if np.nanmin(hist) >= 0:
            fc = np.maximum(fc, 0.0)
        resid = hist - fitted
        resid = resid[np.isfinite(resid)]
        std = float(np.std(resid)) if resid.size else 0.0
        return fc, fitted, std
    except Exception:
        mean = float(np.mean(hist))
        return np.full(horizon, mean), np.full(hist.size, mean), 0.0
