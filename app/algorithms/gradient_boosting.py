"""Real gradient boosting with lag features (XGBoost preferred, sklearn fallback)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def forecast_gradient_boosting(
    history: np.ndarray,
    horizon: int,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float]:
    hist = np.asarray(history, dtype=float).ravel()
    hist = hist[np.isfinite(hist)]
    if hist.size == 0:
        return np.zeros(horizon), np.array([]), 0.0
    if hist.size < 3:
        mean = float(np.mean(hist))
        return np.full(horizon, mean), np.full(hist.size, mean), 0.0

    n_estimators = int(params.get("n_estimators") or 100)
    learning_rate = float(params.get("learning_rate") or 0.1)
    max_depth = int(params.get("max_depth") or 3)
    lags = int(params.get("lags") or min(6, hist.size - 1))
    lags = max(1, min(lags, hist.size - 1))

    X, y = [], []
    for i in range(lags, hist.size):
        X.append(hist[i - lags : i])
        y.append(hist[i])
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # XGBoost is a required dependency. Without it the model fails (and is reported in
    # algorithmFailures) instead of silently switching to a different model, which made
    # the same profile produce different numbers on different machines.
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("gradient_boosting needs the xgboost package (pip install -r requirements.txt)") from exc

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="reg:squarederror",
        verbosity=0,
        random_state=int(params.get("random_state") or 0),
        n_jobs=1,
    )
    model.fit(X, y)

    # In-sample fitted (pad first lags with mean)
    fitted = np.full(hist.size, float(np.mean(hist)))
    preds = model.predict(X)
    fitted[lags:] = preds

    # Recursive multi-step forecast
    series = list(hist.astype(float))
    forecasts = []
    for _ in range(horizon):
        feat = np.asarray([series[-lags:]], dtype=float)
        pred = float(model.predict(feat)[0])
        if not np.isfinite(pred):
            pred = float(np.mean(series[-lags:]))
        if np.nanmin(hist) >= 0:
            pred = max(0.0, pred)
        forecasts.append(pred)
        series.append(pred)

    fc = np.asarray(forecasts, dtype=float)
    resid = hist - fitted
    resid = resid[np.isfinite(resid)]
    std = float(np.std(resid)) if resid.size else 0.0
    return fc, fitted, std
