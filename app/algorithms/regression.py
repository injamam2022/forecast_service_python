"""Trend regression forecasts (linear / polynomial / ridge / lasso)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def forecast_regression(
    history: np.ndarray,
    horizon: int,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float]:
    hist = np.asarray(history, dtype=float).ravel()
    hist = hist[np.isfinite(hist)]
    if hist.size == 0:
        return np.zeros(horizon), np.array([]), 0.0
    if hist.size == 1:
        return np.full(horizon, float(hist[0])), hist.copy(), 0.0

    method = str(params.get("method") or "linear").lower()
    t = np.arange(1, hist.size + 1, dtype=float)
    future_t = np.arange(hist.size + 1, hist.size + horizon + 1, dtype=float)

    try:
        if method == "polynomial":
            deg = int(params.get("degree") or 2)
            deg = max(1, min(deg, min(5, hist.size - 1)))
            coef = np.polyfit(t, hist, deg)
            fitted = np.polyval(coef, t)
            fc = np.polyval(coef, future_t)
        elif method in ("ridge", "lasso"):
            from sklearn.linear_model import Ridge, Lasso
            from sklearn.preprocessing import PolynomialFeatures

            X = t.reshape(-1, 1)
            Xp = PolynomialFeatures(degree=1, include_bias=True).fit_transform(X)
            model = Ridge(alpha=1.0) if method == "ridge" else Lasso(alpha=0.1, max_iter=5000)
            model.fit(Xp, hist)
            fitted = model.predict(Xp)
            Xf = PolynomialFeatures(degree=1, include_bias=True).fit_transform(future_t.reshape(-1, 1))
            fc = model.predict(Xf)
        else:
            # linear OLS
            coef = np.polyfit(t, hist, 1)
            fitted = np.polyval(coef, t)
            fc = np.polyval(coef, future_t)

        fitted = np.asarray(fitted, dtype=float)
        fc = np.asarray(fc, dtype=float)
        # Optional floor at 0 (off by default so downward trends can continue: 40→30→20 → 10,0,-10,-20)
        clamp = params.get("clampNonNegative")
        if clamp is None:
            clamp = params.get("nonNegative")
        if clamp is True or str(clamp).lower() in ("1", "true", "yes"):
            fc = np.maximum(fc, 0.0)
            fitted = np.maximum(fitted, 0.0)
        resid = hist - fitted
        resid = resid[np.isfinite(resid)]
        std = float(np.std(resid)) if resid.size else 0.0
        return fc, fitted, std
    except Exception:
        mean = float(np.mean(hist))
        return np.full(horizon, mean), np.full(hist.size, mean), 0.0
