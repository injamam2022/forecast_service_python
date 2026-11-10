"""Exponential smoothing: Single / Double / Triple / Auto."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def _auto_ets(hist: np.ndarray, horizon: int, seasonal_periods: int):
    """Pick the ETS structure with the lowest AICc; None when nothing fits."""
    try:
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    except Exception:
        return None
    seasonal_ok = seasonal_periods and seasonal_periods > 1 and hist.size >= seasonal_periods * 2
    configs = []
    for seasonal in ([None, "add"] if seasonal_ok else [None]):
        configs.append({"trend": None, "damped_trend": False, "seasonal": seasonal})
        if hist.size >= 4:
            configs.append({"trend": "add", "damped_trend": False, "seasonal": seasonal})
            configs.append({"trend": "add", "damped_trend": True, "seasonal": seasonal})
    best = None
    import warnings

    for cfg in configs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ETSModel(
                    hist,
                    error="add",
                    trend=cfg["trend"],
                    damped_trend=cfg["damped_trend"],
                    seasonal=cfg["seasonal"],
                    seasonal_periods=seasonal_periods if cfg["seasonal"] else None,
                )
                fit = model.fit(disp=False)
            score = float(getattr(fit, "aicc", np.nan))
            if not np.isfinite(score):
                score = float(getattr(fit, "aic", np.inf))
            if best is None or score < best[0]:
                best = (score, fit)
        except Exception:
            continue
    if best is None:
        return None
    fit = best[1]
    fc = np.asarray(fit.forecast(horizon), dtype=float)
    fitted = np.asarray(fit.fittedvalues, dtype=float)[: hist.size]
    resid = hist - fitted
    resid = resid[np.isfinite(resid)]
    std = float(np.std(resid)) if resid.size else 0.0
    return fc, fitted, std


def forecast_exponential_smoothing(
    history: np.ndarray,
    horizon: int,
    params: Dict[str, Any],
    seasonal_periods: int = 12,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    mode / variant:
      - single → SES (level only)
      - double → Holt (level + trend)
      - triple → Holt-Winters (level + trend + seasonal)
      - auto   → optimize ETS structure via statsmodels (fallback to Holt-Winters)
    """
    hist = np.asarray(history, dtype=float).ravel()
    hist = hist[np.isfinite(hist)]
    if hist.size == 0:
        return np.zeros(horizon), np.array([]), 0.0
    if hist.size == 1:
        return np.full(horizon, float(hist[0])), hist.copy(), 0.0

    mode = str(
        params.get("mode")
        or params.get("variant")
        or params.get("smoothingType")
        or "auto"
    ).lower().strip()

    alpha = params.get("alpha")
    beta = params.get("beta")
    gamma = params.get("gamma")

    # Resolve trend/seasonal from mode
    if mode in ("single", "ses", "simple"):
        trend, seasonal = None, None
    elif mode in ("double", "holt"):
        trend, seasonal = "add", None
    elif mode in ("triple", "holt_winters", "hw"):
        trend = "add"
        seasonal = "add" if hist.size >= seasonal_periods * 2 else None
    else:
        # auto — let statsmodels pick if possible; else double/triple by data length
        trend = "add"
        seasonal = "add" if hist.size >= seasonal_periods * 2 else None
        mode = "auto"

    # Allow explicit overrides
    if params.get("trend") is not None:
        t = params.get("trend")
        trend = None if t in (None, "none", "None", "") else t
    if params.get("seasonal") is not None:
        s = params.get("seasonal")
        seasonal = None if s in (None, "none", "None", "") else s

    if seasonal and hist.size < seasonal_periods * 2:
        seasonal = None

    try:
        if mode == "auto":
            # Real automatic ETS: fit level-only, additive trend and damped trend (each
            # with additive seasonality when ≥ 2 full seasons) and keep the lowest AICc.
            auto = _auto_ets(hist, horizon, seasonal_periods)
            if auto is not None:
                fc, fitted, std = auto
                if np.nanmin(hist) >= 0:
                    fc = np.maximum(fc, 0.0)
                return fc, fitted, std

        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        model = ExponentialSmoothing(
            hist,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods if seasonal else None,
            initialization_method="estimated",
        )
        fit_kwargs: Dict[str, Any] = {"optimized": True}
        # For single/double/triple with explicit params, pin smoothing if provided
        if alpha is not None and mode != "auto":
            fit_kwargs["smoothing_level"] = float(alpha)
            fit_kwargs["optimized"] = False
        if beta is not None and trend and mode != "auto":
            fit_kwargs["smoothing_trend"] = float(beta)
        if gamma is not None and seasonal and mode != "auto":
            fit_kwargs["smoothing_seasonal"] = float(gamma)

        fit = model.fit(**fit_kwargs)
        fc = np.asarray(fit.forecast(horizon), dtype=float)
        fitted = np.asarray(fit.fittedvalues, dtype=float)[: hist.size]
        resid = hist - fitted
        resid = resid[np.isfinite(resid)]
        std = float(np.std(resid)) if resid.size else 0.0
        if np.nanmin(hist) >= 0:
            fc = np.maximum(fc, 0.0)
        return fc, fitted, std
    except Exception:
        # Manual SES / Holt fallback
        a = float(alpha) if alpha is not None else 0.3
        b = float(beta) if beta is not None else 0.1
        level = float(hist[0])
        trend_v = float(hist[1] - hist[0]) if hist.size > 1 and mode not in ("single", "ses", "simple") else 0.0
        fitted_list = []
        for x in hist:
            fitted_list.append(level + trend_v)
            err = float(x) - (level + trend_v)
            level = level + trend_v + a * err
            if mode not in ("single", "ses", "simple"):
                trend_v = trend_v + b * err
        fitted = np.asarray(fitted_list, dtype=float)
        forecasts = []
        lvl, tr = level, trend_v
        for h in range(1, horizon + 1):
            forecasts.append(lvl + h * tr if mode not in ("single", "ses", "simple") else lvl)
        fc = np.asarray(forecasts, dtype=float)
        if np.nanmin(hist) >= 0:
            fc = np.maximum(fc, 0.0)
        resid = hist - fitted
        std = float(np.std(resid[np.isfinite(resid)])) if resid.size else 0.0
        return fc, fitted, std
