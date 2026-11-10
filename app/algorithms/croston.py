"""Intermittent / lumpy demand: Croston family.

variant (parameters.variant):
  - "tsb" (default) — Teunter-Syntetos-Babai. Demand size is smoothed when demand
    occurs; demand *probability* is smoothed every period, so a run of zero periods
    lowers the forecast and it follows the trend toward zero.
  - "classic" — Croston (SAP APO classic): size and interval are smoothed only when
    demand occurs, so trailing zeros do not change the forecast.

Fitted values are one-step-ahead (made before seeing the period), so error metrics
are not flattered by the current observation.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def _clip_smoothing(value: Any, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = default
    if not np.isfinite(v):
        v = default
    return min(max(v, 0.01), 1.0)


def _tsb(hist: np.ndarray, horizon: int, alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    nonzero = hist[hist > 0]
    z = float(nonzero[0])  # demand size
    first_idx = int(np.argmax(hist > 0))
    p = 1.0 / max(1, first_idx + 1)  # demand probability (initial: first occurrence rate)
    fitted = np.zeros(hist.size)
    for i, v in enumerate(hist):
        fitted[i] = p * z
        occurred = 1.0 if v > 0 else 0.0
        if occurred:
            z = z + alpha * (float(v) - z)
        p = p + beta * (occurred - p)
    fc = np.full(horizon, float(p * z))
    return fc, fitted


def _classic(hist: np.ndarray, horizon: int, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    positions = np.flatnonzero(hist > 0)
    s = float(hist[positions[0]])
    q = float(positions[1] - positions[0]) if positions.size > 1 else 1.0
    last_nonzero = None
    fitted = np.zeros(hist.size)
    for i, v in enumerate(hist):
        fitted[i] = s / max(q, 1e-9) if last_nonzero is not None else 0.0
        if v > 0:
            if last_nonzero is not None:
                s = alpha * float(v) + (1 - alpha) * s
                q = alpha * float(i - last_nonzero) + (1 - alpha) * q
            else:
                s = float(v)
            last_nonzero = i
    fc = np.full(horizon, float(s / max(q, 1e-9)))
    return fc, fitted


def forecast_croston(
    history: np.ndarray,
    horizon: int,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float]:
    hist = np.asarray(history, dtype=float).ravel()
    hist = hist[np.isfinite(hist)]
    if hist.size == 0:
        return np.zeros(horizon), np.array([]), 0.0
    if not np.any(hist > 0):
        return np.zeros(horizon), np.zeros(hist.size), 0.0

    alpha = _clip_smoothing(params.get("alpha"), 0.1)
    variant = str(params.get("variant") or params.get("method") or "tsb").lower().strip()
    if variant in ("classic", "croston", "sba_off"):
        fc, fitted = _classic(hist, horizon, alpha)
    else:
        beta = _clip_smoothing(params.get("beta"), alpha)
        fc, fitted = _tsb(hist, horizon, alpha, beta)

    resid = hist - fitted
    resid = resid[np.isfinite(resid)]
    std = float(np.std(resid)) if resid.size else 0.0
    return fc, fitted, std
