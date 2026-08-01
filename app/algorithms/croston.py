"""Croston's method for intermittent / lumpy demand (SAP APO classic)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def forecast_croston(
    history: np.ndarray,
    horizon: int,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float]:
    hist = np.asarray(history, dtype=float).ravel()
    hist = hist[np.isfinite(hist)]
    if hist.size == 0:
        return np.zeros(horizon), np.array([]), 0.0

    alpha = float(params.get("alpha") or 0.1)
    alpha = min(max(alpha, 0.01), 1.0)

    demand_sizes = []
    intervals = []
    last_pos = None
    for i, v in enumerate(hist):
        if v > 0:
            demand_sizes.append(float(v))
            if last_pos is not None:
                intervals.append(i - last_pos)
            last_pos = i

    if not demand_sizes:
        return np.zeros(horizon), np.zeros(hist.size), 0.0

    # Initialize
    s = demand_sizes[0]
    q = float(intervals[0]) if intervals else 1.0

    # Smooth chronologically
    pos = 0
    last_nonzero = None
    fitted = np.zeros(hist.size)
    for i, v in enumerate(hist):
        if v > 0:
            if last_nonzero is not None:
                interval = i - last_nonzero
                s = alpha * float(v) + (1 - alpha) * s
                q = alpha * float(interval) + (1 - alpha) * q
            else:
                s = float(v)
            last_nonzero = i
            pos += 1
        rate = s / max(q, 1e-9)
        fitted[i] = rate

    rate = s / max(q, 1e-9)
    fc = np.full(horizon, float(rate))
    resid = hist - fitted
    resid = resid[np.isfinite(resid)]
    std = float(np.std(resid)) if resid.size else 0.0
    return fc, fitted, std
