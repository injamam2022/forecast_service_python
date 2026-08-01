"""Average-family forecasts: simple, SMA, weighted, WMA."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def _std_from_fit(hist: np.ndarray, fitted: np.ndarray) -> float:
    resid = hist - fitted[: hist.size]
    resid = resid[np.isfinite(resid)]
    return float(np.std(resid)) if resid.size else 0.0


def _normalize_weights(weights: List[float], n: int) -> np.ndarray:
    if not weights or len(weights) == 0:
        # Linear increasing weights (older → smaller)
        w = np.arange(1, n + 1, dtype=float)
    else:
        w = np.asarray(weights[:n], dtype=float)
        if w.size < n:
            # pad left with equal small weights
            pad = np.full(n - w.size, float(w[0]) if w.size else 1.0)
            w = np.concatenate([pad, w])
        w = w[-n:]
    s = float(np.sum(w))
    if s <= 0 or not np.isfinite(s):
        return np.ones(n, dtype=float) / n
    return w / s


def forecast_simple_average(
    history: np.ndarray,
    horizon: int,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Mean of all history, flat future."""
    hist = np.asarray(history, dtype=float).ravel()
    hist = hist[np.isfinite(hist)]
    if hist.size == 0:
        return np.zeros(horizon), np.array([]), 0.0
    mean = float(np.mean(hist))
    fitted = np.full(hist.size, mean)
    fc = np.full(horizon, mean)
    return fc, fitted, _std_from_fit(hist, fitted)


def forecast_simple_moving_average(
    history: np.ndarray,
    horizon: int,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Simple Moving Average.
    forecastMode:
      - extend  → last MA value repeated for all future periods
      - rolling → recursive (each forecast enters the next window)
    """
    hist = np.asarray(history, dtype=float).ravel()
    hist = hist[np.isfinite(hist)]
    if hist.size == 0:
        return np.zeros(horizon), np.array([]), 0.0

    window = max(1, int(params.get("window") or 3))
    window = min(window, hist.size)
    mode = str(params.get("forecastMode") or params.get("mode") or "rolling").lower()

    fitted = np.full(hist.size, np.nan)
    for i in range(hist.size):
        start = max(0, i - window + 1)
        fitted[i] = float(np.mean(hist[start : i + 1]))

    last_ma = float(np.mean(hist[-window:]))

    if mode in ("extend", "extend_current", "flat"):
        fc = np.full(horizon, last_ma)
    else:
        # rolling / recursive
        series = list(hist.astype(float))
        forecasts = []
        for _ in range(horizon):
            w = min(window, len(series))
            nxt = float(sum(series[-w:]) / w)
            forecasts.append(nxt)
            series.append(nxt)
        fc = np.asarray(forecasts, dtype=float)

    return fc, fitted, _std_from_fit(hist, fitted)


def forecast_weighted_average(
    history: np.ndarray,
    horizon: int,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Weighted average over full history (or last N), flat future."""
    hist = np.asarray(history, dtype=float).ravel()
    hist = hist[np.isfinite(hist)]
    if hist.size == 0:
        return np.zeros(horizon), np.array([]), 0.0

    lookback = int(params.get("lookback") or params.get("window") or hist.size)
    lookback = max(1, min(lookback, hist.size))
    segment = hist[-lookback:]
    weights = params.get("weights")
    if isinstance(weights, str):
        try:
            weights = [float(x.strip()) for x in weights.split(",") if x.strip()]
        except Exception:
            weights = []
    if not isinstance(weights, (list, tuple)):
        weights = []

    w = _normalize_weights(list(weights), segment.size)
    level = float(np.dot(segment, w))
    fitted = np.full(hist.size, level)
    # better in-sample: expanding weighted mean on trailing lookback
    for i in range(hist.size):
        start = max(0, i - lookback + 1)
        seg = hist[start : i + 1]
        ww = _normalize_weights(list(weights), seg.size)
        fitted[i] = float(np.dot(seg, ww))
    fc = np.full(horizon, level)
    return fc, fitted, _std_from_fit(hist, fitted)


def forecast_weighted_moving_average(
    history: np.ndarray,
    horizon: int,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Weighted Moving Average over a window.
    forecastMode: extend | rolling
    """
    hist = np.asarray(history, dtype=float).ravel()
    hist = hist[np.isfinite(hist)]
    if hist.size == 0:
        return np.zeros(horizon), np.array([]), 0.0

    window = max(1, int(params.get("window") or 3))
    window = min(window, hist.size)
    mode = str(params.get("forecastMode") or params.get("mode") or "rolling").lower()
    weights = params.get("weights")
    if isinstance(weights, str):
        try:
            weights = [float(x.strip()) for x in weights.split(",") if x.strip()]
        except Exception:
            weights = []
    if not isinstance(weights, (list, tuple)):
        weights = []

    def wma(values: List[float]) -> float:
        n = len(values)
        w = _normalize_weights(list(weights), n)
        return float(np.dot(np.asarray(values, dtype=float), w))

    fitted = np.full(hist.size, np.nan)
    for i in range(hist.size):
        start = max(0, i - window + 1)
        fitted[i] = wma(list(hist[start : i + 1]))

    last = wma(list(hist[-window:]))

    if mode in ("extend", "extend_current", "flat"):
        fc = np.full(horizon, last)
    else:
        series = list(hist.astype(float))
        forecasts = []
        for _ in range(horizon):
            w = min(window, len(series))
            nxt = wma(series[-w:])
            forecasts.append(nxt)
            series.append(nxt)
        fc = np.asarray(forecasts, dtype=float)

    return fc, fitted, _std_from_fit(hist, fitted)


# Back-compat alias used by older profiles
def forecast_moving_average(
    history: np.ndarray,
    horizon: int,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float]:
    # Default legacy moving_average → simple moving average rolling
    p = dict(params or {})
    if "forecastMode" not in p and "mode" not in p:
        p["forecastMode"] = "rolling"
    return forecast_simple_moving_average(history, horizon, p)
