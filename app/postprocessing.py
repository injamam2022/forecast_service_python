"""Postprocessing: final selection, bias correction helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def extract_selection_strategy(postprocessing: Any) -> str:
    if isinstance(postprocessing, dict):
        return str(
            postprocessing.get("finalSelectionStrategy")
            or postprocessing.get("final_selection_strategy")
            or "first"
        ).lower()
    return "first"


def select_forecast(
    candidates: Dict[str, np.ndarray],
    history: np.ndarray,
    strategy: str,
) -> Tuple[str, np.ndarray]:
    """Pick final forecast series from candidate models."""
    if not candidates:
        mean = float(np.nanmean(history)) if history.size else 0.0
        return "mean", np.array([mean], dtype=float)

    names = list(candidates.keys())
    strategy = (strategy or "first").lower()

    if strategy in ("first", "manual") or len(names) == 1:
        return names[0], candidates[names[0]]

    if strategy in ("ensemble_average", "average"):
        stacked = np.vstack([candidates[n] for n in names])
        return "ensemble_average", np.nanmean(stacked, axis=0)

    if strategy in ("ensemble_median", "median"):
        stacked = np.vstack([candidates[n] for n in names])
        return "ensemble_median", np.nanmedian(stacked, axis=0)

    # "stability" heuristic (also the pure-function meaning of best_fit): prefer the model
    # whose first forecasts sit closest to the recent history level. The engine itself
    # implements best_fit as a holdout (out-of-sample) error comparison — see selection.py.
    best_name = names[0]
    best_err = float("inf")
    hist_tail = history[-min(len(history), 6) :] if history.size else np.array([])
    hist_mean = float(np.nanmean(hist_tail)) if hist_tail.size else 0.0
    for name in names:
        fc = candidates[name]
        # Prefer model whose first forecasts are closest to recent history level + trend
        err = float(np.nanmean((fc[: min(3, fc.size)] - hist_mean) ** 2))
        if np.isfinite(err) and err < best_err:
            best_err = err
            best_name = name
    return best_name, candidates[best_name]


def apply_bias_correction(
    forecasts: np.ndarray,
    residuals: np.ndarray,
    method: str = "mean",
    lookback: int = 6,
    actuals: Optional[np.ndarray] = None,
    max_ratio: float = 0.5,
    min_points: int = 3,
    non_negative: bool = True,
) -> np.ndarray:
    """
    Adjust forecasts by historical forecast bias.

    ``residuals`` are (forecast - actual): positive = over-forecast.

    - mean / median: subtract the average (or median) error over the lookback.
    - proportional: relative bias = Σ(forecast - actual) / Σ(actual) over the lookback,
      clamped to ±max_ratio, then forecast × (1 - ratio). ``actuals`` (same length and
      order as ``residuals``) is required; without it, or when Σ(actual) ≈ 0, the
      forecast is returned unchanged.

    Fewer than ``min_points`` usable residuals → no correction. With non_negative the
    result is floored at 0.
    """
    fc = np.asarray(forecasts, dtype=float).copy()
    res_all = np.asarray(residuals, dtype=float).ravel()
    act_all = None
    if actuals is not None:
        act_all = np.asarray(actuals, dtype=float).ravel()
        if act_all.size != res_all.size:
            act_all = None
    mask = np.isfinite(res_all)
    if act_all is not None:
        mask = mask & np.isfinite(act_all)
    res = res_all[mask]
    act = act_all[mask] if act_all is not None else None
    if res.size < max(1, int(min_points or 1)):
        return fc
    lookback = max(1, int(lookback or 6))
    window = res[-lookback:]
    method = (method or "mean").lower().replace("_bias", "")

    if method == "proportional":
        if act is None:
            return fc
        act_window = act[-lookback:]
        denom = float(np.sum(act_window))
        if abs(denom) < 1e-9:
            return fc
        ratio = float(np.sum(window)) / denom
        limit = abs(float(max_ratio)) if max_ratio is not None else 0.5
        ratio = max(-limit, min(limit, ratio))
        out = fc * (1.0 - ratio)
    elif method == "median":
        out = fc - float(np.median(window))
    else:
        out = fc - float(np.mean(window))
    if non_negative:
        out = np.maximum(out, 0.0)
    return out


def resolve_bias_config(postprocessing: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(postprocessing, dict):
        return None
    if postprocessing.get("applyBiasCorrection") or (
        isinstance(postprocessing.get("biasCorrection"), dict)
        and postprocessing["biasCorrection"].get("enabled")
    ):
        bc = postprocessing.get("biasCorrection") or {}
        method = (
            bc.get("method")
            or postprocessing.get("biasCorrectionMethod")
            or "mean"
        )
        lookback = (
            bc.get("lookbackPeriods")
            or postprocessing.get("biasLookbackPeriods")
            or 12
        )
        max_ratio = bc.get("maxRatio", postprocessing.get("biasMaxRatio", 0.5))
        min_points = bc.get("minPoints", postprocessing.get("biasMinPoints", 3))
        try:
            max_ratio = float(max_ratio)
        except (TypeError, ValueError):
            max_ratio = 0.5
        try:
            min_points = int(min_points)
        except (TypeError, ValueError):
            min_points = 3
        return {
            "method": str(method),
            "lookbackPeriods": int(lookback),
            "maxRatio": max_ratio,
            "minPoints": min_points,
        }
    return None
