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

    # best_fit / weighted_average → lowest in-sample MSE using fitted proxy (last hist mean of fc)
    # Candidates are future-only; score by stability vs last history window
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
) -> np.ndarray:
    """
    Adjust forecasts by historical forecast bias (actual - fitted residuals flipped).
    residuals should be (fitted - actual) so positive = over-forecast.
    """
    fc = np.asarray(forecasts, dtype=float).copy()
    res = np.asarray(residuals, dtype=float)
    res = res[np.isfinite(res)]
    if res.size == 0:
        return fc
    lookback = max(1, int(lookback or 6))
    window = res[-lookback:]
    method = (method or "mean").lower().replace("_bias", "")

    if method == "median":
        bias = float(np.median(window))
        return fc - bias
    if method == "proportional":
        scale = float(np.mean(np.abs(window))) + 1e-9
        bias_ratio = float(np.mean(window)) / scale
        return fc * (1.0 - bias_ratio)
    # mean
    bias = float(np.mean(window))
    return fc - bias


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
        return {"method": str(method), "lookbackPeriods": int(lookback)}
    return None
