"""Best-fit model selection on a holdout (out-of-sample) window.

For each candidate model: fit on history minus the last ``h`` periods, forecast ``h``
periods, score against the held-back actuals with the profile's error measure, and
pick the lowest error. Ties go to the simpler model. With too little history the
first configured model is used and the reason is reported.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Simpler models first — used to break ties.
COMPLEXITY_ORDER = [
    "simple_average",
    "simple_moving_average",
    "moving_average",
    "weighted_average",
    "weighted_moving_average",
    "croston",
    "exponential_smoothing",
    "regression",
    "sarima",
    "gradient_boosting",
]

SUPPORTED_MEASURES = ("WMAPE", "MAE", "RMSE", "MSE", "MAPE", "SMAPE", "BIAS")


def complexity_rank(atype: str) -> int:
    t = str(atype or "").lower()
    return COMPLEXITY_ORDER.index(t) if t in COMPLEXITY_ORDER else len(COMPLEXITY_ORDER)


def selection_measure(measures: Any) -> str:
    """First supported measure from the profile, defaulting to WMAPE (zero-safe)."""
    items = measures if isinstance(measures, (list, tuple)) else [measures]
    for m in items:
        key = str(m or "").upper().strip()
        if key in SUPPORTED_MEASURES:
            return key
    return "WMAPE"


def holdout_error(actual: np.ndarray, forecast: np.ndarray, measure: str = "WMAPE") -> float:
    a = np.asarray(actual, dtype=float).ravel()
    f = np.asarray(forecast, dtype=float).ravel()
    n = min(a.size, f.size)
    a, f = a[:n], f[:n]
    mask = np.isfinite(a) & np.isfinite(f)
    a, f = a[mask], f[mask]
    if a.size == 0:
        return float("inf")
    err = f - a
    m = str(measure or "WMAPE").upper()
    if m == "MAE":
        return float(np.mean(np.abs(err)))
    if m == "RMSE":
        return float(np.sqrt(np.mean(err ** 2)))
    if m == "MSE":
        return float(np.mean(err ** 2))
    if m == "BIAS":
        return float(abs(np.mean(err)))
    if m == "SMAPE":
        return float(np.mean(2.0 * np.abs(err) / (np.abs(a) + np.abs(f) + 1e-9)))
    if m == "MAPE":
        nz = np.abs(a) > 1e-9
        if nz.any():
            return float(np.mean(np.abs(err[nz] / a[nz])))
        m = "WMAPE"
    denom = float(np.sum(np.abs(a)))
    if denom < 1e-9:
        return float(np.mean(np.abs(err)))
    return float(np.sum(np.abs(err)) / denom)


def _tie_round(score: float) -> float:
    if not np.isfinite(score):
        return score
    return float(f"{score:.9g}")


def _recent_level_error(values: np.ndarray, forecast: Optional[np.ndarray], h: int, measure: str) -> float:
    """Error of a model's first ``h`` forecast periods against the last ``h`` actuals."""
    if forecast is None:
        return float("inf")
    v = np.asarray(values, dtype=float).ravel()
    f = np.asarray(forecast, dtype=float).ravel()
    if f.size == 0 or v.size == 0:
        return float("inf")
    k = min(h, v.size, f.size)
    score = holdout_error(v[-k:], f[:k], measure)
    return score if np.isfinite(score) else float("inf")


def holdout_periods(n: int, configured: Optional[int]) -> int:
    """Holdout length: the configured error horizon (default 6), capped at a third of the
    history so a long reporting horizon never prevents model selection."""
    base = int(configured) if configured and int(configured) > 0 else 6
    return max(1, min(base, n // 3))


def select_by_holdout(
    candidates: Dict[str, np.ndarray],
    key_to_alg: Dict[str, Dict[str, Any]],
    key_to_type: Dict[str, str],
    values: np.ndarray,
    run_fn: Callable[[Dict[str, Any], np.ndarray, int], np.ndarray],
    measure: str,
    configured_h: Optional[int] = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Return (selected candidate key, selection info).

    Ties on the holdout score (equal up to float noise) are broken by how close the
    model's forecast is to the last ``h`` actuals, then by model simplicity.
    Example: 18 months of 100 then 6 zeros — every model trained on the 100s misses the
    zero holdout by ~100, so a model that forecasts the recent zeros (moving average)
    wins instead of the 24-month average (75).
    """
    keys: List[str] = list(candidates.keys())
    info: Dict[str, Any] = {"strategy": "best_fit", "measure": measure, "scores": {}}
    if not keys:
        info["reason"] = "no_candidates"
        return None, info
    if len(keys) == 1:
        info.update({"selected": keys[0], "reason": "single_model"})
        return keys[0], info

    n = int(np.asarray(values).size)
    h = holdout_periods(n, configured_h)
    info["holdoutPeriods"] = h
    if n < 2 * h + 3:
        info.update({"selected": keys[0], "reason": "insufficient_history_for_holdout"})
        return keys[0], info

    train, test = np.asarray(values, dtype=float)[:-h], np.asarray(values, dtype=float)[-h:]
    best_key, best = None, (float("inf"), float("inf"), float("inf"))
    for key in keys:
        try:
            fc_h = run_fn(key_to_alg[key], train, h)
            score = holdout_error(test, fc_h, measure)
        except Exception as exc:  # a model that cannot fit the shorter window loses
            info["scores"][key] = None
            info.setdefault("holdoutFailures", {})[key] = str(exc)
            continue
        info["scores"][key] = score if np.isfinite(score) else None
        # Scores equal up to float noise are a tie (e.g. a holdout of all zeros where
        # every model misses by the same amount) — the simpler model wins, not 1e-14.
        rank = (
            _tie_round(score),
            _tie_round(_recent_level_error(values, candidates.get(key), h, measure)),
            complexity_rank(key_to_type.get(key, "")),
        )
        if np.isfinite(score) and rank < best:
            best, best_key = rank, key
    if best_key is None:
        info.update({"selected": keys[0], "reason": "holdout_failed"})
        return keys[0], info
    info.update({"selected": best_key, "reason": "lowest_holdout_error"})
    return best_key, info
