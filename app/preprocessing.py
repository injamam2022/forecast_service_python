"""Preprocessing: outlier correction, missing values, substitution, historical adjustment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def clean_history(series: List[Any]) -> Tuple[List[str], np.ndarray]:
    """Parse series list into sorted buckets + numeric values."""
    rows = []
    for item in series or []:
        if not isinstance(item, dict):
            continue
        tb = str(item.get("timeBucket") or "").strip()
        if not tb:
            continue
        try:
            val = float(item.get("value") if item.get("value") is not None else 0)
        except (TypeError, ValueError):
            val = 0.0
        if not np.isfinite(val):
            val = np.nan
        rows.append((tb, val))
    rows.sort(key=lambda r: r[0])
    if not rows:
        return [], np.array([], dtype=float)
    buckets = [r[0] for r in rows]
    values = np.array([r[1] for r in rows], dtype=float)
    return buckets, values


def trim_trailing_empty(
    buckets: List[str],
    values: np.ndarray,
    *,
    treat_zero_as_empty: bool = True,
) -> Tuple[List[str], np.ndarray]:
    """
    Drop trailing empty months so blank future cells (often stored as 0) do not
    flatten trend models. Keeps genuine interior zeros.
    """
    vals = np.asarray(values, dtype=float).ravel()
    if vals.size == 0 or not buckets:
        return buckets, vals

    end = vals.size
    while end > 0:
        v = vals[end - 1]
        empty = (not np.isfinite(v)) or (treat_zero_as_empty and abs(float(v)) < 1e-12)
        if not empty:
            break
        end -= 1

    # Keep at least one point; if everything was empty, leave as-is
    if end == 0:
        return buckets, vals
    if end == vals.size:
        return buckets, vals
    return buckets[:end], vals[:end]


def _outlier_correction(values: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    out = values.copy().astype(float)
    method = str(params.get("method") or "iqr").lower()
    finite = out[np.isfinite(out)]
    if finite.size < 3:
        return out

    if method == "z_score":
        threshold = float(params.get("threshold") or params.get("multiplier") or 3)
        mu, sigma = float(np.mean(finite)), float(np.std(finite))
        if sigma < 1e-12:
            return out
        z = (out - mu) / sigma
        out[np.abs(z) > threshold] = mu
    elif method == "percentile":
        lo = float(params.get("lower") or 5)
        hi = float(params.get("upper") or 95)
        p_lo, p_hi = np.nanpercentile(finite, [lo, hi])
        out = np.clip(out, p_lo, p_hi)
    else:
        # IQR (SAP-style default)
        multiplier = float(params.get("multiplier") or params.get("threshold") or 1.5)
        q1, q3 = np.nanpercentile(finite, [25, 75])
        iqr = q3 - q1
        low, high = q1 - multiplier * iqr, q3 + multiplier * iqr
        out = np.where(out < low, q1, out)
        out = np.where(out > high, q3, out)
    return out


def _missing_value_handling(values: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    out = values.copy().astype(float)
    method = str(params.get("method") or "forward_fill").lower()
    if method in ("interpolate", "interpolation"):
        # linear interpolation over NaNs
        n = out.size
        idx = np.arange(n)
        mask = np.isfinite(out)
        if mask.sum() == 0:
            return np.zeros(n, dtype=float)
        if mask.sum() == 1:
            return np.full(n, float(out[mask][0]))
        out[~mask] = np.interp(idx[~mask], idx[mask], out[mask])
        return out

    if method == "zero":
        out[~np.isfinite(out)] = 0.0
        return out

    if method == "mean":
        finite = out[np.isfinite(out)]
        fill = float(np.mean(finite)) if finite.size else 0.0
        out[~np.isfinite(out)] = fill
        return out

    if method == "backward_fill":
        # reverse forward-fill
        rev = out[::-1].copy()
        last = np.nan
        for i in range(rev.size):
            if np.isfinite(rev[i]):
                last = rev[i]
            elif np.isfinite(last):
                rev[i] = last
        return rev[::-1]

    # forward_fill (default)
    last = np.nan
    for i in range(out.size):
        if np.isfinite(out[i]):
            last = out[i]
        elif np.isfinite(last):
            out[i] = last
    # leading NaNs → mean or 0
    if not np.isfinite(out).all():
        finite = out[np.isfinite(out)]
        fill = float(np.mean(finite)) if finite.size else 0.0
        out[~np.isfinite(out)] = fill
    return out


def _substitution(values: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    out = values.copy().astype(float)
    rule = str(params.get("rule") or "replace").lower()
    val = float(params.get("value") or 0)
    if rule == "multiply":
        return out * val
    if rule == "add":
        return out + val
    # replace all with constant (or only zeros if replaceZeros flag)
    if params.get("replaceZeros"):
        out[out == 0] = val
        return out
    return np.full_like(out, val)


def _historical_adjustment(values: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    out = values.copy().astype(float)
    adj_type = str(params.get("type") or "multiplier").lower()
    val = float(params.get("value") if params.get("value") is not None else params.get("factor") or 1)
    if adj_type == "additive":
        return out + val
    if adj_type == "percentage":
        return out * (1.0 + val / 100.0)
    # multiplier
    return out * val


def apply_step(values: np.ndarray, step: Dict[str, Any]) -> np.ndarray:
    if not step or step.get("enabled") is False:
        return values
    stype = str(step.get("type") or "").lower().strip()
    params = step.get("parameters") or {}
    if not isinstance(params, dict):
        params = {}

    if stype in ("outlier_correction", "outlier"):
        return _outlier_correction(values, params)
    if stype in ("missing_value_handling", "missing_value", "missing"):
        return _missing_value_handling(values, params)
    if stype == "substitution":
        return _substitution(values, params)
    if stype in ("historical_adjustment", "historical_adj", "manual_adjustment", "manual_adj"):
        return _historical_adjustment(values, params)
    return values


def step_key_figures(step: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Return (inputKeyFigureId, outputKeyFigureId) for a step, tolerating snake_case."""
    if not isinstance(step, dict):
        return None, None
    in_kf = step.get("inputKeyFigureId") or step.get("input_key_figure_id")
    out_kf = step.get("outputKeyFigureId") or step.get("output_key_figure_id")
    return (str(in_kf) if in_kf else None, str(out_kf) if out_kf else None)


def _resolve_step_input(
    step: Dict[str, Any],
    current: np.ndarray,
    kf_outputs: Dict[str, np.ndarray],
    series_by_kf: Optional[Dict[str, np.ndarray]],
) -> np.ndarray:
    """
    Pick the series a step reads. Priority: an earlier step's Output KF, then a
    key figure loaded from the planner grid, then the running chain value.
    """
    in_kf, _ = step_key_figures(step)
    if not in_kf:
        return current
    if in_kf in kf_outputs:
        return np.asarray(kf_outputs[in_kf], dtype=float).copy()
    if series_by_kf and in_kf in series_by_kf:
        candidate = np.asarray(series_by_kf[in_kf], dtype=float)
        if candidate.size == current.size:
            return candidate.copy()
        # Align to the running length so downstream steps stay comparable
        if candidate.size > current.size:
            return candidate[-current.size :].copy()
        pad = np.full(current.size - candidate.size, np.nan)
        return np.concatenate([pad, candidate])
    return current


def apply_preprocessing(
    values: np.ndarray,
    steps: Optional[List[Dict[str, Any]]],
    return_intermediates: bool = False,
    series_by_kf: Optional[Dict[str, np.ndarray]] = None,
    kf_outputs: Optional[Dict[str, np.ndarray]] = None,
) -> Any:
    """
    Apply preprocessing steps in order.
    If return_intermediates=True, returns (final_values, {stepId: values}).
    Each step reads its own Input KF when one is configured, so profiles can chain
    Input KF -> Output KF across steps. `kf_outputs`, when supplied, is filled with
    {outputKeyFigureId: values} so callers can resolve the forecasting Input KF.
    """
    current = np.asarray(values, dtype=float).copy()
    intermediates: Dict[str, np.ndarray] = {}
    produced: Dict[str, np.ndarray] = kf_outputs if kf_outputs is not None else {}
    if not steps:
        return (current, intermediates) if return_intermediates else current

    ordered = sorted(
        [s for s in steps if isinstance(s, dict)],
        key=lambda s: int(s.get("order") or 0),
    )
    for step in ordered:
        if step.get("enabled") is False:
            continue
        step_input = _resolve_step_input(step, current, produced, series_by_kf)
        current = apply_step(step_input, step)
        step_id = str(step.get("stepId") or step.get("id") or step.get("type") or "")
        if step_id:
            intermediates[step_id] = current.copy()
        _, out_kf = step_key_figures(step)
        if out_kf:
            produced[out_kf] = current.copy()

    return (current, intermediates) if return_intermediates else current
