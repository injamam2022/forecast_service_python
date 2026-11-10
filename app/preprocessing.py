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


def _protect_planner_edits(params: Dict[str, Any]) -> bool:
    raw = params.get("protectPlannerEdits", params.get("protect_planner_edits", True))
    if isinstance(raw, str):
        return raw.strip().lower() not in ("0", "false", "no", "off")
    return bool(raw)


def _outlier_correction(
    values: np.ndarray,
    params: Dict[str, Any],
    protect_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Correct extreme values. For short / intermittent demand series (few observations),
    IQR fences are unstable and can crush real spikes (e.g. 3000 → q3≈1000). Skip
    correction unless there are enough points to estimate a fence reliably.
    """
    out = values.copy().astype(float)
    method = str(params.get("method") or "iqr").lower()
    finite = out[np.isfinite(out)]
    min_points = int(params.get("minPoints") or params.get("min_points") or 8)
    if finite.size < max(3, min_points):
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
        # IQR (SAP-style default). Prefer non-zero observations when series is intermittent
        # so long zero stretches do not collapse the fence onto a few demand spikes.
        nonzero = finite[np.abs(finite) > 1e-12]
        sample = nonzero if nonzero.size >= min_points else finite
        if sample.size < max(3, min_points):
            return out
        multiplier = float(params.get("multiplier") or params.get("threshold") or 1.5)
        q1, q3 = np.nanpercentile(sample, [25, 75])
        iqr = q3 - q1
        if iqr < 1e-12:
            return out
        low, high = q1 - multiplier * iqr, q3 + multiplier * iqr
        # Replace outliers with the outlier limit itself (default "fence"), so a real
        # promotion spike is tempered, not flattened to an ordinary month.
        # replaceWith = "q3" keeps the older behaviour (Q1/Q3), "median" uses the median.
        replace_with = str(params.get("replaceWith") or params.get("replace_with") or "fence").lower()
        if replace_with == "q3":
            low_val, high_val = q1, q3
        elif replace_with == "median":
            med = float(np.nanmedian(sample))
            low_val, high_val = med, med
        else:
            low_val, high_val = low, high
        # Keep explicit zeros (intermittent demand / cleared Adjusted cells). A 0 is
        # not an IQR low-outlier — replacing it with q1 ignored grid edits of 0.
        finite_low = np.isfinite(out) & (np.abs(out) > 1e-12) & (out < low)
        out = np.where(finite_low, low_val, out)
        out = np.where(out > high, high_val, out)
    # Planner-entered cells (a typed 0 or 1000) are deliberate — keep them as entered
    # unless the step sets protectPlannerEdits = false.
    if protect_mask is not None and _protect_planner_edits(params):
        guard = np.asarray(protect_mask, dtype=bool)
        if guard.shape == out.shape:
            out = np.where(guard, np.asarray(values, dtype=float), out)
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


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _substitution(values: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Substitution needs an explicit rule and value.

    A missing value used to default to 0 and the default rule replaced the whole
    history — one unconfigured step silently produced a zero forecast.
    """
    out = values.copy().astype(float)
    rule_raw = params.get("rule")
    if rule_raw is None or str(rule_raw).strip() == "":
        raise ValueError("Substitution step needs a rule (replace, add or multiply).")
    rule = str(rule_raw).lower().strip()
    raw_value = params.get("value")
    try:
        val = float(raw_value)
    except (TypeError, ValueError):
        raise ValueError("Substitution step needs a numeric value.")
    if not np.isfinite(val):
        raise ValueError("Substitution step needs a numeric value.")
    if rule == "multiply":
        return out * val
    if rule == "add":
        return out + val
    if rule != "replace":
        raise ValueError(f"Unknown substitution rule '{rule_raw}'. Use replace, add or multiply.")
    if _truthy(params.get("replaceZeros")):
        out[out == 0] = val
        return out
    if not _truthy(params.get("replaceAll")):
        raise ValueError(
            "Substitution 'replace' overwrites every history value. Set replaceZeros to replace only "
            "zeros, or replaceAll to confirm replacing the whole history."
        )
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


def apply_step(
    values: np.ndarray,
    step: Dict[str, Any],
    protect_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if not step or step.get("enabled") is False:
        return values
    stype = str(step.get("type") or "").lower().strip()
    params = step.get("parameters") or {}
    if not isinstance(params, dict):
        params = {}

    if stype in ("outlier_correction", "outlier"):
        return _outlier_correction(values, params, protect_mask=protect_mask)
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
    series_overlay: Optional[Dict[str, np.ndarray]] = None,
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
        in_kf, _ = step_key_figures(step)
        protect = None
        if series_overlay and in_kf and in_kf in series_overlay and in_kf not in produced:
            protect = np.asarray(series_overlay[in_kf], dtype=bool)
            if protect.size != step_input.size:
                protect = None
        current = apply_step(step_input, step, protect_mask=protect)
        step_id = str(step.get("stepId") or step.get("id") or step.get("type") or "")
        if step_id:
            intermediates[step_id] = current.copy()
        _, out_kf = step_key_figures(step)
        if out_kf:
            produced[out_kf] = current.copy()

    return (current, intermediates) if return_intermediates else current
