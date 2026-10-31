"""Core forecast orchestration — SAP IBP-grade pipeline."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .algorithms import (
    forecast_croston,
    forecast_exponential_smoothing,
    forecast_gradient_boosting,
    forecast_moving_average,
    forecast_regression,
    forecast_sarima,
    forecast_simple_average,
    forecast_simple_moving_average,
    forecast_weighted_average,
    forecast_weighted_moving_average,
)
from .postprocessing import (
    apply_bias_correction,
    extract_selection_strategy,
    resolve_bias_config,
    select_forecast,
)
from .preprocessing import (
    apply_preprocessing,
    clean_history,
    step_key_figures,
    trim_trailing_empty,
)
from .time_buckets import generate_future_buckets, seasonal_period

logger = logging.getLogger("forecast_service")


def _bucket_variants(bucket: str) -> List[str]:
    b = str(bucket)
    variants = [b]
    if len(b) >= 7:
        variants.append(b[:7])
    if len(b) >= 10:
        variants.append(b[:10])
    return variants


def _align_to_buckets(series: List[Any], buckets: List[str]) -> np.ndarray:
    """Map a raw {timeBucket, value} list onto the primary bucket axis (NaN when absent)."""
    lookup: Dict[str, float] = {}
    for item in series or []:
        if not isinstance(item, dict):
            continue
        tb = str(item.get("timeBucket") or "").strip()
        if not tb:
            continue
        try:
            val = float(item.get("value") if item.get("value") is not None else 0)
        except (TypeError, ValueError):
            val = np.nan
        for key in _bucket_variants(tb):
            lookup.setdefault(key, val)

    out = np.full(len(buckets), np.nan, dtype=float)
    for i, bucket in enumerate(buckets):
        for key in _bucket_variants(bucket):
            if key in lookup:
                out[i] = lookup[key]
                break
    return out


def _series_entry_identity(entry: Dict[str, Any]) -> Tuple[str, str, str]:
    attrs = entry.get("attributes") if isinstance(entry.get("attributes"), dict) else {}
    pid = str(
        entry.get("productId")
        or attrs.get("product_id")
        or attrs.get("PRODUCT_ID")
        or ""
    ).strip()
    lid = str(
        entry.get("locationId")
        or attrs.get("location_id")
        or attrs.get("LOCATION_ID")
        or ""
    ).strip()
    cid = str(
        entry.get("customerId")
        or attrs.get("customer_id")
        or attrs.get("CUSTOMER_ID")
        or ""
    ).strip()
    return pid, lid, cid


def _series_by_key_figure(
    payload: Dict[str, Any],
    po_id: str,
    buckets: List[str],
    attrs: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """Per-key-figure history for this planning object, aligned to the primary buckets.

    Prefer exact planningObjectId / seriesId match. Fall back to Product×Location×Customer
    identity so Adjusted Sales History UI edits on a sibling PO still feed Missing/Manual.
    """
    raw = payload.get("seriesByKeyFigure") or {}
    if not isinstance(raw, dict):
        return {}
    attrs = attrs if isinstance(attrs, dict) else {}
    want_pid = str(attrs.get("product_id") or attrs.get("PRODUCT_ID") or "").strip()
    want_lid = str(attrs.get("location_id") or attrs.get("LOCATION_ID") or "").strip()
    want_cid = str(attrs.get("customer_id") or attrs.get("CUSTOMER_ID") or "").strip()
    resolved: Dict[str, np.ndarray] = {}
    for kf_id, entries in raw.items():
        if not isinstance(entries, list):
            continue
        matched = None
        identity_hit = None
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_po = str(entry.get("planningObjectId") or "").strip()
            entry_base = str(entry.get("basePlanningObjectId") or "").strip()
            if entry_po == po_id or (entry_base and entry_base == po_id):
                matched = entry
                break
            if want_pid and want_lid and want_cid:
                ep, el, ec = _series_entry_identity(entry)
                if ep == want_pid and el == want_lid and ec == want_cid:
                    identity_hit = entry
        chosen = matched or identity_hit
        if chosen is not None:
            resolved[str(kf_id)] = _align_to_buckets(chosen.get("series") or [], buckets)
    return resolved


def _fill_gaps(values: np.ndarray) -> np.ndarray:
    """Forward/backward fill remaining NaNs so algorithms always get a dense series."""
    arr = np.asarray(values, dtype=float).ravel().copy()
    if arr.size == 0 or np.all(np.isfinite(arr)):
        return arr
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    last = np.nan
    for i in range(arr.size):
        if np.isfinite(arr[i]):
            last = arr[i]
        elif np.isfinite(last):
            arr[i] = last
    first = float(finite[0])
    arr[~np.isfinite(arr)] = first
    return arr


def _stored_forecast_points(payload: Dict[str, Any], po_id: str) -> List[Dict[str, Any]]:
    """Future/stored forecast series written by a prior forecasting profile."""
    raw = payload.get("storedForecasts")
    if not isinstance(raw, dict) or not raw:
        return []
    post = payload.get("postprocessingSteps") or {}
    preferred: List[str] = []
    if isinstance(post, dict):
        for key in (
            "finalInputKeyFigureId",
            "biasInputKeyFigureId",
            "errorInputKeyFigureId",
            "snapshotInputKeyFigureId",
        ):
            val = post.get(key)
            if val:
                preferred.append(str(val))
    keys = preferred + [str(k) for k in raw.keys() if str(k) not in preferred]
    for kf_id in keys:
        entries = raw.get(kf_id) or raw.get(str(kf_id)) or []
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("planningObjectId") or "").strip() != po_id:
                continue
            points: List[Dict[str, Any]] = []
            for item in entry.get("series") or []:
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
                    val = 0.0
                points.append({"timeBucket": tb, "value": val})
            points.sort(key=lambda p: str(p["timeBucket"]))
            return points
    return []


def _stage_flags(payload: Dict[str, Any]) -> Tuple[bool, bool, bool]:
    """Return (run_preprocessing, run_forecasting, run_postprocessing)."""
    raw = str(payload.get("profileType") or "").upper().strip()
    if raw == "PREPROCESSING":
        return True, False, False
    if raw == "FORECASTING":
        return False, True, False
    if raw == "POSTPROCESSING":
        return False, False, True
    return True, True, True


def _as_algo_list(payload: Dict[str, Any], run_forecasting: bool = True) -> List[Dict[str, Any]]:
    if not run_forecasting:
        return []
    steps = payload.get("forecastingSteps") or payload.get("algorithms") or []
    if isinstance(steps, dict):
        steps = [steps]
    out = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        if s.get("enabled") is False:
            continue
        out.append(s)
    if not out:
        profile_type = str(payload.get("profileType") or "").upper().strip()
        if profile_type in ("PREPROCESSING", "POSTPROCESSING"):
            return []
        out = [{"id": "alg-default-ma", "type": "moving_average", "parameters": {"window": 3}, "enabled": True}]
    return out


def _error_metrics(
    actual: np.ndarray,
    fitted: np.ndarray,
    measures: List[str],
    horizon_periods: Optional[int] = None,
) -> Dict[str, float]:
    """Error metrics on last `horizon_periods` periods only (SAP-style)."""
    actual = np.asarray(actual, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    n = min(actual.size, fitted.size)
    if n == 0:
        return {}
    a = actual[-n:]
    f = fitted[-n:]
    if horizon_periods and horizon_periods > 0:
        h = min(int(horizon_periods), a.size)
        a, f = a[-h:], f[-h:]

    mask = np.isfinite(a) & np.isfinite(f)
    a, f = a[mask], f[mask]
    if a.size == 0:
        return {}

    residuals = a - f
    result: Dict[str, float] = {}
    for m in measures or ["MAPE"]:
        key = str(m).upper()
        if key == "MAPE":
            denom = np.where(np.abs(a) < 1e-9, np.nan, a)
            mape = float(np.nanmean(np.abs(residuals / denom)))
            if np.isfinite(mape):
                result["MAPE"] = mape
        elif key == "SMAPE":
            denom = np.abs(a) + np.abs(f) + 1e-9
            smape = float(np.mean(2.0 * np.abs(residuals) / denom))
            if np.isfinite(smape):
                result["SMAPE"] = smape
        elif key == "MSE":
            result["MSE"] = float(np.mean(residuals ** 2))
        elif key == "RMSE":
            result["RMSE"] = float(np.sqrt(np.mean(residuals ** 2)))
        elif key == "MAE":
            result["MAE"] = float(np.mean(np.abs(residuals)))
        elif key == "BIAS":
            # Positive = over-forecast (fitted > actual)
            result["BIAS"] = float(np.mean(f - a))
    return result


def _run_algorithm(
    alg: Dict[str, Any],
    history: np.ndarray,
    horizon: int,
    granularity: str,
) -> Tuple[str, np.ndarray, np.ndarray, float]:
    atype = str(alg.get("type") or "").lower().strip()
    params = alg.get("parameters") or {}
    if not isinstance(params, dict):
        params = {}
    sp = seasonal_period(granularity)

    if atype in ("simple_average", "mean", "average"):
        fc, fitted, std = forecast_simple_average(history, horizon, params)
        atype = "simple_average"
    elif atype in ("simple_moving_average", "sma"):
        fc, fitted, std = forecast_simple_moving_average(history, horizon, params)
        atype = "simple_moving_average"
    elif atype in ("weighted_average", "wa"):
        fc, fitted, std = forecast_weighted_average(history, horizon, params)
        atype = "weighted_average"
    elif atype in ("weighted_moving_average", "wma"):
        fc, fitted, std = forecast_weighted_moving_average(history, horizon, params)
        atype = "weighted_moving_average"
    elif atype == "moving_average":
        # Legacy alias → SMA rolling
        fc, fitted, std = forecast_moving_average(history, horizon, params)
    elif atype == "exponential_smoothing":
        fc, fitted, std = forecast_exponential_smoothing(history, horizon, params, seasonal_periods=sp)
    elif atype == "regression":
        fc, fitted, std = forecast_regression(history, horizon, params)
    elif atype in ("gradient_boosting", "gradientboosting", "gb"):
        fc, fitted, std = forecast_gradient_boosting(history, horizon, params)
    elif atype == "croston":
        fc, fitted, std = forecast_croston(history, horizon, params)
    elif atype in ("sarima", "arima"):
        fc, fitted, std = forecast_sarima(history, horizon, params, seasonal_periods=sp)
    else:
        fc, fitted, std = forecast_simple_moving_average(history, horizon, {"window": 3, "forecastMode": "rolling"})
        atype = f"fallback_sma_for_{atype or 'unknown'}"

    fc = np.asarray(fc, dtype=float).ravel()
    if fc.size < horizon:
        pad = float(fc[-1]) if fc.size else float(np.nanmean(history) if history.size else 0.0)
        fc = np.concatenate([fc, np.full(horizon - fc.size, pad)])
    elif fc.size > horizon:
        fc = fc[:horizon]
    return atype, fc, np.asarray(fitted, dtype=float), float(std or 0.0)


def _algo_key(alg: Dict[str, Any], atype: str, used: set) -> str:
    key = str(alg.get("algorithmId") or alg.get("id") or atype)
    base = key
    idx = 1
    while key in used:
        idx += 1
        key = f"{base}_{idx}"
    used.add(key)
    return key


def run_forecast(payload: Dict[str, Any]) -> Dict[str, Any]:
    time_series = payload.get("timeSeries") or []
    if not time_series:
        raise ValueError("timeSeries missing")

    run_prep, run_fc, run_post = _stage_flags(payload)
    horizon = int(payload.get("forecastPeriods") or 6)
    horizon = max(1, horizon)
    granularity = str(payload.get("timeGranularity") or "monthly")
    algorithms = _as_algo_list(payload, run_fc)
    strategy = extract_selection_strategy(payload.get("postprocessingSteps")) if run_post or run_fc else "best_fit"
    bias_cfg = resolve_bias_config(payload.get("postprocessingSteps")) if run_post else None
    measures = payload.get("errorMeasures") or ["MAPE"]
    if isinstance(measures, str):
        measures = [measures]

    error_horizon = payload.get("errorCalculationHorizonPeriods")
    if error_horizon is None:
        post = payload.get("postprocessingSteps") or {}
        if isinstance(post, dict):
            error_horizon = post.get("errorCalculationHorizonPeriods")
    try:
        error_horizon = int(error_horizon) if error_horizon is not None else None
    except (TypeError, ValueError):
        error_horizon = None

    prep_steps = payload.get("preprocessingSteps") or [] if run_prep else []
    if isinstance(prep_steps, dict):
        prep_steps = [prep_steps]

    # Key figure the primary timeSeries was loaded from (chain root when preprocessing
    # reads a different KF than the profile Input KF).
    primary_kf = str(
        payload.get("primaryKeyFigure") or payload.get("inputKeyFigure") or ""
    ).strip()

    all_forecasts: List[Dict[str, Any]] = []
    all_confidence: List[Dict[str, Any]] = []
    all_expost: List[Dict[str, Any]] = []
    all_fitted: List[Dict[str, Any]] = []
    error_accum: Dict[str, List[float]] = {}
    error_by_series: Dict[str, Dict[str, float]] = {}

    # Per-algorithm and preprocessing outputs across all POs
    algorithm_outputs: Dict[str, List[Dict[str, Any]]] = {}
    preprocessing_outputs: Dict[str, List[Dict[str, Any]]] = {}
    algorithm_meta: Dict[str, Dict[str, Any]] = {}

    for obj in time_series:
        if not isinstance(obj, dict):
            continue
        po_id = str(obj.get("planningObjectId") or "").strip()
        if not po_id:
            continue
        attrs = obj.get("attributes") if isinstance(obj.get("attributes"), dict) else {}
        product_id = obj.get("productId") or attrs.get("product_id") or attrs.get("PRODUCT_ID")
        location_id = obj.get("locationId") or attrs.get("location_id") or attrs.get("LOCATION_ID")
        customer_id = obj.get("customerId") or attrs.get("customer_id") or attrs.get("CUSTOMER_ID")

        buckets, values = clean_history(obj.get("series") or [])
        stored_fc_early = _stored_forecast_points(payload, po_id) if (run_post and not run_fc) else []
        if values.size == 0 and not stored_fc_early:
            continue

        # Drop trailing blank/zero months (empty grid cells) so trend continues: 40,30,20 → 10,0,-10,...
        if values.size:
            buckets, values = trim_trailing_empty(buckets, values)
        if values.size == 0 and not stored_fc_early:
            continue

        intermediates: Dict[str, Any] = {}
        if values.size:
            series_by_kf = _series_by_key_figure(payload, po_id, buckets, attrs)
            kf_outputs: Dict[str, np.ndarray] = {}
            values, intermediates = apply_preprocessing(
                values,
                prep_steps,
                return_intermediates=True,
                series_by_kf=series_by_kf,
                kf_outputs=kf_outputs,
            )

            # Forecasting reads its own Input KF, which is normally a preprocessing Output KF
            forecast_input_kf = str(payload.get("inputKeyFigure") or "").strip()
            if forecast_input_kf:
                if forecast_input_kf in kf_outputs:
                    values = np.asarray(kf_outputs[forecast_input_kf], dtype=float).copy()
                elif forecast_input_kf in series_by_kf and forecast_input_kf != primary_kf:
                    values = np.asarray(series_by_kf[forecast_input_kf], dtype=float).copy()

            values = _fill_gaps(values)
            if values.size == 0 and not stored_fc_early:
                continue

        # Write preprocessing intermediate series (historical buckets)
        for step_id, series_vals in intermediates.items():
            preprocessing_outputs.setdefault(step_id, [])
            for tb, v in zip(buckets, series_vals):
                preprocessing_outputs[step_id].append(
                    {
                        "planningObjectId": po_id,
                        "productId": product_id,
                        "locationId": location_id,
                        "customerId": customer_id,
                        "timeBucket": tb,
                        "value": float(v) if np.isfinite(v) else 0.0,
                    }
                )

        if not run_fc and not run_post:
            continue

        stored_fc = stored_fc_early
        if run_post and not run_fc:
            if not stored_fc:
                continue
            forecast_points = list(stored_fc)
            if bias_cfg is not None and values.size and buckets:
                hist_lookup = {
                    str(tb): float(values[i])
                    for i, tb in enumerate(buckets)
                    if i < values.size and np.isfinite(values[i])
                }
                residuals = []
                for pt in forecast_points:
                    actual = None
                    for key in _bucket_variants(pt["timeBucket"]):
                        if key in hist_lookup:
                            actual = hist_lookup[key]
                            break
                    if actual is None:
                        continue
                    residuals.append(pt["value"] - actual)
                if residuals:
                    corrected = apply_bias_correction(
                        np.array([pt["value"] for pt in forecast_points], dtype=float),
                        np.array(residuals, dtype=float),
                        method=bias_cfg.get("method", "mean"),
                        lookback=int(bias_cfg.get("lookbackPeriods") or 12),
                    )
                    for i, pt in enumerate(forecast_points):
                        if i < corrected.size and np.isfinite(corrected[i]):
                            pt["value"] = float(corrected[i])
            for pt in forecast_points:
                all_forecasts.append(
                    {
                        "planningObjectId": po_id,
                        "productId": product_id,
                        "locationId": location_id,
                        "timeBucket": pt["timeBucket"],
                        "value": float(pt["value"]),
                    }
                )
            continue

        candidates: Dict[str, np.ndarray] = {}
        fitted_by_name: Dict[str, np.ndarray] = {}
        std_by_name: Dict[str, float] = {}
        key_used: set = set()
        key_to_alg: Dict[str, Dict[str, Any]] = {}

        for alg in algorithms:
            try:
                atype, fc, fitted, std = _run_algorithm(alg, values, horizon, granularity)
            except Exception as e:
                logger.exception(
                    "Algorithm %s failed for %s: %s",
                    alg.get("type") or alg.get("id"),
                    po_id,
                    e,
                )
                # Skip failed algorithm; others still write to the grid
                continue
            key = _algo_key(alg, atype, key_used)
            candidates[key] = fc
            fitted_by_name[key] = fitted
            std_by_name[key] = std
            key_to_alg[key] = alg
            algorithm_meta[key] = {
                "type": atype,
                "algorithmId": alg.get("algorithmId") or alg.get("id"),
                "outputKeyFigureId": alg.get("outputKeyFigureId") or alg.get("output_key_figure_id"),
            }

        if values.size < 2 and not candidates:
            mean = float(values[0]) if values.size else 0.0
            selected_name = "single_point"
            forecast_values = np.full(horizon, mean)
            fitted = values.copy()
            std = 0.0
        else:
            selected_name, forecast_values = select_forecast(candidates, values, strategy)
            if forecast_values.size < horizon:
                pad = float(forecast_values[-1]) if forecast_values.size else float(np.nanmean(values))
                forecast_values = np.concatenate(
                    [forecast_values, np.full(horizon - forecast_values.size, pad)]
                )
            forecast_values = forecast_values[:horizon]
            fitted = fitted_by_name.get(selected_name, np.full(values.size, float(np.nanmean(values))))
            std = std_by_name.get(selected_name, 0.0)

            # Bias correction on selected forecast
            if run_post and bias_cfg is not None:
                fit_arr = np.asarray(fitted, dtype=float).ravel()
                if fit_arr.size < values.size:
                    pad = float(np.nanmean(values))
                    fit_arr = np.concatenate([np.full(values.size - fit_arr.size, pad), fit_arr])
                fit_arr = fit_arr[-values.size :]
                residuals = fit_arr - values  # over-forecast positive
                forecast_values = apply_bias_correction(
                    forecast_values,
                    residuals,
                    method=bias_cfg.get("method", "mean"),
                    lookback=int(bias_cfg.get("lookbackPeriods") or 12),
                )

        last_bucket = buckets[-1]
        future = generate_future_buckets(last_bucket, horizon, granularity)
        if len(future) < horizon:
            future = future + [f"P{i+1}" for i in range(horizon - len(future))]

        # Per-algorithm future outputs
        for key, fc in candidates.items():
            algorithm_outputs.setdefault(key, [])
            for i in range(horizon):
                val = float(fc[i]) if i < fc.size and np.isfinite(fc[i]) else float(np.nanmean(values))
                algorithm_outputs[key].append(
                    {
                        "planningObjectId": po_id,
                        "productId": product_id,
                        "locationId": location_id,
                        "timeBucket": future[i],
                        "value": val,
                    }
                )

        z = 1.96
        for i in range(horizon):
            val = float(forecast_values[i])
            if not np.isfinite(val):
                val = float(np.nanmean(values)) if values.size else 0.0
            tb = future[i]
            all_forecasts.append(
                {
                    "planningObjectId": po_id,
                    "productId": product_id,
                    "locationId": location_id,
                    "timeBucket": tb,
                    "value": val,
                }
            )
            half = abs(std) * z if std and np.isfinite(std) else abs(val) * 0.1
            all_confidence.append(
                {
                    "planningObjectId": po_id,
                    "timeBucket": tb,
                    "lower": float(val - half),
                    "upper": float(val + half),
                }
            )

        fit_arr = np.asarray(fitted, dtype=float).ravel()
        if fit_arr.size < values.size:
            pad = float(np.nanmean(values))
            fit_arr = np.concatenate([np.full(values.size - fit_arr.size, pad), fit_arr])
        fit_arr = fit_arr[-values.size :]
        for tb, fv in zip(buckets, fit_arr):
            v = float(fv) if np.isfinite(fv) else float(np.nanmean(values))
            point = {"planningObjectId": po_id, "timeBucket": tb, "value": v}
            all_expost.append(point)
            all_fitted.append(point)

        metrics = _error_metrics(values, fit_arr, list(measures), error_horizon)
        error_by_series[po_id] = metrics
        for k, v in metrics.items():
            error_accum.setdefault(k, []).append(v)

        logger.info(
            "Forecasted %s selected=%s horizon=%s n_hist=%s err_horizon=%s",
            po_id,
            selected_name,
            horizon,
            values.size,
            error_horizon,
        )

    errors = {
        k: float(np.mean(vs)) for k, vs in error_accum.items() if vs and np.isfinite(np.mean(vs))
    }

    if not all_forecasts and run_fc:
        first = time_series[0] if time_series else {}
        po = str((first or {}).get("planningObjectId") or "UNKNOWN")
        all_forecasts = [{"planningObjectId": po, "timeBucket": "1970-01", "value": 0.0}]

    # Attach output KF hints from prep steps
    prep_meta = {}
    for step in prep_steps:
        if not isinstance(step, dict):
            continue
        sid = str(step.get("stepId") or step.get("id") or "")
        if sid:
            in_kf, out_kf = step_key_figures(step)
            prep_meta[sid] = {
                "type": step.get("type"),
                "inputKeyFigureId": in_kf,
                "outputKeyFigureId": out_kf,
            }

    return {
        "status": "SUCCESS",
        "forecasts": all_forecasts,
        "finalForecast": all_forecasts,
        "algorithmOutputs": algorithm_outputs,
        "algorithmMeta": algorithm_meta,
        "preprocessingOutputs": preprocessing_outputs,
        "preprocessingMeta": prep_meta,
        "errors": errors,
        "errorMetrics": error_by_series,
        "confidence": all_confidence,
        "exPostForecasts": all_expost,
        "fitted": all_fitted,
        "snapshotReady": True,
    }
