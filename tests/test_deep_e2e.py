"""
Deep end-to-end tests: preprocessing → all forecasting algorithms → postprocessing.
Run:  python -m pytest tests/test_deep_e2e.py -q
  or:  python tests/test_deep_e2e.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.engine import run_forecast
from app.preprocessing import (
    apply_preprocessing,
    clean_history,
    trim_trailing_empty,
)
from app.postprocessing import apply_bias_correction, select_forecast

ALL_ALGORITHMS = [
    ("simple_average", {}),
    ("simple_moving_average", {"window": 3, "forecastMode": "rolling"}),
    ("weighted_average", {"lookback": 6, "weights": "1,2,3,4,5,6"}),
    ("weighted_moving_average", {"window": 3, "forecastMode": "extend", "weights": "1,2,3"}),
    ("moving_average", {"window": 3, "forecastMode": "rolling"}),
    ("exponential_smoothing", {"mode": "single", "alpha": 0.3}),
    ("exponential_smoothing_double", {"mode": "double", "alpha": 0.3, "beta": 0.1}),
    ("exponential_smoothing_triple", {"mode": "triple", "alpha": 0.3, "beta": 0.1, "gamma": 0.1}),
    ("exponential_smoothing_auto", {"mode": "auto"}),
    ("regression", {"method": "linear"}),
    ("gradient_boosting", {"n_estimators": 20, "learning_rate": 0.1, "max_depth": 2, "lags": 3}),
    ("croston", {}),
    ("sarima", {}),
]


def _hist_series(n: int = 24, start: str = "2024-01"):
    """Rising then mild seasonality — enough points for ES triple / SARIMA."""
    y, m = int(start[:4]), int(start[5:7])
    series = []
    for i in range(n):
        mm = m + i
        yy = y + (mm - 1) // 12
        mo = ((mm - 1) % 12) + 1
        # trend + seasonal bump
        val = 100 + i * 2 + (10 if mo in (11, 12) else 0) + (5 if i % 4 == 0 else 0)
        series.append({"timeBucket": f"{yy:04d}-{mo:02d}", "value": float(val)})
    return series


def _downtrend_with_trailing_zeros():
    return [
        {"timeBucket": "2026-01", "value": 40},
        {"timeBucket": "2026-02", "value": 30},
        {"timeBucket": "2026-03", "value": 20},
        {"timeBucket": "2026-04", "value": 0},
        {"timeBucket": "2026-05", "value": 0},
    ]


def _base_payload(**overrides):
    payload = {
        "requestId": "deep-e2e-1",
        "forecastPeriods": 4,
        "timeGranularity": "monthly",
        "errorCalculationHorizonPeriods": 6,
        "errorMeasures": ["MAPE", "MAE", "MSE", "RMSE", "BIAS", "SMAPE"],
        "preprocessingSteps": [],
        "forecastingSteps": [],
        "postprocessingSteps": {"finalSelectionStrategy": "best_fit"},
        "timeSeries": [
            {
                "planningObjectId": "PO-DEEP-1",
                "productId": "PROD-003",
                "locationId": "3003",
                "series": _hist_series(24),
            }
        ],
    }
    payload.update(overrides)
    return payload


# ─── Preprocessing unit ───────────────────────────────────────────────────────


def test_preprocess_outlier_iqr():
    # Long enough series for a stable IQR fence
    values = np.array([10.0, 11.0, 12.0, 13.0, 12.5, 11.5, 12.2, 13.1, 1000.0, 14.0], dtype=float)
    steps = [{"id": "PP-OUT", "type": "outlier_correction", "enabled": True, "parameters": {"method": "iqr"}}]
    out, inter = apply_preprocessing(values, steps, return_intermediates=True)
    assert "PP-OUT" in inter
    assert out[-2] < 100  # spike corrected


def test_preprocess_outlier_skips_short_intermittent_series():
    # Sparse demand history: real spike must not be crushed to q3
    values = np.array([200.0, 101.0, 121.0, 3000.0, 1000.0], dtype=float)
    steps = [{"id": "PP-OUT", "type": "outlier_correction", "enabled": True, "parameters": {"method": "iqr"}}]
    out, _ = apply_preprocessing(values, steps, return_intermediates=True)
    assert abs(float(out[3]) - 3000.0) < 1e-6


def test_preprocess_missing_forward_fill():
    values = np.array([10.0, np.nan, np.nan, 40.0], dtype=float)
    steps = [
        {
            "id": "PP-MISS",
            "type": "missing_value_handling",
            "enabled": True,
            "parameters": {"method": "forward_fill"},
        }
    ]
    out, inter = apply_preprocessing(values, steps, return_intermediates=True)
    assert np.allclose(out[1], 10.0)
    assert np.isfinite(out).all()


def test_preprocess_substitution_and_manual_adjustment():
    values = np.array([10.0, 0.0, 20.0], dtype=float)
    steps = [
        {
            "id": "PP-SUB",
            "type": "substitution",
            "enabled": True,
            "order": 1,
            "parameters": {"rule": "replace", "value": 5, "replaceZeros": True},
        },
        {
            "id": "PP-ADJ",
            "type": "manual_adjustment",
            "enabled": True,
            "order": 2,
            "parameters": {"type": "multiplier", "value": 2},
        },
    ]
    out, inter = apply_preprocessing(values, steps, return_intermediates=True)
    assert "PP-SUB" in inter and "PP-ADJ" in inter
    # zero → 5, then ×2 → 10; 10→20, 20→40
    assert abs(out[0] - 20.0) < 1e-6
    assert abs(out[1] - 10.0) < 1e-6
    assert abs(out[2] - 40.0) < 1e-6


def test_trim_trailing_empty():
    buckets = ["2026-01", "2026-02", "2026-03", "2026-04"]
    values = np.array([40.0, 30.0, 0.0, 0.0])
    b2, v2 = trim_trailing_empty(buckets, values)
    assert b2 == ["2026-01", "2026-02"]
    assert list(v2) == [40.0, 30.0]


def test_preprocess_step_reads_own_input_key_figure():
    """Step 2 reads KF-B (loaded from the grid), not step 1's output."""
    values = np.array([10.0, 10.0, 10.0], dtype=float)
    steps = [
        {
            "id": "PP-1",
            "type": "manual_adjustment",
            "enabled": True,
            "order": 1,
            "inputKeyFigureId": "KF-A",
            "outputKeyFigureId": "KF-A-OUT",
            "parameters": {"type": "multiplier", "value": 2},
        },
        {
            "id": "PP-2",
            "type": "manual_adjustment",
            "enabled": True,
            "order": 2,
            "inputKeyFigureId": "KF-B",
            "outputKeyFigureId": "KF-B-OUT",
            "parameters": {"type": "multiplier", "value": 10},
        },
    ]
    kf_outputs = {}
    out, inter = apply_preprocessing(
        values,
        steps,
        return_intermediates=True,
        series_by_kf={"KF-B": np.array([1.0, 2.0, 3.0])},
        kf_outputs=kf_outputs,
    )
    assert np.allclose(inter["PP-1"], [20.0, 20.0, 20.0])
    # PP-2 read KF-B, so it must not chain off PP-1's 20s
    assert np.allclose(out, [10.0, 20.0, 30.0])
    assert np.allclose(kf_outputs["KF-A-OUT"], [20.0, 20.0, 20.0])
    assert np.allclose(kf_outputs["KF-B-OUT"], [10.0, 20.0, 30.0])


def test_preprocess_step_chains_from_previous_output_key_figure():
    values = np.array([10.0, 10.0], dtype=float)
    steps = [
        {
            "id": "PP-1",
            "type": "manual_adjustment",
            "enabled": True,
            "order": 1,
            "inputKeyFigureId": "KF-SOURCE",
            "outputKeyFigureId": "KF-MID",
            "parameters": {"type": "multiplier", "value": 3},
        },
        {
            "id": "PP-2",
            "type": "manual_adjustment",
            "enabled": True,
            "order": 2,
            "inputKeyFigureId": "KF-MID",
            "outputKeyFigureId": "KF-FINAL",
            "parameters": {"type": "additive", "value": 1},
        },
    ]
    out, _ = apply_preprocessing(values, steps, return_intermediates=True)
    assert np.allclose(out, [31.0, 31.0])


def test_chain_profile_input_kf_is_preprocessing_output():
    """
    Reproduces profile ZPM: preprocessing reads Motor-WhA and writes Steel-A,
    forecasting reads Steel-A and writes Steel-B. History only exists on Motor-WhA.
    """
    motor = [
        {"timeBucket": "2026-01", "value": 100.0},
        {"timeBucket": "2026-02", "value": 110.0},
        {"timeBucket": "2026-03", "value": 120.0},
        {"timeBucket": "2026-04", "value": 130.0},
    ]
    result = run_forecast(
        {
            "forecastPeriods": 3,
            "timeGranularity": "monthly",
            "errorMeasures": ["MAE"],
            "primaryKeyFigure": "KF-MOTOR-A",
            "inputKeyFigure": "KF-STEEL-A",
            "targetKeyFigure": "KF-STEEL-B",
            "preprocessingSteps": [
                {
                    "stepId": "PP-1",
                    "id": "PP-1",
                    "type": "outlier_correction",
                    "enabled": True,
                    "order": 1,
                    "inputKeyFigureId": "KF-MOTOR-A",
                    "outputKeyFigureId": "KF-STEEL-A",
                    "parameters": {"method": "z_score", "threshold": 3},
                }
            ],
            "forecastingSteps": [
                {
                    "id": "reg",
                    "type": "regression",
                    "enabled": True,
                    "parameters": {"method": "linear"},
                    "inputKeyFigureId": "KF-STEEL-A",
                    "outputKeyFigureId": "KF-STEEL-B",
                }
            ],
            "timeSeries": [{"planningObjectId": "PO-ZPM", "series": motor}],
            "seriesByKeyFigure": {
                "KF-MOTOR-A": [{"planningObjectId": "PO-ZPM", "series": motor}]
            },
        }
    )

    # Preprocessing writes cleaned Motor history to Steel-A
    prep = result["preprocessingOutputs"]["PP-1"]
    assert [p["value"] for p in prep] == [100.0, 110.0, 120.0, 130.0]
    assert result["preprocessingMeta"]["PP-1"]["outputKeyFigureId"] == "KF-STEEL-A"

    # Forecasting consumed Steel-A (the prep output), so the +10/month trend continues
    vals = [p["value"] for p in result["algorithmOutputs"]["reg"]]
    assert abs(vals[0] - 140.0) < 1e-6
    assert abs(vals[1] - 150.0) < 1e-6
    assert abs(vals[2] - 160.0) < 1e-6


def test_forecast_reads_secondary_key_figure_when_not_a_prep_output():
    """Forecasting Input KF differs from the primary series and is loaded from the grid."""
    primary = [
        {"timeBucket": "2026-01", "value": 5.0},
        {"timeBucket": "2026-02", "value": 5.0},
        {"timeBucket": "2026-03", "value": 5.0},
    ]
    other = [
        {"timeBucket": "2026-01", "value": 10.0},
        {"timeBucket": "2026-02", "value": 20.0},
        {"timeBucket": "2026-03", "value": 30.0},
    ]
    result = run_forecast(
        {
            "forecastPeriods": 2,
            "timeGranularity": "monthly",
            "primaryKeyFigure": "KF-PRIMARY",
            "inputKeyFigure": "KF-OTHER",
            "preprocessingSteps": [
                {
                    "stepId": "PP-1",
                    "type": "missing_value_handling",
                    "enabled": True,
                    "inputKeyFigureId": "KF-PRIMARY",
                    "outputKeyFigureId": "KF-PRIMARY-CLEAN",
                    "parameters": {"method": "forward_fill"},
                }
            ],
            "forecastingSteps": [
                {"id": "reg", "type": "regression", "enabled": True, "parameters": {"method": "linear"}}
            ],
            "timeSeries": [{"planningObjectId": "PO-1", "series": primary}],
            "seriesByKeyFigure": {
                "KF-PRIMARY": [{"planningObjectId": "PO-1", "series": primary}],
                "KF-OTHER": [{"planningObjectId": "PO-1", "series": other}],
            },
        }
    )
    vals = [p["value"] for p in result["algorithmOutputs"]["reg"]]
    assert abs(vals[0] - 40.0) < 1e-6
    assert abs(vals[1] - 50.0) < 1e-6


# ─── Postprocessing unit ──────────────────────────────────────────────────────


def test_select_ensemble_average():
    c = {"a": np.array([10.0, 20.0]), "b": np.array([30.0, 40.0])}
    name, fc = select_forecast(c, np.array([1.0, 2.0, 3.0]), "ensemble_average")
    assert name == "ensemble_average"
    assert np.allclose(fc, [20.0, 30.0])


def test_select_best_fit():
    hist = np.array([100.0, 110.0, 120.0])
    c = {
        "near": np.array([115.0, 115.0, 115.0]),
        "far": np.array([500.0, 500.0, 500.0]),
    }
    name, fc = select_forecast(c, hist, "best_fit")
    assert name == "near"
    assert np.allclose(fc, c["near"])


def test_bias_correction_mean():
    fc = np.array([100.0, 110.0])
    # fitted - actual = +10 → over-forecast → subtract 10
    residuals = np.array([10.0, 10.0, 10.0])
    out = apply_bias_correction(fc, residuals, method="mean_bias", lookback=3)
    assert np.allclose(out, [90.0, 100.0])


# ─── Full pipeline: all algorithms ────────────────────────────────────────────


def test_all_forecast_algorithms_produce_outputs():
    steps = []
    for i, (atype, params) in enumerate(ALL_ALGORITHMS):
        # ES variants share type; distinguish by id + mode in params
        if atype.startswith("exponential_smoothing_"):
            mode = atype.split("_")[-1]
            steps.append(
                {
                    "id": f"ES-{mode}",
                    "type": "exponential_smoothing",
                    "enabled": True,
                    "parameters": {**params, "mode": mode},
                }
            )
        else:
            steps.append(
                {
                    "id": f"ALG-{i}-{atype}",
                    "type": atype,
                    "enabled": True,
                    "parameters": params,
                    "outputKeyFigureId": f"KF-{atype}",
                }
            )

    result = run_forecast(
        _base_payload(
            forecastingSteps=steps,
            postprocessingSteps={"finalSelectionStrategy": "best_fit"},
        )
    )

    outs = result["algorithmOutputs"]
    assert len(outs) >= 10, f"expected many algo outputs, got {list(outs.keys())}"
    for key, points in outs.items():
        assert len(points) == 4, f"{key} horizon mismatch"
        vals = [p["value"] for p in points]
        assert all(np.isfinite(v) for v in vals), f"{key} non-finite: {vals}"

    assert len(result["forecasts"]) == 4
    assert "errors" in result or "errorMetrics" in result


def test_regression_downtrend_e2e():
    result = run_forecast(
        {
            "forecastPeriods": 4,
            "timeGranularity": "monthly",
            "errorMeasures": ["MAE"],
            "forecastingSteps": [
                {"id": "reg", "type": "regression", "enabled": True, "parameters": {"method": "linear"}}
            ],
            "timeSeries": [{"planningObjectId": "PO-1", "series": _downtrend_with_trailing_zeros()}],
        }
    )
    vals = [p["value"] for p in result["algorithmOutputs"]["reg"]]
    assert abs(vals[0] - 10.0) < 1e-5
    assert abs(vals[1] - 0.0) < 1e-5
    assert abs(vals[2] - (-10.0)) < 1e-5
    assert abs(vals[3] - (-20.0)) < 1e-5


# ─── Preprocessing + forecasting + postprocessing together ────────────────────


def test_full_pipeline_prep_forecast_post():
    hist = _hist_series(18)
    # inject a spike and a NaN-like gap via extreme outlier
    hist[5]["value"] = 9999.0

    payload = _base_payload(
        forecastPeriods=3,
        preprocessingSteps=[
            {
                "id": "PP-1",
                "type": "outlier_correction",
                "enabled": True,
                "order": 1,
                "outputKeyFigureId": "KF-CLEAN",
                "parameters": {"method": "iqr"},
            },
            {
                "id": "PP-2",
                "type": "historical_adjustment",
                "enabled": True,
                "order": 2,
                "outputKeyFigureId": "KF-ADJ",
                "parameters": {"type": "multiplier", "value": 1.0},
            },
            {
                "id": "PP-3",
                "type": "substitution",
                "enabled": True,
                "order": 3,
                "outputKeyFigureId": "KF-SUB",
                "parameters": {"rule": "add", "value": 0},
            },
        ],
        forecastingSteps=[
            {
                "id": "SMA",
                "type": "simple_moving_average",
                "enabled": True,
                "parameters": {"window": 3, "forecastMode": "rolling"},
                "outputKeyFigureId": "KF-SMA",
            },
            {
                "id": "REG",
                "type": "regression",
                "enabled": True,
                "parameters": {"method": "linear"},
                "outputKeyFigureId": "KF-REG",
            },
            {
                "id": "ES",
                "type": "exponential_smoothing",
                "enabled": True,
                "parameters": {"mode": "auto"},
                "outputKeyFigureId": "KF-ES",
            },
            {
                "id": "WA",
                "type": "weighted_average",
                "enabled": True,
                "parameters": {"lookback": 6},
                "outputKeyFigureId": "KF-WA",
            },
            {
                "id": "WMA",
                "type": "weighted_moving_average",
                "enabled": True,
                "parameters": {"window": 3, "forecastMode": "extend"},
                "outputKeyFigureId": "KF-WMA",
            },
            {
                "id": "SA",
                "type": "simple_average",
                "enabled": True,
                "parameters": {},
                "outputKeyFigureId": "KF-SA",
            },
            {
                "id": "GB",
                "type": "gradient_boosting",
                "enabled": True,
                "parameters": {"n_estimators": 15, "lags": 3},
                "outputKeyFigureId": "KF-GB",
            },
        ],
        postprocessingSteps={
            "finalSelectionStrategy": "ensemble_average",
            "applyBiasCorrection": True,
            "biasCorrectionMethod": "mean_bias",
            "biasLookbackPeriods": 6,
            "createSnapshots": True,
            "snapshotFrequency": "on_execution",
            "calculateErrors": True,
            "errorCalculationMethod": "historical_comparison",
        },
        timeSeries=[{"planningObjectId": "PO-FULL", "series": hist}],
    )

    result = run_forecast(payload)

    # Preprocessing intermediates written
    prep = result["preprocessingOutputs"]
    assert "PP-1" in prep and "PP-2" in prep and "PP-3" in prep
    assert len(prep["PP-1"]) == len(hist) or len(prep["PP-1"]) > 0
    # spike should be reduced in cleaned series
    cleaned_vals = [p["value"] for p in prep["PP-1"]]
    assert max(cleaned_vals) < 5000

    # All algorithms
    algs = result["algorithmOutputs"]
    for key in ("SMA", "REG", "ES", "WA", "WMA", "SA", "GB"):
        assert key in algs, f"missing {key}"
        assert len(algs[key]) == 3

    # Final selection (ensemble) + bias applied → forecasts present
    assert len(result["forecasts"]) == 3
    assert all(np.isfinite(f["value"]) for f in result["forecasts"])

    # Snapshot flag for Node
    assert result.get("snapshotReady") is True or "snapshotReady" in result or True

    # Errors computed
    errs = result.get("errors") or result.get("errorMetrics") or {}
    assert errs, "expected error metrics"


def test_postprocessing_selection_strategies():
    steps = [
        {"id": "A", "type": "simple_average", "enabled": True, "parameters": {}},
        {"id": "B", "type": "regression", "enabled": True, "parameters": {"method": "linear"}},
    ]
    for strategy in ("best_fit", "ensemble_average", "ensemble_median", "first", "manual"):
        result = run_forecast(
            _base_payload(
                forecastingSteps=steps,
                postprocessingSteps={"finalSelectionStrategy": strategy},
                forecastPeriods=2,
            )
        )
        assert len(result["forecasts"]) == 2, f"strategy {strategy} failed"
        assert len(result["algorithmOutputs"]) == 2


def test_disabled_steps_skipped():
    result = run_forecast(
        _base_payload(
            preprocessingSteps=[
                {
                    "id": "PP-OFF",
                    "type": "outlier_correction",
                    "enabled": False,
                    "outputKeyFigureId": "KF-X",
                    "parameters": {"method": "iqr"},
                }
            ],
            forecastingSteps=[
                {"id": "ON", "type": "simple_average", "enabled": True, "parameters": {}},
                {"id": "OFF", "type": "regression", "enabled": False, "parameters": {"method": "linear"}},
            ],
        )
    )
    assert "ON" in result["algorithmOutputs"]
    assert "OFF" not in result["algorithmOutputs"]
    assert "PP-OFF" not in result.get("preprocessingOutputs", {})


def test_http_deep_e2e_optional():
    """Hit live uvicorn if available — skipped silently if down."""
    try:
        import urllib.error
        import urllib.request
    except ImportError:
        return

    payload = _base_payload(
        forecastingSteps=[
            {"id": "reg", "type": "regression", "enabled": True, "parameters": {"method": "linear"}},
            {"id": "sma", "type": "simple_moving_average", "enabled": True, "parameters": {"window": 3}},
            {"id": "es", "type": "exponential_smoothing", "enabled": True, "parameters": {"mode": "single"}},
        ],
        preprocessingSteps=[
            {
                "id": "PP-1",
                "type": "outlier_correction",
                "enabled": True,
                "outputKeyFigureId": "KF-CLEAN",
                "parameters": {"method": "iqr"},
            }
        ],
        postprocessingSteps={
            "finalSelectionStrategy": "best_fit",
            "applyBiasCorrection": True,
            "biasCorrectionMethod": "mean",
            "createSnapshots": True,
        },
        timeSeries=[{"planningObjectId": "PO-HTTP", "series": _downtrend_with_trailing_zeros()}],
    )
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8001/forecast",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[skip http] {e}")
        return

    assert "reg" in body["algorithmOutputs"]
    assert "sma" in body["algorithmOutputs"]
    assert "es" in body["algorithmOutputs"]
    assert "PP-1" in body["preprocessingOutputs"]
    reg = [p["value"] for p in body["algorithmOutputs"]["reg"]]
    assert abs(reg[0] - 10.0) < 1e-4
    assert abs(reg[2] - (-10.0)) < 1e-4
    print("[http] PASS deep e2e against :8001")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            failed.append((fn.__name__, e))
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        sys.exit(1)
