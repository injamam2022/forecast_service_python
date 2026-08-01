"""Smoke tests for average-family algorithms, ES, and error horizon."""

import numpy as np

from app.algorithms.ets import forecast_exponential_smoothing
from app.algorithms.moving_average import (
    forecast_moving_average,
    forecast_simple_average,
    forecast_simple_moving_average,
    forecast_weighted_average,
    forecast_weighted_moving_average,
)
from app.engine import _error_metrics, run_forecast


def test_moving_average_recursive():
    hist = np.array([10.0, 20.0, 30.0])
    fc, fitted, std = forecast_moving_average(hist, 3, {"window": 3})
    # first = mean(10,20,30)=20; second = mean(20,30,20)=23.333...
    assert abs(fc[0] - 20.0) < 1e-6
    assert abs(fc[1] - (20 + 30 + 20) / 3) < 1e-6


def test_simple_average_flat():
    hist = np.array([10.0, 20.0, 30.0])
    fc, fitted, _ = forecast_simple_average(hist, 3, {})
    assert abs(fc[0] - 20.0) < 1e-6
    assert abs(fc[1] - 20.0) < 1e-6
    assert abs(fitted[0] - 20.0) < 1e-6


def test_sma_extend_vs_rolling():
    hist = np.array([10.0, 20.0, 30.0])
    fc_ext, _, _ = forecast_simple_moving_average(hist, 2, {"window": 3, "forecastMode": "extend"})
    assert abs(fc_ext[0] - 20.0) < 1e-6
    assert abs(fc_ext[1] - 20.0) < 1e-6
    fc_roll, _, _ = forecast_simple_moving_average(hist, 2, {"window": 3, "forecastMode": "rolling"})
    assert abs(fc_roll[0] - 20.0) < 1e-6
    assert abs(fc_roll[1] - (20 + 30 + 20) / 3) < 1e-6


def test_weighted_average():
    hist = np.array([10.0, 20.0, 30.0])
    fc, _, _ = forecast_weighted_average(hist, 2, {"lookback": 3, "weights": "1,2,3"})
    expected = (10 * 1 + 20 * 2 + 30 * 3) / 6
    assert abs(fc[0] - expected) < 1e-6
    assert abs(fc[1] - expected) < 1e-6


def test_wma_rolling():
    hist = np.array([10.0, 20.0, 30.0])
    fc, _, _ = forecast_weighted_moving_average(
        hist, 1, {"window": 3, "forecastMode": "extend", "weights": "1,1,1"}
    )
    assert abs(fc[0] - 20.0) < 1e-6


def test_exponential_smoothing_modes():
    hist = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0] * 4, dtype=float)
    for mode in ("single", "double", "triple", "auto"):
        fc, fitted, std = forecast_exponential_smoothing(
            hist, 3, {"mode": mode, "alpha": 0.3, "beta": 0.1, "gamma": 0.1}, seasonal_periods=6
        )
        assert fc.size == 3
        assert fitted.size == hist.size
        assert np.all(np.isfinite(fc))


def test_regression_continues_downward_trend():
    """40→30→20 must continue as 10, 0, -10, -20 (no non-negative clamp)."""
    from app.algorithms.regression import forecast_regression

    hist = np.array([40.0, 30.0, 20.0])
    fc, _, _ = forecast_regression(hist, 4, {"method": "linear"})
    assert abs(fc[0] - 10.0) < 1e-6
    assert abs(fc[1] - 0.0) < 1e-6
    assert abs(fc[2] - (-10.0)) < 1e-6
    assert abs(fc[3] - (-20.0)) < 1e-6


def test_trim_trailing_zeros_then_regression():
    """Trailing empty months must not flatten the trend to zeros."""
    payload = {
        "forecastPeriods": 4,
        "timeGranularity": "monthly",
        "errorMeasures": ["MAE"],
        "forecastingSteps": [
            {"id": "reg", "type": "regression", "enabled": True, "parameters": {"method": "linear"}},
        ],
        "timeSeries": [
            {
                "planningObjectId": "PO-1",
                "series": [
                    {"timeBucket": "2026-01", "value": 40},
                    {"timeBucket": "2026-02", "value": 30},
                    {"timeBucket": "2026-03", "value": 20},
                    {"timeBucket": "2026-04", "value": 0},
                    {"timeBucket": "2026-05", "value": 0},
                ],
            }
        ],
    }
    result = run_forecast(payload)
    vals = [p["value"] for p in result["algorithmOutputs"]["reg"]]
    assert abs(vals[0] - 10.0) < 1e-6
    assert abs(vals[1] - 0.0) < 1e-6
    assert abs(vals[2] - (-10.0)) < 1e-6
    assert abs(vals[3] - (-20.0)) < 1e-6
    # Forecast starts after last non-empty history month (March)
    assert result["algorithmOutputs"]["reg"][0]["timeBucket"].startswith("2026-04")


def test_error_horizon_last_n():
    actual = np.array([1.0, 2.0, 3.0, 10.0, 10.0])
    fitted = np.array([1.0, 2.0, 3.0, 10.0, 12.0])
    m = _error_metrics(actual, fitted, ["MAE"], horizon_periods=2)
    # last 2: |10-10| + |10-12| / 2 = 1
    assert abs(m["MAE"] - 1.0) < 1e-6


def test_run_forecast_basic():
    payload = {
        "forecastPeriods": 2,
        "timeGranularity": "monthly",
        "errorCalculationHorizonPeriods": 3,
        "errorMeasures": ["MAPE", "MAE", "BIAS"],
        "preprocessingSteps": [
            {
                "id": "PP-1",
                "type": "outlier_correction",
                "enabled": True,
                "outputKeyFigureId": "KF-CLEAN",
                "parameters": {"method": "iqr"},
            }
        ],
        "forecastingSteps": [
            {
                "id": "ALG-SA",
                "type": "simple_average",
                "enabled": True,
                "parameters": {},
            },
            {
                "id": "ALG-SMA",
                "type": "simple_moving_average",
                "enabled": True,
                "parameters": {"window": 2, "forecastMode": "rolling"},
            },
            {
                "id": "ALG-WA",
                "type": "weighted_average",
                "enabled": True,
                "parameters": {"lookback": 4},
            },
            {
                "id": "ALG-WMA",
                "type": "weighted_moving_average",
                "enabled": True,
                "parameters": {"window": 2, "forecastMode": "extend"},
            },
            {
                "id": "ALG-ES",
                "type": "exponential_smoothing",
                "enabled": True,
                "parameters": {"mode": "single", "alpha": 0.4},
            },
            {
                "id": "ALG-REG",
                "type": "regression",
                "enabled": True,
                "parameters": {"method": "linear"},
            },
        ],
        "postprocessingSteps": {"finalSelectionStrategy": "best_fit"},
        "timeSeries": [
            {
                "planningObjectId": "PO-1",
                "series": [
                    {"timeBucket": "2024-01", "value": 100},
                    {"timeBucket": "2024-02", "value": 110},
                    {"timeBucket": "2024-03", "value": 120},
                    {"timeBucket": "2024-04", "value": 130},
                ],
            }
        ],
    }
    result = run_forecast(payload)
    # Final selected series = horizon periods; per-algorithm outputs live in algorithmOutputs
    assert len(result["forecasts"]) == 2
    for key in ("ALG-SA", "ALG-SMA", "ALG-WA", "ALG-WMA", "ALG-ES", "ALG-REG"):
        assert key in result["algorithmOutputs"]
        assert len(result["algorithmOutputs"][key]) == 2
    assert "PP-1" in result["preprocessingOutputs"]
    assert len(result["preprocessingOutputs"]["PP-1"]) == 4
