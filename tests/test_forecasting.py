"""
PyMBAL - FORECASTING LAYER VALIDATION

This test validates the forecasting layer above the already validated
model-selection and volumetric layers.

The forecasting engine is tested as a reservoir-state projection
engine. It does not perform production-rate forecasting.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.model_selection import (
    ModelSelectionConstraints,
    ReservoirModelSelector,
)

from src.forecasting.forecasting import (
    ForecastingResult,
    MaterialBalanceForecaster,
)


# ======================================================================
# TEST DATA
# ======================================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT
    / "data"
    / "sample_data"
    / "synthetic_8_cases"
)

CASE_1_PATHS = [
    DATA_DIR
    / "case_1_no_gas_no_aquifer"
    / "case_1_no_gas_no_aquifer.csv",
]


def load_case_1():
    path = next(
        (
            candidate
            for candidate in CASE_1_PATHS
            if candidate.exists()
        ),
        None,
    )

    if path is None:
        raise FileNotFoundError(
            "Could not locate case_1_no_gas_no_aquifer.csv."
        )

    df = pd.read_csv(path)

    required = [
        "Days",
        "Pressure_psia",
        "F_rb",
        "Eo_rb_per_STB",
        "Eg_rb_per_STB",
        "Efw_rb_per_STB",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"CASE 1 is missing required columns: {missing}"
        )

    return df


# ======================================================================
# BUILD MODEL SELECTION RESULT
# ======================================================================

def build_case_1_model_selection(df):
    selector = ReservoirModelSelector(
        F_rb=df["F_rb"].to_numpy(dtype=float),
        Eo_rb_per_STB=df[
            "Eo_rb_per_STB"
        ].to_numpy(dtype=float),
        Eg_rb_per_STB=df[
            "Eg_rb_per_STB"
        ].to_numpy(dtype=float),
        Efw_rb_per_STB=df[
            "Efw_rb_per_STB"
        ].to_numpy(dtype=float),
        time_days=df[
            "Days"
        ].to_numpy(dtype=float),
        pressure_psia=df[
            "Pressure_psia"
        ].to_numpy(dtype=float),
    )

    result = selector.select(
        constraints=ModelSelectionConstraints(
            gas_cap_present=False,
            aquifer_present=False,
        )
    )

    return result


# ======================================================================
# BUILD FORECASTER
# ======================================================================

def build_forecaster(df):
    model_selection = build_case_1_model_selection(df)

    forecaster = MaterialBalanceForecaster(
        model_selection_result=model_selection,
        historical_time_days=df[
            "Days"
        ].to_numpy(dtype=float),
        historical_pressure_psia=df[
            "Pressure_psia"
        ].to_numpy(dtype=float),
        historical_F_rb=df[
            "F_rb"
        ].to_numpy(dtype=float),
        historical_Eo_rb_per_STB=df[
            "Eo_rb_per_STB"
        ].to_numpy(dtype=float),
        historical_Eg_rb_per_STB=df[
            "Eg_rb_per_STB"
        ].to_numpy(dtype=float),
        historical_Efw_rb_per_STB=df[
            "Efw_rb_per_STB"
        ].to_numpy(dtype=float),
    )

    return forecaster


# ======================================================================
# FORECAST PATH
# ======================================================================

def build_simple_forecast(df):
    """
    Construct a controlled synthetic future pressure/PVT path.

    We deliberately provide future expansion terms rather than asking
    the forecasting engine to invent them.
    """

    last_time = float(df["Days"].iloc[-1])
    last_pressure = float(
        df["Pressure_psia"].iloc[-1]
    )

    last_Eo = float(
        df["Eo_rb_per_STB"].iloc[-1]
    )

    last_Eg = float(
        df["Eg_rb_per_STB"].iloc[-1]
    )

    last_Efw = float(
        df["Efw_rb_per_STB"].iloc[-1]
    )

    future_time = np.array(
        [
            last_time + 100.0,
            last_time + 200.0,
            last_time + 300.0,
        ]
    )

    future_pressure = np.array(
        [
            last_pressure - 100.0,
            last_pressure - 200.0,
            last_pressure - 300.0,
        ]
    )

    # Controlled synthetic continuation.
    future_Eo = np.array(
        [
            last_Eo * 1.10,
            last_Eo * 1.20,
            last_Eo * 1.30,
        ]
    )

    future_Eg = np.array(
        [
            last_Eg * 1.10,
            last_Eg * 1.20,
            last_Eg * 1.30,
        ]
    )

    future_Efw = np.array(
        [
            last_Efw * 1.10,
            last_Efw * 1.20,
            last_Efw * 1.30,
        ]
    )

    return (
        future_time,
        future_pressure,
        future_Eo,
        future_Eg,
        future_Efw,
    )


# ======================================================================
# TEST 1
# ======================================================================

def test_result_structure():
    print("=" * 70)
    print("FORECASTING RESULT STRUCTURE")
    print("=" * 70)

    df = load_case_1()
    forecaster = build_forecaster(df)

    (
        future_time,
        future_pressure,
        future_Eo,
        future_Eg,
        future_Efw,
    ) = build_simple_forecast(df)

    result = forecaster.forecast(
        forecast_time_days=future_time,
        forecast_pressure_psia=future_pressure,
        forecast_Eo_rb_per_STB=future_Eo,
        forecast_Eg_rb_per_STB=future_Eg,
        forecast_Efw_rb_per_STB=future_Efw,
    )

    assert isinstance(
        result,
        ForecastingResult,
    )

    assert isinstance(
        result.historical_history,
        pd.DataFrame,
    )

    assert isinstance(
        result.forecast_history,
        pd.DataFrame,
    )

    required_columns = [
        "Time_days",
        "Pressure_psia",
        "Eo_rb_per_STB",
        "Eg_rb_per_STB",
        "Efw_rb_per_STB",
        "We_rb",
        "F_rb",
    ]

    for column in required_columns:
        assert column in result.forecast_history.columns

    print("✓ ForecastingResult returned.")
    print("✓ Historical history exposed.")
    print("✓ Forecast history exposed.")
    print("✓ Forecast pressure exposed.")
    print("✓ Forecast material-balance F exposed.")
    print("✓ Result structure PASSED.")
    print()


# ======================================================================
# TEST 2
# ======================================================================

def test_historical_data_preserved():
    print("=" * 70)
    print("HISTORICAL DATA PRESERVATION")
    print("=" * 70)

    df = load_case_1()

    original_time = df[
        "Days"
    ].to_numpy(dtype=float).copy()

    original_pressure = df[
        "Pressure_psia"
    ].to_numpy(dtype=float).copy()

    original_F = df[
        "F_rb"
    ].to_numpy(dtype=float).copy()

    forecaster = build_forecaster(df)

    (
        future_time,
        future_pressure,
        future_Eo,
        future_Eg,
        future_Efw,
    ) = build_simple_forecast(df)

    result = forecaster.forecast(
        forecast_time_days=future_time,
        forecast_pressure_psia=future_pressure,
        forecast_Eo_rb_per_STB=future_Eo,
        forecast_Eg_rb_per_STB=future_Eg,
        forecast_Efw_rb_per_STB=future_Efw,
    )

    assert np.array_equal(
        result.historical_history[
            "Time_days"
        ].to_numpy(),
        original_time,
    )

    assert np.array_equal(
        result.historical_history[
            "Pressure_psia"
        ].to_numpy(),
        original_pressure,
    )

    assert np.array_equal(
        result.historical_history[
            "F_rb"
        ].to_numpy(),
        original_F,
    )

    print("✓ Historical time preserved.")
    print("✓ Historical pressure preserved.")
    print("✓ Historical F preserved.")
    print("✓ Historical data preservation PASSED.")
    print()


# ======================================================================
# TEST 3
# ======================================================================

def test_material_balance_equation():
    print("=" * 70)
    print("FUTURE MATERIAL-BALANCE EQUATION")
    print("=" * 70)

    df = load_case_1()
    forecaster = build_forecaster(df)

    (
        future_time,
        future_pressure,
        future_Eo,
        future_Eg,
        future_Efw,
    ) = build_simple_forecast(df)

    result = forecaster.forecast(
        forecast_time_days=future_time,
        forecast_pressure_psia=future_pressure,
        forecast_Eo_rb_per_STB=future_Eo,
        forecast_Eg_rb_per_STB=future_Eg,
        forecast_Efw_rb_per_STB=future_Efw,
    )

    expected_F = (
        forecaster.N
        * (
            future_Eo
            + forecaster.m * future_Eg
            + (1.0 + forecaster.m) * future_Efw
        )
    )

    calculated_F = result.forecast_history[
        "F_rb"
    ].to_numpy(dtype=float)

    assert np.allclose(
        calculated_F,
        expected_F,
        rtol=1.0e-10,
        atol=1.0e-10,
    )

    print(
        "✓ F = N[Eo + mEg + (1+m)Efw] verified."
    )

    print("✓ Future material-balance equation PASSED.")
    print()


# ======================================================================
# TEST 4
# ======================================================================

def test_forecast_time_validation():
    print("=" * 70)
    print("FORECAST INPUT VALIDATION")
    print("=" * 70)

    df = load_case_1()
    forecaster = build_forecaster(df)

    (
        future_time,
        future_pressure,
        future_Eo,
        future_Eg,
        future_Efw,
    ) = build_simple_forecast(df)

    # Duplicate historical time.
    invalid_time = future_time.copy()
    invalid_time[0] = df["Days"].iloc[-1]

    try:
        forecaster.forecast(
            forecast_time_days=invalid_time,
            forecast_pressure_psia=future_pressure,
            forecast_Eo_rb_per_STB=future_Eo,
            forecast_Eg_rb_per_STB=future_Eg,
            forecast_Efw_rb_per_STB=future_Efw,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Duplicate historical/forecast time was not rejected."
        )

    # Non-increasing future time.
    invalid_time = future_time.copy()
    invalid_time[1] = invalid_time[0]

    try:
        forecaster.forecast(
            forecast_time_days=invalid_time,
            forecast_pressure_psia=future_pressure,
            forecast_Eo_rb_per_STB=future_Eo,
            forecast_Eg_rb_per_STB=future_Eg,
            forecast_Efw_rb_per_STB=future_Efw,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Non-increasing forecast time was not rejected."
        )

    print("✓ Invalid forecast time rejected.")
    print("✓ Forecast input validation PASSED.")
    print()


# ======================================================================
# TEST 5
# ======================================================================

def test_pressure_decline():
    print("=" * 70)
    print("PRESSURE DECLINE CALCULATION")
    print("=" * 70)

    df = load_case_1()
    forecaster = build_forecaster(df)

    (
        future_time,
        future_pressure,
        future_Eo,
        future_Eg,
        future_Efw,
    ) = build_simple_forecast(df)

    result = forecaster.forecast(
        forecast_time_days=future_time,
        forecast_pressure_psia=future_pressure,
        forecast_Eo_rb_per_STB=future_Eo,
        forecast_Eg_rb_per_STB=future_Eg,
        forecast_Efw_rb_per_STB=future_Efw,
    )

    expected_decline = (
        df["Pressure_psia"].iloc[-1]
        - future_pressure[-1]
    )

    assert np.isclose(
        result.pressure_decline_psia,
        expected_decline,
    )

    assert result.pressure_decline_percent > 0.0

    print(
        f"✓ Forecast pressure decline: "
        f"{result.pressure_decline_psia:.2f} psi"
    )

    print(
        f"✓ Pressure decline: "
        f"{result.pressure_decline_percent:.2f}%"
    )

    print("✓ Pressure forecast metrics PASSED.")
    print()


# ======================================================================
# TEST 6
# ======================================================================

def test_zero_aquifer_handling():
    print("=" * 70)
    print("ZERO AQUIFER FORECAST HANDLING")
    print("=" * 70)

    df = load_case_1()
    forecaster = build_forecaster(df)

    (
        future_time,
        future_pressure,
        future_Eo,
        future_Eg,
        future_Efw,
    ) = build_simple_forecast(df)

    result = forecaster.forecast(
        forecast_time_days=future_time,
        forecast_pressure_psia=future_pressure,
        forecast_Eo_rb_per_STB=future_Eo,
        forecast_Eg_rb_per_STB=future_Eg,
        forecast_Efw_rb_per_STB=future_Efw,
    )

    assert np.allclose(
        result.forecast_history[
            "We_rb"
        ].to_numpy(dtype=float),
        0.0,
    )

    print("✓ No-aquifer forecast correctly uses We = 0.")
    print("✓ Zero-aquifer handling PASSED.")
    print()


# ======================================================================
# TEST 7
# ======================================================================

def test_positive_forecast_state():
    print("=" * 70)
    print("FORECAST STATE VALIDATION")
    print("=" * 70)

    df = load_case_1()
    forecaster = build_forecaster(df)

    (
        future_time,
        future_pressure,
        future_Eo,
        future_Eg,
        future_Efw,
    ) = build_simple_forecast(df)

    result = forecaster.forecast(
        forecast_time_days=future_time,
        forecast_pressure_psia=future_pressure,
        forecast_Eo_rb_per_STB=future_Eo,
        forecast_Eg_rb_per_STB=future_Eg,
        forecast_Efw_rb_per_STB=future_Efw,
    )

    assert np.all(
        result.forecast_history[
            "F_rb"
        ].to_numpy(dtype=float) >= 0.0
    )

    assert np.all(
        result.forecast_history[
            "Pressure_psia"
        ].to_numpy(dtype=float) > 0.0
    )

    assert np.isfinite(
        result.final_forecast_f_rb
    )

    assert result.forecast_status == "SUCCESS"

    print("✓ Forecast F values are physical.")
    print("✓ Forecast pressures are physical.")
    print("✓ Final forecast state is finite.")
    print("✓ Forecast status = SUCCESS.")
    print("✓ Forecast state validation PASSED.")
    print()


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("PyMBAL - FORECASTING ENGINE VALIDATION")
    print("=" * 70)
    print()
    print(
        "This test validates material-balance reservoir-state "
        "forecasting above the already validated analysis layers."
    )
    print()

    test_result_structure()
    test_historical_data_preserved()
    test_material_balance_equation()
    test_forecast_time_validation()
    test_pressure_decline()
    test_zero_aquifer_handling()
    test_positive_forecast_state()

    print("=" * 70)
    print("✓ ALL FORECASTING ENGINE CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()