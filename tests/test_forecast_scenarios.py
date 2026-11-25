"""
PyMBAL - FORECAST SCENARIO / DCA VALIDATION

Validates:
    - Exponential DCA
    - Harmonic DCA
    - Hyperbolic DCA
    - Parameter fitting
    - Historical fit metrics
    - Production-rate forecasting
    - Cumulative production
    - EUR
    - Model comparison
"""

import numpy as np

from src.forecasting.scenarios import (
    DCA_MODEL_NAMES,
    DCAFitResult,
    ForecastScenarioEngine,
    ForecastScenarioResult,
)


# ======================================================================
# SYNTHETIC PRODUCTION HISTORY
# ======================================================================

def build_synthetic_history():
    """
    Generate a controlled hyperbolic production history.

    This gives us a known physical curve against which the DCA
    implementation can be tested.
    """

    time = np.linspace(
        0.0,
        1000.0,
        41,
    )

    qi = 1000.0
    Di = 0.001
    b = 0.8

    rate = (
        qi
        / (
            1.0
            + b * Di * time
        )
        ** (1.0 / b)
    )

    return time, rate


# ======================================================================
# TEST 1
# ======================================================================

def test_result_structure():
    print("=" * 70)
    print("DCA RESULT STRUCTURE")
    print("=" * 70)

    time, rate = build_synthetic_history()

    forecast_time = np.linspace(
        1000.0,
        3000.0,
        21,
    )

    engine = ForecastScenarioEngine(
        time_days=time,
        rate=rate,
    )

    result = engine.analyze(
        forecast_time_days=forecast_time,
    )

    assert isinstance(
        result,
        ForecastScenarioResult,
    )

    assert isinstance(
        result.best_dca_result,
        DCAFitResult,
    )

    assert set(
        result.dca_results.keys()
    ) == set(
        DCA_MODEL_NAMES
    )

    print("✓ ForecastScenarioResult returned.")
    print("✓ DCAFitResult returned.")
    print("✓ Three DCA models exposed.")
    print("✓ Result structure PASSED.")
    print()


# ======================================================================
# TEST 2
# ======================================================================

def test_all_three_models_successful():
    print("=" * 70)
    print("THREE ARPS MODELS")
    print("=" * 70)

    time, rate = build_synthetic_history()

    forecast_time = np.linspace(
        1000.0,
        3000.0,
        21,
    )

    engine = ForecastScenarioEngine(
        time_days=time,
        rate=rate,
    )

    results = engine.fit_dca_models(
        forecast_time_days=forecast_time,
    )

    for model_name in DCA_MODEL_NAMES:
        assert model_name in results
        assert results[
            model_name
        ].success

    print("✓ Exponential DCA fitted.")
    print("✓ Harmonic DCA fitted.")
    print("✓ Hyperbolic DCA fitted.")
    print("✓ All three DCA models PASSED.")
    print()


# ======================================================================
# TEST 3
# ======================================================================

def test_hyperbolic_recovery():
    print("=" * 70)
    print("HYPERBOLIC PARAMETER RECOVERY")
    print("=" * 70)

    time, rate = build_synthetic_history()

    forecast_time = np.linspace(
        1000.0,
        3000.0,
        21,
    )

    engine = ForecastScenarioEngine(
        time_days=time,
        rate=rate,
    )

    result = engine.fit_dca_models(
        forecast_time_days=forecast_time,
    )["Hyperbolic"]

    assert result.success

    assert np.isclose(
        result.qi,
        1000.0,
        rtol=0.01,
    )

    assert np.isclose(
        result.Di,
        0.001,
        rtol=0.05,
    )

    assert np.isclose(
        result.b,
        0.8,
        rtol=0.05,
    )

    assert result.rmse < 1.0e-6

    print(
        f"✓ qi recovered = {result.qi:.4f}"
    )

    print(
        f"✓ Di recovered = {result.Di:.8f}"
    )

    print(
        f"✓ b recovered = {result.b:.4f}"
    )

    print(
        f"✓ RMSE = {result.rmse:.6e}"
    )

    print("✓ Hyperbolic parameter recovery PASSED.")
    print()


# ======================================================================
# TEST 4
# ======================================================================

def test_forecast_rate_positive():
    print("=" * 70)
    print("PRODUCTION FORECAST VALIDATION")
    print("=" * 70)

    time, rate = build_synthetic_history()

    forecast_time = np.linspace(
        1000.0,
        3000.0,
        21,
    )

    engine = ForecastScenarioEngine(
        time_days=time,
        rate=rate,
    )

    results = engine.fit_dca_models(
        forecast_time_days=forecast_time,
    )

    for model_name, result in results.items():
        assert result.success

        assert np.all(
            result.forecast_rate > 0.0
        )

        assert np.all(
            np.isfinite(
                result.forecast_rate
            )
        )

        assert np.all(
            np.diff(
                result.forecast_rate
            ) < 0.0
        )

    print(
        "✓ All three models produce positive future rates."
    )

    print(
        "✓ All three models produce declining rates."
    )

    print("✓ Production forecast validation PASSED.")
    print()


# ======================================================================
# TEST 5
# ======================================================================

def test_cumulative_production_increases():
    print("=" * 70)
    print("CUMULATIVE PRODUCTION")
    print("=" * 70)

    time, rate = build_synthetic_history()

    forecast_time = np.linspace(
        1000.0,
        3000.0,
        21,
    )

    engine = ForecastScenarioEngine(
        time_days=time,
        rate=rate,
    )

    results = engine.fit_dca_models(
        forecast_time_days=forecast_time,
    )

    for result in results.values():
        assert result.success

        assert np.all(
            np.isfinite(
                result.forecast_cumulative
            )
        )

        assert np.all(
            np.diff(
                result.forecast_cumulative
            ) > 0.0
        )

        assert result.eur > 0.0

    print(
        "✓ Cumulative production increases for all models."
    )

    print(
        "✓ EUR is positive for all models."
    )

    print("✓ Cumulative-production validation PASSED.")
    print()


# ======================================================================
# TEST 6
# ======================================================================

def test_model_comparison():
    print("=" * 70)
    print("DCA MODEL COMPARISON")
    print("=" * 70)

    time, rate = build_synthetic_history()

    forecast_time = np.linspace(
        1000.0,
        3000.0,
        21,
    )

    engine = ForecastScenarioEngine(
        time_days=time,
        rate=rate,
    )

    result = engine.analyze(
        forecast_time_days=forecast_time,
    )

    table = result.comparison_table()

    assert len(table) == 3

    assert set(
        table["Model"]
    ) == set(
        DCA_MODEL_NAMES
    )

    assert table.iloc[0]["Rank"] == 1

    assert np.isfinite(
        table.iloc[0]["RMSE"]
    )

    assert result.best_dca_model in (
        DCA_MODEL_NAMES
    )

    print()
    print(table.to_string(index=False))
    print()

    print(
        f"✓ Best mathematical DCA fit: "
        f"{result.best_dca_model}"
    )

    print("✓ Model comparison PASSED.")
    print()


# ======================================================================
# TEST 7
# ======================================================================

def test_known_exponential():
    print("=" * 70)
    print("KNOWN EXPONENTIAL DATA")
    print("=" * 70)

    time = np.linspace(
        0.0,
        1000.0,
        41,
    )

    qi = 800.0
    Di = 0.0015

    rate = (
        qi
        * np.exp(
            -Di * time
        )
    )

    forecast_time = np.linspace(
        1000.0,
        3000.0,
        21,
    )

    engine = ForecastScenarioEngine(
        time_days=time,
        rate=rate,
    )

    result = engine.fit_dca_models(
        forecast_time_days=forecast_time
    )["Exponential"]

    assert result.success

    assert np.isclose(
        result.qi,
        qi,
        rtol=0.01,
    )

    assert np.isclose(
        result.Di,
        Di,
        rtol=0.02,
    )

    assert result.rmse < 1.0e-6

    print(
        f"✓ qi recovered = {result.qi:.4f}"
    )

    print(
        f"✓ Di recovered = {result.Di:.8f}"
    )

    print(
        f"✓ RMSE = {result.rmse:.6e}"
    )

    print("✓ Known exponential case PASSED.")
    print()


# ======================================================================
# TEST 8
# ======================================================================

def test_known_harmonic():
    print("=" * 70)
    print("KNOWN HARMONIC DATA")
    print("=" * 70)

    time = np.linspace(
        0.0,
        1000.0,
        41,
    )

    qi = 800.0
    Di = 0.0015

    rate = (
        qi
        / (
            1.0
            + Di * time
        )
    )

    forecast_time = np.linspace(
        1000.0,
        3000.0,
        21,
    )

    engine = ForecastScenarioEngine(
        time_days=time,
        rate=rate,
    )

    result = engine.fit_dca_models(
        forecast_time_days=forecast_time
    )["Harmonic"]

    assert result.success

    assert np.isclose(
        result.qi,
        qi,
        rtol=0.01,
    )

    assert np.isclose(
        result.Di,
        Di,
        rtol=0.02,
    )

    assert result.rmse < 1.0e-6

    print(
        f"✓ qi recovered = {result.qi:.4f}"
    )

    print(
        f"✓ Di recovered = {result.Di:.8f}"
    )

    print(
        f"✓ RMSE = {result.rmse:.6e}"
    )

    print("✓ Known harmonic case PASSED.")
    print()


# ======================================================================
# TEST 9
# ======================================================================

def test_input_validation():
    print("=" * 70)
    print("DCA INPUT VALIDATION")
    print("=" * 70)

    time = np.array(
        [0.0, 100.0, 200.0, 300.0, 400.0]
    )

    rate = np.array(
        [1000.0, 900.0, 800.0, 700.0, 600.0]
    )

    # Negative rate.
    invalid_rate = rate.copy()
    invalid_rate[2] = -1.0

    try:
        ForecastScenarioEngine(
            time_days=time,
            rate=invalid_rate,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Negative production rate was not rejected."
        )

    # Non-increasing time.
    invalid_time = time.copy()
    invalid_time[2] = invalid_time[1]

    try:
        ForecastScenarioEngine(
            time_days=invalid_time,
            rate=rate,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Non-increasing time was not rejected."
        )

    # Mismatched lengths.
    try:
        ForecastScenarioEngine(
            time_days=time,
            rate=rate[:-1],
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Mismatched input lengths were not rejected."
        )

    print(
        "✓ Negative rate rejected."
    )

    print(
        "✓ Non-increasing time rejected."
    )

    print(
        "✓ Mismatched lengths rejected."
    )

    print("✓ Input validation PASSED.")
    print()


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("PyMBAL - FORECAST SCENARIO / DCA VALIDATION")
    print("=" * 70)
    print()

    print(
        "This test validates the production-forecast scenario layer "
        "above the material-balance forecasting engine."
    )

    print()

    test_result_structure()
    test_all_three_models_successful()
    test_hyperbolic_recovery()
    test_forecast_rate_positive()
    test_cumulative_production_increases()
    test_model_comparison()
    test_known_exponential()
    test_known_harmonic()
    test_input_validation()

    print("=" * 70)
    print("✓ ALL FORECAST SCENARIO / DCA CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()