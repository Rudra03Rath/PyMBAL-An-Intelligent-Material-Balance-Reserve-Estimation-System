"""
PyMBAL - Reserves Estimation Layer Validation
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analysis.reserves import (
    ReserveResult,
    ReservesEstimator,
)


# ----------------------------------------------------------------------
# Lightweight integration fixtures
# ----------------------------------------------------------------------
# The reserves layer is tested independently from the already validated
# calibration, volumetric and DCA layers. These fixtures reproduce only
# their public result interfaces.

@dataclass(frozen=True)
class FakeVolumetricResult:
    best_ooip_stb: float


@dataclass(frozen=True)
class FakeDCAResult:
    model_name: str
    qi: float
    Di: float
    b: float
    eur: float
    success: bool
    error: str = None


@dataclass(frozen=True)
class FakeScenarioResult:
    dca_results: dict
    best_dca_model: str


# The production implementation type-checks the public result classes.
# Import the real classes and construct minimal valid instances for the
# test instead of using the lightweight dataclasses above.
from src.analysis.volumetrics import VolumetricResult, VolumetricCandidate
from src.forecasting.scenarios import ForecastScenarioResult, DCAFitResult


def make_volumetric_result(ooip=1_000_000.0):
    candidate = VolumetricCandidate(
        configuration="No Gas Cap + No Aquifer",
        aquifer_model=None,
        gas_cap_present=False,
        aquifer_present=False,
        success=True,
        competitive=True,
        ooip_stb=ooip,
        gas_cap_ratio_m=0.0,
        giip_scf=0.0,
        rmse_rb=0.0,
        r2=1.0,
    )

    return VolumetricResult(
        best_candidate=candidate,
        candidates=pd.DataFrame(),
        best_ooip_stb=ooip,
        best_giip_scf=0.0,
        competitive_ooip_min_stb=ooip,
        competitive_ooip_max_stb=ooip,
        competitive_giip_min_scf=None,
        competitive_giip_max_scf=None,
        identifiability_status="IDENTIFIED",
        gas_cap_status="Absent",
        initial_boi_rb_per_STB=1.2,
        initial_bgi_rb_per_scf=0.005,
    )


def make_dca_result(
    name,
    eur,
    qi=1000.0,
    Di=0.001,
    b=0.0,
):
    time = np.array([0.0, 100.0, 200.0])
    rate = np.array([1000.0, 900.0, 800.0])

    return DCAFitResult(
        model_name=name,
        qi=qi,
        Di=Di,
        b=b,
        rmse=1.0,
        mae=1.0,
        r2=0.99,
        historical_time=time,
        historical_rate=rate,
        fitted_rate=rate.copy(),
        forecast_time=np.array([300.0, 400.0]),
        forecast_rate=np.array([700.0, 600.0]),
        historical_cumulative=np.zeros_like(time),
        forecast_cumulative=np.array([300_000.0, eur]),
        eur=eur,
        success=True,
    )


def make_scenario(best="Hyperbolic"):
    results = {
        "Exponential": make_dca_result(
            "Exponential",
            eur=750_000.0,
            qi=950.0,
            Di=0.001,
            b=0.0,
        ),
        "Harmonic": make_dca_result(
            "Harmonic",
            eur=800_000.0,
            qi=980.0,
            Di=0.001,
            b=1.0,
        ),
        "Hyperbolic": make_dca_result(
            "Hyperbolic",
            eur=900_000.0,
            qi=1000.0,
            Di=0.001,
            b=0.8,
        ),
    }

    return ForecastScenarioResult(
        dca_results=results,
        pressure_scenarios=pd.DataFrame(),
        best_dca_model=best,
        best_dca_result=results[best],
        warnings=(),
    )


def make_estimator(
    ooip=1_000_000.0,
    best="Hyperbolic",
    time=None,
    rate=None,
):
    if time is None:
        time = np.array([0.0, 100.0, 200.0])

    if rate is None:
        rate = np.array([1000.0, 900.0, 800.0])

    return ReservesEstimator(
        volumetric_result=make_volumetric_result(ooip),
        forecast_scenario_result=make_scenario(best),
        historical_time_days=time,
        historical_rate=rate,
    )


def test_result_structure():
    result = make_estimator().estimate()

    assert isinstance(result, ReserveResult)
    assert isinstance(result.best_candidate.model_name, str)
    assert isinstance(result.candidates, pd.DataFrame)

    required = {
        "Model",
        "EUR_Raw_STB",
        "Remaining_Raw_STB",
        "Technical_EUR_STB",
        "Technical_Reserves_STB",
        "Historical_RF_percent",
        "Ultimate_RF_percent",
        "Selected",
    }

    assert required.issubset(result.candidates.columns)


def test_historical_cumulative_production():
    result = make_estimator().estimate()

    # Trapezoidal integration:
    # 0.5*(1000+900)*100 + 0.5*(900+800)*100 = 180,000 STB
    assert np.isclose(
        result.historical_np_stb,
        180_000.0,
        atol=1.0e-10,
    )


def test_remaining_reserves_equation():
    result = make_estimator().estimate()

    # Selected hyperbolic EUR = 900,000 STB.
    # Historical Np = 180,000 STB.
    # Remaining = 720,000 STB.
    assert np.isclose(
        result.best_remaining_reserves_stb,
        720_000.0,
        atol=1.0e-10,
    )

    assert np.isclose(
        result.best_eur_stb,
        900_000.0,
        atol=1.0e-10,
    )


def test_ooip_constraint():
    result = make_estimator(ooip=800_000.0).estimate()

    # Raw remaining = 900,000 - 180,000 = 720,000.
    # OOIP remaining = 800,000 - 180,000 = 620,000.
    assert np.isclose(
        result.best_remaining_reserves_stb,
        620_000.0,
        atol=1.0e-10,
    )

    assert result.is_ooip_constrained
    assert any(
        "exceeded the selected OOIP" in warning
        for warning in result.warnings
    )


def test_no_negative_remaining_reserves():
    result = make_estimator(ooip=200_000.0).estimate()

    assert result.best_remaining_reserves_stb == 20_000.0
    assert result.best_remaining_reserves_stb >= 0.0


def test_all_dca_models_exposed():
    result = make_estimator().estimate()

    assert set(result.candidates["Model"]) == {
        "Exponential",
        "Harmonic",
        "Hyperbolic",
    }

    assert len(result.candidates) == 3


def test_selected_dca_model_preserved():
    result = make_estimator(best="Harmonic").estimate()

    assert result.selected_dca_model == "Harmonic"
    assert result.best_candidate.model_name == "Harmonic"

    selected_rows = result.candidates[
        result.candidates["Selected"]
    ]

    assert len(selected_rows) == 1
    assert selected_rows.iloc[0]["Model"] == "Harmonic"


def test_recovery_factor_bounds():
    result = make_estimator().estimate()

    assert 0.0 <= result.historical_recovery_factor_percent <= 100.0
    assert 0.0 <= result.ultimate_recovery_factor_percent <= 100.0


def test_failed_dca_candidate_is_preserved():
    scenario = make_scenario()
    failed = DCAFitResult(
        model_name="Broken",
        qi=np.nan,
        Di=np.nan,
        b=np.nan,
        rmse=np.inf,
        mae=np.inf,
        r2=-np.inf,
        historical_time=np.array([0.0, 100.0]),
        historical_rate=np.array([1000.0, 900.0]),
        fitted_rate=np.array([np.nan, np.nan]),
        forecast_time=np.array([300.0]),
        forecast_rate=np.array([np.nan]),
        historical_cumulative=np.zeros(2),
        forecast_cumulative=np.array([np.nan]),
        eur=np.nan,
        success=False,
        error="synthetic failure",
    )

    scenario.dca_results["Broken"] = failed

    result = ReservesEstimator(
        volumetric_result=make_volumetric_result(),
        forecast_scenario_result=scenario,
        historical_time_days=np.array([0.0, 100.0, 200.0]),
        historical_rate=np.array([1000.0, 900.0, 800.0]),
    ).estimate()

    failed_row = result.candidates[
        result.candidates["Model"] == "Broken"
    ].iloc[0]

    assert not bool(failed_row["Success"])
    assert failed_row["Error"] == "synthetic failure"


def test_input_validation():
    with np.testing.assert_raises(ValueError):
        ReservesEstimator(
            volumetric_result=make_volumetric_result(),
            forecast_scenario_result=make_scenario(),
            historical_time_days=[0.0, 100.0, 50.0],
            historical_rate=[1000.0, 900.0, 800.0],
        )

    with np.testing.assert_raises(ValueError):
        ReservesEstimator(
            volumetric_result=make_volumetric_result(),
            forecast_scenario_result=make_scenario(),
            historical_time_days=[0.0, 100.0, 200.0],
            historical_rate=[1000.0, -1.0, 800.0],
        )


def main():
    print("=" * 70)
    print("PyMBAL - RESERVES ESTIMATION LAYER VALIDATION")
    print("=" * 70)
    print()
    print(
        "This test validates technical remaining-recoverable-volume "
        "interpretation above the already validated volumetric and "
        "DCA layers."
    )

    tests = [
        ("RESULT STRUCTURE", test_result_structure),
        ("HISTORICAL CUMULATIVE PRODUCTION", test_historical_cumulative_production),
        ("REMAINING RESERVES EQUATION", test_remaining_reserves_equation),
        ("OOIP CONSTRAINT", test_ooip_constraint),
        ("NO NEGATIVE REMAINING RESERVES", test_no_negative_remaining_reserves),
        ("ALL DCA MODELS EXPOSED", test_all_dca_models_exposed),
        ("SELECTED DCA MODEL", test_selected_dca_model_preserved),
        ("RECOVERY FACTOR BOUNDS", test_recovery_factor_bounds),
        ("FAILED DCA CANDIDATE", test_failed_dca_candidate_is_preserved),
        ("INPUT VALIDATION", test_input_validation),
    ]

    for title, test in tests:
        print("=" * 70)
        print(title)
        print("=" * 70)
        test()
        print("✓ PASSED")
        print()

    print("=" * 70)
    print("✓ ALL RESERVES ESTIMATION CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
