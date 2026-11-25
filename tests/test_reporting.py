"""
PyMBAL - REPORTING LAYER VALIDATION

This test validates only the aggregation/reporting layer.

The analytical result objects are constructed directly from their public
dataclasses. No reservoir calibration, DDI calculation, DCA fitting or
reserve calculation is performed here.
"""

import numpy as np
import pandas as pd

from src.analysis.model_selection import (
    IdentifiabilityAssessment,
    ModelSelectionConstraints,
    ModelSelectionResult,
)
from src.analysis.drive_mechanism import DriveMechanismResult
from src.analysis.volumetrics import (
    VolumetricCandidate,
    VolumetricResult,
)
from src.analysis.reserves import (
    ReserveCandidate,
    ReserveResult,
)
from src.forecasting.scenarios import (
    DCAFitResult,
    ForecastScenarioResult,
)
from src.reporting.report import (
    PyMBALReport,
    PyMBALReportBuilder,
    build_report,
)


# ======================================================================
# MOCK VALIDATED RESULT OBJECTS
# ======================================================================

def build_model_selection():
    comparison = pd.DataFrame(
        {
            "Configuration": [
                "Gas Cap + Aquifer",
                "No Gas Cap + No Aquifer",
                "No Gas Cap + Aquifer",
            ],
            "Aquifer_Model": [
                "Schilthuis Steady-State",
                "None",
                "Fetkovich Finite Aquifer",
            ],
            "Gas_Cap": [True, False, False],
            "Aquifer": [True, False, True],
            "Success": [True, True, True],
            "RMSE_rb": [1.0, 1.2, 1.3],
            "R2": [0.999, 0.998, 0.997],
            "OOIP_STB": [9.8e6, 10.0e6, 9.9e6],
            "m": [0.05, 0.0, 0.0],
            "Competitive": [True, True, False],
            "Rank": [1, 2, 3],
        }
    )

    possible = comparison.copy()

    class SelectionStub:
        pass


    selection = SelectionStub()

    selection.best_case = "Gas Cap + Aquifer"
    selection.best_result = {"success": True}
    selection.best_rmse_rb = 1.0
    selection.best_r2 = 0.999
    selection.best_ooip_stb = 9.8e6
    selection.best_gas_cap_ratio_m = 0.05
    selection.best_aquifer_model = "Schilthuis Steady-State"
    selection.comparison = comparison
    selection.all_results = comparison
    selection.possible_cases = possible
    selection.competitive_results = comparison[
        comparison["Competitive"]
    ].copy()

    ident = IdentifiabilityAssessment(
        status="AMBIGUOUS",
        message="Multiple competitive interpretations remain.",
        competitive_count=2,
        best_case="Gas Cap + Aquifer",
        competitive_cases=(
            "Gas Cap + Aquifer [Schilthuis Steady-State]",
            "No Gas Cap + No Aquifer",
        ),
    )

    return ModelSelectionResult(
        selection=selection,
        constraints=ModelSelectionConstraints(),
        identifiability=ident,
    )


def build_drive():
    ddi = pd.DataFrame(
        {
            "Time_days": [100.0, 200.0, 300.0],
            "Pressure_psia": [4000.0, 3800.0, 3600.0],
            "Solution-Gas Expansion": [60.0, 58.0, 55.0],
            "Gas-Cap Expansion": [10.0, 12.0, 15.0],
            "Aquifer Influx": [15.0, 17.0, 20.0],
            "Rock + Connate-Water Expansion": [15.0, 13.0, 10.0],
        }
    )

    return DriveMechanismResult(
        ddi_history=ddi,
        final_ddi={
            "Solution-Gas Drive": 55.0,
            "Gas-Cap Expansion": 15.0,
            "Aquifer Influx": 20.0,
            "Rock + Connate-Water Expansion": 10.0,
        },
        dominant_mechanism="Solution-Gas Drive",
        overall_mechanism="Solution-Gas Drive",
        confidence="Moderate",
        gas_cap_status="Present",
        aquifer_status="Present",
        aquifer_model="Schilthuis Steady-State",
        identifiability_status="AMBIGUOUS",
        competitive_drive_interpretations=(
            "Gas Cap + Aquifer [Schilthuis Steady-State]",
            "No Gas Cap + No Aquifer",
        ),
        evidence=("Selected interpretation is conditional.",),
    )


def build_volumetrics():
    candidate = VolumetricCandidate(
        configuration="Gas Cap + Aquifer",
        aquifer_model="Schilthuis Steady-State",
        gas_cap_present=True,
        aquifer_present=True,
        success=True,
        competitive=True,
        ooip_stb=9.8e6,
        gas_cap_ratio_m=0.05,
        giip_scf=5.0e9,
        rmse_rb=1.0,
        r2=0.999,
    )

    table = pd.DataFrame(
        {
            "Configuration": [
                "Gas Cap + Aquifer",
                "No Gas Cap + No Aquifer",
            ],
            "Aquifer_Model": [
                "Schilthuis Steady-State",
                None,
            ],
            "OOIP_STB": [9.8e6, 10.0e6],
            "GIIP_scf": [5.0e9, 0.0],
            "GIIP_Bscf": [5.0, 0.0],
            "Competitive": [True, True],
            "Success": [True, True],
        }
    )

    return VolumetricResult(
        best_candidate=candidate,
        candidates=table,
        best_ooip_stb=9.8e6,
        best_giip_scf=5.0e9,
        competitive_ooip_min_stb=9.8e6,
        competitive_ooip_max_stb=10.0e6,
        competitive_giip_min_scf=5.0e9,
        competitive_giip_max_scf=5.0e9,
        identifiability_status="AMBIGUOUS",
        gas_cap_status="Present",
        initial_boi_rb_per_STB=1.2,
        initial_bgi_rb_per_scf=0.005,
    )


def build_forecasting():
    historical_time = np.array([0.0, 100.0, 200.0, 300.0])
    historical_rate = np.array([1000.0, 900.0, 820.0, 750.0])
    forecast_time = np.array([300.0, 400.0, 500.0, 600.0])
    forecast_rate = np.array([750.0, 690.0, 640.0, 590.0])

    def dca(name, offset):
        return DCAFitResult(
            model_name=name,
            qi=1000.0 + offset,
            Di=0.001,
            b=0.8 if name == "Hyperbolic" else (
                1.0 if name == "Harmonic" else 0.0
            ),
            rmse=1.0 + abs(offset),
            mae=0.5 + abs(offset),
            r2=0.999,
            historical_time=historical_time.copy(),
            historical_rate=historical_rate.copy(),
            fitted_rate=historical_rate + offset,
            forecast_time=forecast_time.copy(),
            forecast_rate=forecast_rate.copy(),
            historical_cumulative=np.zeros(4),
            forecast_cumulative=np.array(
                [250000.0, 320000.0, 380000.0, 430000.0]
            ),
            eur=430000.0,
            success=True,
        )

    results = {
        "Exponential": dca("Exponential", 2.0),
        "Harmonic": dca("Harmonic", 1.0),
        "Hyperbolic": dca("Hyperbolic", 0.0),
    }

    return ForecastScenarioResult(
        dca_results=results,
        pressure_scenarios=pd.DataFrame(
            {
                "Time_days": [300.0, 400.0, 500.0],
                "Pressure_psia": [3600.0, 3400.0, 3200.0],
            }
        ),
        best_dca_model="Hyperbolic",
        best_dca_result=results["Hyperbolic"],
        warnings=("DCA selection is mathematical.",),
    )


def build_reserves():
    candidate = ReserveCandidate(
        model_name="Hyperbolic",
        success=True,
        qi=1000.0,
        Di=0.001,
        b=0.8,
        historical_np_stb=100000.0,
        eur_raw_stb=430000.0,
        remaining_raw_stb=330000.0,
        technical_eur_stb=330000.0,
        technical_reserves_stb=230000.0,
        historical_recovery_factor_percent=1.02,
        ultimate_recovery_factor_percent=3.37,
        constrained_by_ooip=False,
        error=None,
    )

    table = pd.DataFrame(
        {
            "Model": [
                "Exponential",
                "Harmonic",
                "Hyperbolic",
            ],
            "Technical_Reserves_STB": [
                210000.0,
                220000.0,
                230000.0,
            ],
            "Selected": [
                False,
                False,
                True,
            ],
        }
    )

    return ReserveResult(
        best_candidate=candidate,
        candidates=table,
        selected_dca_model="Hyperbolic",
        historical_np_stb=100000.0,
        best_eur_stb=330000.0,
        best_remaining_reserves_stb=230000.0,
        best_ooip_stb=9.8e6,
        historical_recovery_factor_percent=1.02,
        ultimate_recovery_factor_percent=3.37,
        warnings=("Technical estimate only.",),
    )


def build_all():
    return (
        build_model_selection(),
        build_drive(),
        build_volumetrics(),
        build_forecasting(),
        build_reserves(),
    )


# ======================================================================
# TESTS
# ======================================================================

def test_report_structure():
    report = build_report(*build_all())

    assert isinstance(report, PyMBALReport)
    assert report.selected_configuration == "Gas Cap + Aquifer"
    assert report.selected_aquifer_model == "Schilthuis Steady-State"
    assert report.selected_dca_model == "Hyperbolic"
    assert report.identifiability_status == "AMBIGUOUS"

    print("✓ PyMBALReport returned.")
    print("✓ All five analytical result layers attached.")
    print("✓ Report structure PASSED.")


def test_executive_summary():
    report = build_report(*build_all())
    summary = report.executive_summary()

    required = {
        "selected_configuration",
        "selected_aquifer_model",
        "identifiability_status",
        "dominant_drive",
        "overall_drive",
        "ooip_stb",
        "giip_scf",
        "selected_dca_model",
        "eur_stb",
        "remaining_reserves_stb",
    }

    assert required.issubset(summary.keys())
    assert summary["ooip_stb"] == 9.8e6
    assert summary["giip_scf"] == 5.0e9

    print("✓ Executive summary exposes key engineering outputs.")
    print("✓ Executive summary PASSED.")


def test_candidate_exposure():
    report = build_report(*build_all())

    possible = report.possible_candidates()
    competitive = report.competitive_candidates()

    assert len(possible) == 3
    assert len(competitive) == 2

    assert "Configuration" in possible.columns
    assert "Competitive" in competitive.columns

    print("✓ Possible candidates exposed.")
    print("✓ Competitive candidates exposed.")
    print("✓ Candidate exposure PASSED.")


def test_ddi_and_volumetric_exposure():
    report = build_report(*build_all())

    ddi = report.ddi_history()
    final_ddi = report.final_ddi()
    volumes = report.volumetric_candidates()

    assert len(ddi) == 3
    assert len(final_ddi) == 4
    assert "OOIP_STB" in volumes.columns
    assert "GIIP_Bscf" in volumes.columns

    assert np.isclose(
        sum(final_ddi.values()),
        100.0,
    )

    print("✓ DDI history exposed.")
    print("✓ Final DDI percentages exposed.")
    print("✓ OOIP/GIIP candidate table exposed.")
    print("✓ DDI/volumetric exposure PASSED.")


def test_forecast_and_reserves_exposure():
    report = build_report(*build_all())

    dca = report.dca_comparison()
    reserves = report.reserve_candidates()

    assert len(dca) == 3
    assert len(reserves) == 3
    assert report.forecasting.best_dca_model == "Hyperbolic"
    assert report.reserves.selected_dca_model == "Hyperbolic"

    print("✓ Three DCA models exposed.")
    print("✓ Reserves candidates exposed.")
    print("✓ Forecast/reserves exposure PASSED.")


def test_warning_consolidation():
    report = build_report(
        *build_all(),
        warnings=(
            "Application warning.",
            "Application warning.",
        ),
        notes=("Generated for engineering review.",),
    )

    warnings = report.all_warnings()
    evidence = report.all_evidence()

    assert warnings.count("Application warning.") == 1
    assert any(
        "DCA selection" in warning
        for warning in warnings
    )
    assert len(evidence) == 1
    assert len(report.notes) == 1

    print("✓ Duplicate warnings removed.")
    print("✓ Layer warnings preserved.")
    print("✓ Evidence and notes exposed.")
    print("✓ Warning consolidation PASSED.")


def test_dict_serialization():
    report = build_report(*build_all())
    data = report.to_dict()

    required = {
        "executive_summary",
        "model_selection",
        "drive_mechanism",
        "volumetrics",
        "forecasting",
        "reserves",
        "possible_candidates",
        "competitive_candidates",
        "volumetric_candidates",
        "dca_comparison",
        "reserve_candidates",
        "ddi_history",
        "final_ddi",
        "warnings",
        "evidence",
        "notes",
    }

    assert required.issubset(data.keys())
    assert isinstance(
        data["competitive_candidates"],
        list,
    )
    assert isinstance(
        data["dca_comparison"],
        list,
    )

    print("✓ Report converted to structured dictionary.")
    print("✓ Tables serialized as records.")
    print("✓ Dictionary serialization PASSED.")


def test_builder_type_validation():
    model, drive, volume, forecast, reserves = build_all()

    try:
        PyMBALReportBuilder(
            model_selection="invalid",
            drive_mechanism=drive,
            volumetrics=volume,
            forecasting=forecast,
            reserves=reserves,
        )
    except TypeError:
        pass
    else:
        raise AssertionError(
            "Invalid model-selection result was not rejected."
        )

    print("✓ Invalid component types rejected.")
    print("✓ Type validation PASSED.")


def main():
    print("=" * 70)
    print("PyMBAL - REPORTING LAYER VALIDATION")
    print("=" * 70)
    print()
    print(
        "This test validates the read-only report aggregation layer "
        "above the already validated analytical components."
    )
    print()

    print("=" * 70)
    print("REPORT STRUCTURE")
    print("=" * 70)
    test_report_structure()
    print()

    print("=" * 70)
    print("EXECUTIVE SUMMARY")
    print("=" * 70)
    test_executive_summary()
    print()

    print("=" * 70)
    print("CANDIDATE EXPOSURE")
    print("=" * 70)
    test_candidate_exposure()
    print()

    print("=" * 70)
    print("DDI / VOLUMETRIC EXPOSURE")
    print("=" * 70)
    test_ddi_and_volumetric_exposure()
    print()

    print("=" * 70)
    print("FORECAST / RESERVES EXPOSURE")
    print("=" * 70)
    test_forecast_and_reserves_exposure()
    print()

    print("=" * 70)
    print("WARNING / EVIDENCE HANDLING")
    print("=" * 70)
    test_warning_consolidation()
    print()

    print("=" * 70)
    print("STRUCTURED SERIALIZATION")
    print("=" * 70)
    test_dict_serialization()
    print()

    print("=" * 70)
    print("INPUT / TYPE VALIDATION")
    print("=" * 70)
    test_builder_type_validation()
    print()

    print("=" * 70)
    print("✓ ALL REPORTING LAYER CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
