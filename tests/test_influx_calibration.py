"""
PyMBAL - WATER INFLUX CALIBRATION VALIDATION

Validates the calibration engine against synthetic reservoirs
generated with known aquifer parameters.

Models tested
-------------
1. Schilthuis
2. Fetkovich
3. Van Everdingen-Hurst

The synthetic We history is used here as the calibration target
ONLY for validating the calibration engine.

In the final application, We will be inferred from reservoir
material balance rather than directly supplied.
"""

import os
import sys

import pandas as pd


# ============================================================================
# PROJECT ROOT
# ============================================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# IMPORTS
# ============================================================================

from src.influx.calibration import (
    WaterInfluxCalibrator,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

SCHILTHUIS_FILE = (
    "data/sample_data/"
    "synthetic_water_influx_reservoir.csv"
)

FETKOVICH_FILE = (
    "data/sample_data/"
    "synthetic_fetkovich_reservoir.csv"
)

VEH_FILE = (
    "data/sample_data/"
    "synthetic_van_everdingen_hurst/"
    "synthetic_van_everdingen_hurst_reservoir.csv"
)


# ============================================================================
# HELPERS
# ============================================================================

def load_dataset(path: str) -> pd.DataFrame:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Synthetic dataset not found:\n{path}"
        )

    df = pd.read_csv(path)

    required = [
        "Days",
        "Pressure_psia",
        "We_rb",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {missing}"
        )

    return df


def create_calibrator(
    df: pd.DataFrame,
) -> WaterInfluxCalibrator:

    return WaterInfluxCalibrator(
        time_days=df["Days"].to_numpy(),
        pressure_psia=df["Pressure_psia"].to_numpy(),
        target_we_rb=df["We_rb"].to_numpy(),
    )


def print_result(result):

    print(
        f"Model : {result.model_name}"
    )

    print(
        f"Success : {result.success}"
    )

    print(
        f"RMSE : {result.rmse_rb:.6f} rb"
    )

    print(
        f"MAE  : {result.mae_rb:.6f} rb"
    )

    print(
        f"R²   : {result.r2:.9f}"
    )

    print(
        "Parameters:"
    )

    for key, value in result.parameters.items():

        print(
            f"  {key:<40} : {value:.10g}"
        )

    print()


# ============================================================================
# SCHILTHUIS TEST
# ============================================================================

def test_schilthuis_calibration():

    print()
    print("=" * 70)
    print("SCHILTHUIS CALIBRATION")
    print("=" * 70)

    df = load_dataset(
        SCHILTHUIS_FILE
    )

    calibrator = create_calibrator(df)

    result = calibrator.calibrate_schilthuis(
        aquifer_pressure_psia=4000.0,
        c_bounds=(1e-4, 100.0),
    )

    print_result(result)

    assert result.success, (
        "Schilthuis calibration failed."
    )

    calibrated_c = result.parameters[
        "C_rb_per_day_psi"
    ]

    expected_c = 2.0

    relative_error = abs(
        calibrated_c - expected_c
    ) / expected_c

    print(
        f"Expected C      : {expected_c:.6f}"
    )

    print(
        f"Calibrated C    : {calibrated_c:.6f}"
    )

    print(
        f"Relative error  : {relative_error:.6e}"
    )

    assert relative_error < 1e-4, (
        "Schilthuis calibration did not recover "
        "the known influx constant."
    )

    assert result.rmse_rb < 1e-3, (
        "Schilthuis calibration RMSE is too large."
    )

    print(
        "✓ Schilthuis parameter recovery PASSED."
    )

    return result


# ============================================================================
# FETKOVICH TEST
# ============================================================================

def test_fetkovich_calibration():

    print()
    print("=" * 70)
    print("FETKOVICH CALIBRATION")
    print("=" * 70)

    df = load_dataset(
        FETKOVICH_FILE
    )

    calibrator = create_calibrator(df)

    result = calibrator.calibrate_fetkovich(
        initial_aquifer_pressure_psia=4000.0,
        total_compressibility_psi_inv=2.0e-5,

        productivity_index_bounds=(
            1.0,
            1000.0,
        ),

        aquifer_volume_bounds=(
            1.0e5,
            1.0e8,
        ),
    )

    print_result(result)

    assert result.success, (
        "Fetkovich calibration failed."
    )

    calibrated_j = result.parameters[
        "productivity_index_rb_per_day_psi"
    ]

    calibrated_vi = result.parameters[
        "initial_water_volume_rb"
    ]

    expected_j = 100.0
    expected_vi = 5_000_000.0

    j_relative_error = abs(
        calibrated_j - expected_j
    ) / expected_j

    vi_relative_error = abs(
        calibrated_vi - expected_vi
    ) / expected_vi

    print(
        f"Expected J              : {expected_j:.6f}"
    )

    print(
        f"Calibrated J            : {calibrated_j:.6f}"
    )

    print(
        f"J relative error        : {j_relative_error:.6e}"
    )

    print()

    print(
        f"Expected Vi             : {expected_vi:.2f} rb"
    )

    print(
        f"Calibrated Vi           : {calibrated_vi:.2f} rb"
    )

    print(
        f"Vi relative error       : {vi_relative_error:.6e}"
    )

    assert j_relative_error < 1e-3, (
        "Fetkovich calibration did not recover "
        "the known productivity index."
    )

    assert vi_relative_error < 1e-3, (
        "Fetkovich calibration did not recover "
        "the known aquifer volume."
    )

    assert result.rmse_rb < 1e-3, (
        "Fetkovich calibration RMSE is too large."
    )

    print(
        "✓ Fetkovich parameter recovery PASSED."
    )

    return result


# ============================================================================
# VAN EVERDINGEN-HURST TEST
# ============================================================================

def test_van_everdingen_hurst_calibration():

    print()
    print("=" * 70)
    print("VAN EVERDINGEN-HURST CALIBRATION")
    print("=" * 70)

    df = load_dataset(
        VEH_FILE
    )

    calibrator = create_calibrator(df)

    result = (
        calibrator
        .calibrate_van_everdingen_hurst(
            aquifer_porosity=0.20,
            aquifer_thickness_ft=100.0,
            total_compressibility_psi_inv=2.0e-5,
            water_viscosity_cp=0.50,
            reservoir_radius_ft=1000.0,
            encroachment_angle_deg=360.0,

            permeability_bounds=(
                1.0,
                1000.0,
            ),
        )
    )

    print_result(result)

    assert result.success, (
        "Van Everdingen-Hurst calibration failed."
    )

    calibrated_k = result.parameters[
        "aquifer_permeability_md"
    ]

    expected_k = 100.0

    relative_error = abs(
        calibrated_k - expected_k
    ) / expected_k

    print(
        f"Expected k      : {expected_k:.6f} md"
    )

    print(
        f"Calibrated k    : {calibrated_k:.6f} md"
    )

    print(
        f"Relative error  : {relative_error:.6e}"
    )

    assert relative_error < 1e-3, (
        "V-E-H calibration did not recover "
        "the known permeability."
    )

    assert result.rmse_rb < 1e-3, (
        "V-E-H calibration RMSE is too large."
    )

    print(
        "✓ Van Everdingen-Hurst parameter "
        "recovery PASSED."
    )

    return result


# ============================================================================
# MODEL COMPARISON TEST
# ============================================================================

def test_model_comparison(
    schilthuis_result,
    fetkovich_result,
    veh_result,
):

    print()
    print("=" * 70)
    print("AQUIFER MODEL COMPARISON")
    print("=" * 70)

    results = [
        schilthuis_result,
        fetkovich_result,
        veh_result,
    ]

    calibrator = None

    comparison = pd.DataFrame([
        {
            "Model": result.model_name,
            "Success": result.success,
            "RMSE_rb": result.rmse_rb,
            "MAE_rb": result.mae_rb,
            "R2": result.r2,
        }
        for result in results
    ])

    comparison = comparison.sort_values(
        by="RMSE_rb",
        ascending=True,
    ).reset_index(drop=True)

    print()
    print(comparison.to_string(index=False))

    print()

    best_model = min(
        results,
        key=lambda result: result.rmse_rb,
    )

    print(
        f"Best-fit model : {best_model.model_name}"
    )

    print(
        f"Best RMSE      : {best_model.rmse_rb:.6f} rb"
    )

    assert best_model.success, (
        "Selected best model is unsuccessful."
    )

    print(
        "✓ Model comparison check PASSED."
    )


# ============================================================================
# INPUT VALIDATION TEST
# ============================================================================

def test_calibration_input_validation():

    print()
    print("=" * 70)
    print("CALIBRATION INPUT VALIDATION")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Mismatched lengths
    # ------------------------------------------------------------------------

    try:

        WaterInfluxCalibrator(
            time_days=[0, 10, 20],
            pressure_psia=[4000, 3900],
            target_we_rb=[0, 1000, 2000],
        )

        raise AssertionError(
            "Mismatched lengths were not rejected."
        )

    except ValueError:

        print(
            "✓ Mismatched input lengths rejected."
        )

    # ------------------------------------------------------------------------
    # Non-zero initial time
    # ------------------------------------------------------------------------

    try:

        WaterInfluxCalibrator(
            time_days=[1, 10, 20],
            pressure_psia=[4000, 3900, 3800],
            target_we_rb=[0, 1000, 2000],
        )

        raise AssertionError(
            "Non-zero initial time was not rejected."
        )

    except ValueError:

        print(
            "✓ Non-zero initial time rejected."
        )

    # ------------------------------------------------------------------------
    # Non-monotonic time
    # ------------------------------------------------------------------------

    try:

        WaterInfluxCalibrator(
            time_days=[0, 20, 10],
            pressure_psia=[4000, 3900, 3800],
            target_we_rb=[0, 1000, 2000],
        )

        raise AssertionError(
            "Non-monotonic time was not rejected."
        )

    except ValueError:

        print(
            "✓ Non-monotonic time rejected."
        )

    # ------------------------------------------------------------------------
    # Negative We
    # ------------------------------------------------------------------------

    try:

        WaterInfluxCalibrator(
            time_days=[0, 10, 20],
            pressure_psia=[4000, 3900, 3800],
            target_we_rb=[0, -1000, 2000],
        )

        raise AssertionError(
            "Negative We was not rejected."
        )

    except ValueError:

        print(
            "✓ Negative water influx rejected."
        )

    # ------------------------------------------------------------------------
    # Non-monotonic We
    # ------------------------------------------------------------------------

    try:

        WaterInfluxCalibrator(
            time_days=[0, 10, 20],
            pressure_psia=[4000, 3900, 3800],
            target_we_rb=[0, 2000, 1000],
        )

        raise AssertionError(
            "Non-monotonic We was not rejected."
        )

    except ValueError:

        print(
            "✓ Non-monotonic water influx rejected."
        )

    print(
        "✓ Calibration input validation PASSED."
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 70)
    print(
        "PyMBAL - WATER INFLUX CALIBRATION VALIDATION"
    )
    print("=" * 70)

    print()
    print(
        "NOTE:"
    )
    print(
        "Synthetic We is used here only to validate "
        "parameter-recovery capability."
    )

    print(
        "Final application calibration will infer "
        "required We from material balance."
    )

    # ------------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------------

    test_calibration_input_validation()

    # ------------------------------------------------------------------------
    # Individual model calibration
    # ------------------------------------------------------------------------

    schilthuis_result = (
        test_schilthuis_calibration()
    )

    fetkovich_result = (
        test_fetkovich_calibration()
    )

    veh_result = (
        test_van_everdingen_hurst_calibration()
    )

    # ------------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------------

    test_model_comparison(
        schilthuis_result,
        fetkovich_result,
        veh_result,
    )

    # ------------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "✓ ALL WATER INFLUX CALIBRATION CHECKS PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()