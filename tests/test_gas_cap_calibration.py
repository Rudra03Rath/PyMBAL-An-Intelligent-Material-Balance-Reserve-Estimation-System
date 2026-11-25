"""
PyMBAL - Gas-Cap Ratio Calibration Validation

Purpose:
    Validate recovery of gas-cap ratio m and OOIP from
    material-balance history.

The synthetic dataset is generated from:

    F - We = N(Eo + mEg + Efw)

The true m is NOT supplied to the calibration engine.
"""

import numpy as np

from src.core.gas_cap_calibration import (
    GasCapCalibrator,
    calibrate_gas_cap_ratio,
)


# ======================================================================
# GROUND TRUTH
# ======================================================================

TRUE_N = 10_000_000.0
TRUE_M = 0.80


# ======================================================================
# SYNTHETIC HISTORY
# ======================================================================

def build_synthetic_history():

    n = 30

    # --------------------------------------------------------------
    # Independent synthetic expansion histories
    # --------------------------------------------------------------

    x = np.linspace(0.0, 1.0, n)

    Eo = (
        0.120
        * x
        ** 1.15
    )

    Eg = (
        0.250
        * x
        ** 1.75
    )

    Efw = (
        0.010
        * x
        ** 1.35
    )

    # --------------------------------------------------------------
    # Synthetic water influx
    # --------------------------------------------------------------

    We = (
        1_000_000.0
        * x
        ** 1.25
    )

    # --------------------------------------------------------------
    # Generate F from the true material balance
    # --------------------------------------------------------------

    F = (
        TRUE_N
        * (
            Eo
            + TRUE_M * Eg
            + Efw
        )
        + We
    )

    return F, Eo, Eg, Efw, We


# ======================================================================
# INPUT VALIDATION
# ======================================================================

def test_input_validation():

    print("=" * 70)
    print("GAS-CAP CALIBRATION INPUT VALIDATION")
    print("=" * 70)

    F, Eo, Eg, Efw, We = build_synthetic_history()

    # --------------------------------------------------------------
    # Mismatched lengths
    # --------------------------------------------------------------

    try:
        GasCapCalibrator(
            F[:-1],
            Eo,
            Eg,
            Efw,
            We,
        )

        raise AssertionError(
            "Mismatched input lengths were not rejected."
        )

    except ValueError:
        print("✓ Mismatched input lengths rejected.")

    # --------------------------------------------------------------
    # Negative F
    # --------------------------------------------------------------

    bad_F = F.copy()
    bad_F[5] = -1.0

    try:
        GasCapCalibrator(
            bad_F,
            Eo,
            Eg,
            Efw,
            We,
        )

        raise AssertionError(
            "Negative F was not rejected."
        )

    except ValueError:
        print("✓ Negative underground withdrawal rejected.")

    # --------------------------------------------------------------
    # Negative We
    # --------------------------------------------------------------

    bad_We = We.copy()
    bad_We[5] = -1.0

    try:
        GasCapCalibrator(
            F,
            Eo,
            Eg,
            Efw,
            bad_We,
        )

        raise AssertionError(
            "Negative We was not rejected."
        )

    except ValueError:
        print("✓ Negative water influx rejected.")

    # --------------------------------------------------------------
    # Non-monotonic We
    # --------------------------------------------------------------

    bad_We = We.copy()
    bad_We[10] = bad_We[9] - 1000.0

    try:
        GasCapCalibrator(
            F,
            Eo,
            Eg,
            Efw,
            bad_We,
        )

        raise AssertionError(
            "Non-monotonic We was not rejected."
        )

    except ValueError:
        print("✓ Non-monotonic water influx rejected.")

    print()
    print("✓ Gas-cap calibration input validation PASSED.")


# ======================================================================
# BASIC CALIBRATION
# ======================================================================

def test_gas_cap_calibration():

    print("=" * 70)
    print("GAS-CAP RATIO CALIBRATION")
    print("=" * 70)

    F, Eo, Eg, Efw, We = build_synthetic_history()

    calibrator = GasCapCalibrator(
        F_rb=F,
        Eo_rb_per_STB=Eo,
        Eg_rb_per_STB=Eg,
        Efw_rb_per_STB=Efw,
        We_rb=We,
    )

    result = calibrator.calibrate(
        m_bounds=(0.0, 2.0)
    )

    print()
    print("Model")
    print("-" * 70)
    print(f"Name       : {result.model_name}")
    print(f"Success    : {result.success}")
    print(f"RMSE       : {result.rmse_rb:.12f} rb")
    print(f"MAE        : {result.mae_rb:.12f} rb")
    print(f"R²         : {result.r2:.12f}")

    print()
    print("Estimated OOIP")
    print("-" * 70)
    print(f"N           : {result.estimated_ooip_stb:.6f} STB")
    print(
        f"N           : "
        f"{result.estimated_ooip_stb / 1e6:.6f} MMSTB"
    )

    print()
    print("Calibrated Gas-Cap Parameter")
    print("-" * 70)
    print(
        f"m           : "
        f"{result.gas_cap_ratio_m:.12f}"
    )

    assert result.success

    assert abs(
        result.gas_cap_ratio_m - TRUE_M
    ) / TRUE_M < 1e-6

    assert abs(
        result.estimated_ooip_stb - TRUE_N
    ) / TRUE_N < 1e-6

    assert result.rmse_rb < 1e-3

    assert result.r2 > 0.999999

    print()
    print(
        f"True m          : {TRUE_M:.12f}"
    )

    print(
        f"Calibrated m    : "
        f"{result.gas_cap_ratio_m:.12f}"
    )

    print(
        f"Relative error  : "
        f"{abs(result.gas_cap_ratio_m - TRUE_M) / TRUE_M:.6e}"
    )

    print()
    print(
        f"True OOIP       : {TRUE_N:.6f} STB"
    )

    print(
        f"Calibrated OOIP : "
        f"{result.estimated_ooip_stb:.6f} STB"
    )

    print(
        f"Relative error  : "
        f"{abs(result.estimated_ooip_stb - TRUE_N) / TRUE_N:.6e}"
    )

    print()
    print("✓ Gas-cap ratio recovery PASSED.")
    print("✓ OOIP recovery PASSED.")
    print("✓ Material-balance fit PASSED.")


# ======================================================================
# m = 0 CASE
# ======================================================================

def test_zero_gas_cap():

    print()
    print("=" * 70)
    print("ZERO GAS-CAP CASE")
    print("=" * 70)

    n = 30

    N = TRUE_N
    m = 0.0

    x = np.linspace(0.0, 1.0, n)

    Eo = (
        0.10
        * x
        ** 1.15
    )

    Eg = (
        0.25
        * x
        ** 1.75
    )

    Efw = (
        0.01
        * x
        ** 1.35
    )

    We = (
        500_000.0
        * x
        ** 1.25
    )

    F = (
        N
        * (
            Eo
            + m * Eg
            + Efw
        )
        + We
    )

    result = calibrate_gas_cap_ratio(
        F_rb=F,
        Eo_rb_per_STB=Eo,
        Eg_rb_per_STB=Eg,
        Efw_rb_per_STB=Efw,
        We_rb=We,
        m_bounds=(0.0, 2.0),
    )

    print(
        f"True m          : {m:.12f}"
    )

    print(
        f"Calibrated m    : "
        f"{result.gas_cap_ratio_m:.12f}"
    )

    assert result.success

    assert abs(
        result.gas_cap_ratio_m - m
    ) < 1e-8

    print("✓ m = 0 gas-cap case PASSED.")


# ======================================================================
# RESULT STRUCTURE
# ======================================================================

def test_result_structure():

    print()
    print("=" * 70)
    print("RESULT STRUCTURE VALIDATION")
    print("=" * 70)

    F, Eo, Eg, Efw, We = build_synthetic_history()

    result = calibrate_gas_cap_ratio(
        F,
        Eo,
        Eg,
        Efw,
        We,
    )

    required_attributes = [
        "success",
        "model_name",
        "gas_cap_ratio_m",
        "estimated_ooip_stb",
        "rmse_rb",
        "mae_rb",
        "r2",
        "residuals_rb",
        "apparent_ooip_stb",
        "message",
    ]

    for attribute in required_attributes:
        assert hasattr(result, attribute), (
            f"Missing result attribute: {attribute}"
        )

    assert len(result.residuals_rb) == len(F)
    assert len(result.apparent_ooip_stb) == len(F)

    print("✓ Result structure validation PASSED.")


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print("PyMBAL - GAS-CAP RATIO CALIBRATION VALIDATION")
    print("=" * 70)

    print()
    print("FORMULATION")
    print("-" * 70)
    print("F - We = N(Eo + mEg + Efw)")
    print()
    print("The calibration engine receives no true m.")
    print("It must recover m from material-balance history.")

    test_input_validation()
    test_gas_cap_calibration()
    test_zero_gas_cap()
    test_result_structure()

    print()
    print("=" * 70)
    print("✓ ALL GAS-CAP CALIBRATION CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()