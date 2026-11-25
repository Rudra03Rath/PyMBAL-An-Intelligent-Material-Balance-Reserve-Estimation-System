"""
Validation suite for the coupled gas-cap + aquifer model selector.

The tests deliberately use synthetic histories with known physical
configurations and verify:

- input validation
- no-gas/no-aquifer recovery
- gas-cap-only recovery
- aquifer-only recovery
- gas-cap + aquifer recovery
- correct identification of the best configuration
- result structure and comparison table
"""

import numpy as np
import pandas as pd

from src.influx.gas_cap_aquifer_calibration import (
    GasCapAquiferCalibrator,
)


# ======================================================================
# TEST DATA
# ======================================================================

TRUE_N = 10_000_000.0
TRUE_M = 0.8
TRUE_PA = 4000.0
TRUE_C = 2.0

# Nonlinear depletion history.
TIME = np.arange(0.0, 301.0, 15.0)
u = TIME / TIME[-1]
PRESSURE_DROP = 1500.0 * u**1.35
PRESSURE = TRUE_PA - PRESSURE_DROP

# ----------------------------------------------------------------------
# Controlled synthetic PVT
#
# These are deliberately smooth, positive, pressure-dependent trends.
# The expansion equations are the same formulations used by PyMBAL:
#
#   Eo = (Bo - Boi) + (Rsi - Rs) Bg
#   Eg = Boi * (Bg/Bgi - 1)
#
# Efw supplied to the gas-cap calibrator is the base formation/water
# expansion (without embedding the unknown m inside Efw).
# ----------------------------------------------------------------------

BOI = 1.42
BGI = 0.00082
RSI = 780.0
SWI = 0.25
CF = 6.0e-6
CW = 3.0e-6

# Bo decreases smoothly during depletion.
BO = (
    BOI
    - 1.00e-4 * PRESSURE_DROP
    + 1.50e-8 * PRESSURE_DROP**2
)

# Bg increases during depletion. The quadratic term prevents a simple
# linear relationship between Eg and pressure.
BG = (
    BGI
    + 1.00e-7 * PRESSURE_DROP
    + 2.00e-11 * PRESSURE_DROP**2
)

# Rs decreases during depletion with mild curvature.
RS = (
    RSI
    - 0.22 * PRESSURE_DROP
    + 2.00e-5 * PRESSURE_DROP**2
)

BO = np.maximum(BO, 1.05)
BG = np.maximum(BG, 1.0e-8)
RS = np.maximum(RS, 50.0)

# PyMBAL gas-cap expansion formulation.
EO = (
    (BO - BOI)
    + (RSI - RS) * BG
)

EG = (
    BOI * (BG / BGI - 1.0)
)

# Base formation + connate-water expansion.
# This corresponds to the material-balance basis used by the coupled
# calibrator; the unknown m is handled separately through m*Eg.
EFW = (
    BOI
    * (
        (CF + CW * SWI)
        / (1.0 - SWI)
    )
    * PRESSURE_DROP
)

EO[0] = 0.0
EG[0] = 0.0
EFW[0] = 0.0


def schilthuis_we():
    from src.influx.schilthuis import SchilthuisAquifer

    model = SchilthuisAquifer(
        aquifer_pressure_psia=TRUE_PA,
        influx_constant_rb_per_day_psi=TRUE_C,
    )

    result = model.calculate_influx(
        pressure_psia=PRESSURE,
        time_days=TIME,
    )

    if isinstance(result, pd.DataFrame):
        return result["We_rb"].to_numpy(dtype=float)

    return np.asarray(result["We_rb"], dtype=float)


WE = schilthuis_we()


def make_F(N, m, We):
    return N * (
        EO
        + m * EG
        + EFW
    ) + We


# ======================================================================
# CONFIGURATION
# ======================================================================

MODEL_CONFIGS = {
    "schilthuis": {
        "aquifer_pressure_psia": TRUE_PA,
        "c_bounds": (0.01, 10.0),
    },
    "fetkovich": {
        "initial_aquifer_pressure_psia": TRUE_PA,
        "total_compressibility_psi_inv": 2.0e-5,
        "productivity_index_bounds": (1.0, 1000.0),
        "aquifer_volume_bounds": (1.0e5, 1.0e8),
    },
    "van_everdingen_hurst": {
        "aquifer_porosity": 0.20,
        "aquifer_thickness_ft": 100.0,
        "total_compressibility_psi_inv": 2.0e-5,
        "water_viscosity_cp": 0.50,
        "reservoir_radius_ft": 1000.0,
        "encroachment_angle_deg": 360.0,
        "permeability_bounds": (1.0, 1000.0),
    },
}


# ======================================================================
# HELPERS
# ======================================================================

def build_selector(F):
    return GasCapAquiferCalibrator(
        F_rb=F,
        Eo_rb_per_STB=EO,
        Eg_rb_per_STB=EG,
        Efw_rb_per_STB=EFW,
        time_days=TIME,
        pressure_psia=PRESSURE,
    )


def assert_close(value, truth, rel_tol=1e-6, label="value"):
    assert abs(value - truth) / max(abs(truth), 1.0) < rel_tol, (
        f"{label} recovery failed: value={value}, truth={truth}"
    )


# ======================================================================
# INPUT VALIDATION
# ======================================================================

def test_input_validation():
    print("=" * 70)
    print("PyMBAL - GAS-CAP + AQUIFER MODEL SELECTION VALIDATION")
    print("=" * 70)

    F = make_F(TRUE_N, TRUE_M, WE)

    print()
    print("INPUT VALIDATION")
    print("-" * 70)

    try:
        GasCapAquiferCalibrator(
            F[:-1], EO, EG, EFW, TIME, PRESSURE
        )
        raise AssertionError("Mismatched lengths were not rejected.")
    except ValueError:
        print("✓ Mismatched input lengths rejected.")

    bad_time = TIME.copy()
    bad_time[1] = 0.0

    try:
        GasCapAquiferCalibrator(
            F, EO, EG, EFW, bad_time, PRESSURE
        )
        raise AssertionError("Non-monotonic time was not rejected.")
    except ValueError:
        print("✓ Non-monotonic time rejected.")

    bad_F = F.copy()
    bad_F[2] = -1.0

    try:
        GasCapAquiferCalibrator(
            bad_F, EO, EG, EFW, TIME, PRESSURE
        )
        raise AssertionError("Negative F was not rejected.")
    except ValueError:
        print("✓ Negative underground withdrawal rejected.")

    print()
    print("✓ INPUT VALIDATION PASSED")


# ======================================================================
# CASE 1
# ======================================================================

def test_no_gas_no_aquifer():
    print()
    print("=" * 70)
    print("CASE 1: m = 0, We = 0")
    print("=" * 70)

    F = make_F(TRUE_N, 0.0, np.zeros_like(TIME))
    selector = build_selector(F)

    result = selector.select_best_model(
        MODEL_CONFIGS,
        m_bounds=(0.0, 5.0),
    )

    print(result.comparison.to_string(index=False))

    assert result.best_case == "No Gas Cap + No Aquifer"
    assert_close(
        result.best_ooip_stb,
        TRUE_N,
        label="OOIP",
    )
    assert_close(
        result.best_gas_cap_ratio_m,
        0.0,
        label="m",
    )
    assert result.best_rmse_rb < 1e-6

    print("✓ Case 1 correctly selected.")


# ======================================================================
# CASE 2
# ======================================================================

def test_gas_cap_only():
    print()
    print("=" * 70)
    print("CASE 2: m > 0, We = 0")
    print("=" * 70)

    F = make_F(TRUE_N, TRUE_M, np.zeros_like(TIME))
    selector = build_selector(F)

    result = selector.select_best_model(
        MODEL_CONFIGS,
        m_bounds=(0.0, 5.0),
    )

    print(result.comparison.to_string(index=False))

    assert result.best_case == "Gas Cap + No Aquifer"
    assert_close(
        result.best_ooip_stb,
        TRUE_N,
        label="OOIP",
    )
    assert_close(
        result.best_gas_cap_ratio_m,
        TRUE_M,
        label="m",
    )
    assert result.best_rmse_rb < 1e-6

    print("✓ Case 2 correctly selected.")


# ======================================================================
# CASE 3
# ======================================================================

def test_aquifer_only():
    print()
    print("=" * 70)
    print("CASE 3: m = 0, We > 0")
    print("=" * 70)

    F = make_F(TRUE_N, 0.0, WE)
    selector = build_selector(F)

    result = selector.select_best_model(
        MODEL_CONFIGS,
        m_bounds=(0.0, 5.0),
    )

    print(result.comparison.to_string(index=False))

    assert result.best_case == "No Gas Cap + Aquifer"
    assert result.best_aquifer_model == "Schilthuis Steady-State"
    assert_close(
        result.best_ooip_stb,
        TRUE_N,
        label="OOIP",
    )
    assert_close(
        result.best_gas_cap_ratio_m,
        0.0,
        label="m",
    )
    assert result.best_rmse_rb < 1e-5

    print("✓ Case 3 correctly selected.")


# ======================================================================
# CASE 4
# ======================================================================

def test_gas_cap_and_aquifer():
    print()
    print("=" * 70)
    print("CASE 4: m > 0, We > 0")
    print("=" * 70)

    F = make_F(TRUE_N, TRUE_M, WE)
    selector = build_selector(F)

    result = selector.select_best_model(
        MODEL_CONFIGS,
        m_bounds=(0.0, 5.0),
    )

    print(result.comparison.to_string(index=False))

    if result.is_ambiguous:
        raise AssertionError(
            "Case 4 is not physically identifiable from the supplied "
            "material-balance history. Near-equivalent candidates: "
            + ", ".join(result.near_best_candidates)
        )

    assert result.best_case == "Gas Cap + Aquifer"
    assert result.best_aquifer_model == "Schilthuis Steady-State"

    assert_close(
        result.best_ooip_stb,
        TRUE_N,
        label="OOIP",
    )
    assert_close(
        result.best_gas_cap_ratio_m,
        TRUE_M,
        label="m",
    )

    assert_close(
        result.best_result.aquifer_parameters["C_rb_per_day_psi"],
        TRUE_C,
        rel_tol=1e-5,
        label="Schilthuis C",
    )

    assert result.best_rmse_rb < 1e-5

    print("✓ Case 4 correctly selected.")


# ======================================================================
# GAS-CAP PRESENCE SAFEGUARD
# ======================================================================

def test_gas_cap_presence_safeguard():
    """
    A gas-cap-present candidate must not be accepted when calibrated m
    collapses to zero. m = 0 belongs to the explicit no-gas cases.
    """
    print()
    print("=" * 70)
    print("GAS-CAP PRESENCE SAFEGUARD")
    print("=" * 70)

    F = make_F(TRUE_N, 0.0, np.zeros_like(TIME))
    selector = build_selector(F)

    result = selector.select_best_model(
        MODEL_CONFIGS,
        m_bounds=(0.0, 5.0),
        gas_cap_presence_threshold=1.0e-6,
    )

    gas_rows = result.comparison[result.comparison["Gas_Cap"]]
    successful_gas = gas_rows[gas_rows["Success"]]

    assert not (
        np.isfinite(successful_gas["m"])
        & (successful_gas["m"] <= 1.0e-6)
    ).any()

    assert result.best_case == "No Gas Cap + No Aquifer"

    print("✓ Gas-cap presence criterion PASSED.")


# ======================================================================
# IDENTIFIABILITY REPORTING
# ======================================================================

def test_identifiability_reporting():
    print()
    print("=" * 70)
    print("IDENTIFIABILITY REPORTING")
    print("=" * 70)

    # Construct a deliberately degenerate history where the influx
    # contribution is represented almost exactly by the expansion basis.
    t = np.arange(0.0, 301.0, 30.0)
    p_drop = 5.0 * t
    p = TRUE_PA - p_drop

    eo = 3.0e-4 * t + 6.0e-6 * t**2
    eg = 8.0e-4 * t + 1.0e-5 * t**2
    efw = 3.0e-5 * t

    eo[0] = 0.0
    eg[0] = 0.0
    efw[0] = 0.0

    from src.influx.schilthuis import SchilthuisAquifer

    aquifer = SchilthuisAquifer(
        aquifer_pressure_psia=TRUE_PA,
        influx_constant_rb_per_day_psi=TRUE_C,
    )

    influx = aquifer.calculate_influx(
        pressure_psia=p,
        time_days=t,
    )

    if isinstance(influx, pd.DataFrame):
        we = influx["We_rb"].to_numpy(dtype=float)
    else:
        we = np.asarray(influx["We_rb"], dtype=float)

    F = TRUE_N * (
        eo + TRUE_M * eg + efw
    ) + we

    selector = GasCapAquiferCalibrator(
        F_rb=F,
        Eo_rb_per_STB=eo,
        Eg_rb_per_STB=eg,
        Efw_rb_per_STB=efw,
        time_days=t,
        pressure_psia=p,
    )

    result = selector.select_best_model(
        MODEL_CONFIGS,
        m_bounds=(0.0, 5.0),
    )

    assert isinstance(result.is_ambiguous, bool)
    assert len(result.near_best_candidates) >= 1

    print(
        "✓ Identifiability reporting is available."
    )


# ======================================================================
# RESULT STRUCTURE
# ======================================================================

def test_result_structure():
    print()
    print("=" * 70)
    print("RESULT STRUCTURE VALIDATION")
    print("=" * 70)

    F = make_F(TRUE_N, TRUE_M, WE)
    selector = build_selector(F)

    result = selector.select_best_model(
        MODEL_CONFIGS,
        m_bounds=(0.0, 5.0),
    )

    assert isinstance(result.comparison, pd.DataFrame)
    assert len(result.all_results) >= 4

    required_columns = {
        "Configuration",
        "Aquifer_Model",
        "Gas_Cap",
        "Aquifer",
        "Success",
        "RMSE_rb",
        "MAE_rb",
        "R2",
        "OOIP_STB",
        "m",
    }

    assert required_columns.issubset(result.comparison.columns)

    assert result.best_result.success
    assert np.isfinite(result.best_rmse_rb)
    assert np.isfinite(result.best_r2)

    assert isinstance(result.is_ambiguous, bool)
    assert isinstance(result.near_best_candidates, tuple)

    print("✓ Result structure validation PASSED.")


# ======================================================================
# MAIN
# ======================================================================

def main():
    test_input_validation()
    test_no_gas_no_aquifer()
    test_gas_cap_only()
    test_aquifer_only()
    test_gas_cap_and_aquifer()
    test_gas_cap_presence_safeguard()
    test_identifiability_reporting()
    test_result_structure()

    print()
    print("=" * 70)
    print("✓ ALL GAS-CAP + AQUIFER MODEL SELECTION CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
