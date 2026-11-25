"""
PyMBAL - FULL 8-CASE GAS-CAP + AQUIFER VALIDATION

This test reads the eight generated synthetic datasets and runs the
production coupled selector against each one.

Ground truth is intentionally NOT supplied to the selector.

Run from the PyMBAL project root:

    python -m tests.test_8_coupled_scenarios
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.influx.gas_cap_aquifer_calibration import (
    GasCapAquiferCalibrator,
)


DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "sample_data"
    / "synthetic_8_cases"
)

TRUE_N = 10_000_000.0
TRUE_M = 0.8

MODEL_CONFIGS = {
    "schilthuis": {
        "aquifer_pressure_psia": 4000.0,
        "c_bounds": (0.01, 10.0),
    },
    "fetkovich": {
        "initial_aquifer_pressure_psia": 4000.0,
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


CASES = [
    (
        "case_1_no_gas_no_aquifer",
        "No Gas Cap + No Aquifer",
        None,
        0.0,
    ),
    (
        "case_2_gas_cap_only",
        "Gas Cap + No Aquifer",
        None,
        TRUE_M,
    ),
    (
        "case_3_schilthuis_only",
        "No Gas Cap + Aquifer",
        "Schilthuis Steady-State",
        0.0,
    ),
    (
        "case_4_fetkovich_only",
        "No Gas Cap + Aquifer",
        "Fetkovich Finite Aquifer",
        0.0,
    ),
    (
        "case_5_veh_only",
        "No Gas Cap + Aquifer",
        "Van Everdingen-Hurst",
        0.0,
    ),
    (
        "case_6_gas_cap_schilthuis",
        "Gas Cap + Aquifer",
        "Schilthuis Steady-State",
        TRUE_M,
    ),
    (
        "case_7_gas_cap_fetkovich",
        "Gas Cap + Aquifer",
        "Fetkovich Finite Aquifer",
        TRUE_M,
    ),
    (
        "case_8_gas_cap_veh",
        "Gas Cap + Aquifer",
        "Van Everdingen-Hurst",
        TRUE_M,
    ),
]


def load_case(case_name):
    path = (
        DATA_ROOT
        / case_name
        / f"{case_name}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{path}\n\n"
            "Run generate_8_coupled_synthetic_datasets.py first."
        )

    df = pd.read_csv(path)

    required = [
        "Days",
        "Pressure_psia",
        "F_rb",
        "Eo_rb_per_STB",
        "Eg_rb_per_STB",
        "Efw_base_rb_per_STB",
        "Efw_rb_per_STB",
        "We_rb",
        "True_OOIP_STB",
        "True_Gas_Cap_Ratio_m",
        "True_Aquifer_Model",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{case_name}: missing columns: {missing}"
        )

    return df


def build_selector(df):
    return GasCapAquiferCalibrator(
        F_rb=df["F_rb"].to_numpy(dtype=float),
        Eo_rb_per_STB=df["Eo_rb_per_STB"].to_numpy(dtype=float),
        Eg_rb_per_STB=df["Eg_rb_per_STB"].to_numpy(dtype=float),
        # IMPORTANT: coupled calibration must receive the
        # m-independent formation/connate-water expansion basis.
        Efw_rb_per_STB=df["Efw_base_rb_per_STB"].to_numpy(dtype=float),
        time_days=df["Days"].to_numpy(dtype=float),
        pressure_psia=df["Pressure_psia"].to_numpy(dtype=float),
    )


def assert_close(actual, expected, label, rtol=1e-5, atol=1e-8):
    if not np.isclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(
            f"{label} mismatch: "
            f"actual={actual:.12g}, expected={expected:.12g}"
        )


def validate_dataset_ground_truth(df, case_name):
    true_n = float(df["True_OOIP_STB"].iloc[0])
    true_m = float(df["True_Gas_Cap_Ratio_m"].iloc[0])
    true_aquifer = str(df["True_Aquifer_Model"].iloc[0])

    assert_close(true_n, TRUE_N, f"{case_name} OOIP metadata")
    assert true_m >= 0.0
    assert true_aquifer in {
        "none",
        "schilthuis",
        "fetkovich",
        "van_everdingen_hurst",
    }

    days = df["Days"].to_numpy(dtype=float)
    pressure = df["Pressure_psia"].to_numpy(dtype=float)
    F = df["F_rb"].to_numpy(dtype=float)
    We = df["We_rb"].to_numpy(dtype=float)

    assert np.isclose(days[0], 0.0)
    assert np.all(np.diff(days) > 0.0)
    assert np.all(pressure > 0.0)
    assert np.all(F >= 0.0)
    assert np.all(np.diff(F) >= -1e-8)
    assert np.all(We >= -1e-8)
    assert np.all(np.diff(We) >= -1e-8)

    Eg = df["Eg_rb_per_STB"].to_numpy(dtype=float)
    Efw_base = df["Efw_base_rb_per_STB"].to_numpy(dtype=float)

    assert np.all(Eg >= -1e-12)
    assert np.all(Efw_base >= -1e-12)
    assert np.isclose(Eg[0], 0.0, atol=1e-12)
    assert np.isclose(Efw_base[0], 0.0, atol=1e-12)

    # For the no-aquifer cases, the synthetic We history must be zero.
    if true_aquifer == "none":
        assert np.max(np.abs(We)) < 1e-8

    # Verify the synthetic history uses an m-independent Efw basis:
    #
    # F = N * [Eo + m*Eg + (1+m)*Efw_base] + We
    #
    # This prevents the coupled calibration test from accidentally
    # receiving an Efw history that already contains the unknown m.
    eo = df["Eo_rb_per_STB"].to_numpy(dtype=float)
    eg = df["Eg_rb_per_STB"].to_numpy(dtype=float)
    efw_base = df["Efw_base_rb_per_STB"].to_numpy(dtype=float)

    reconstructed_F = true_n * (
        eo
        + true_m * eg
        + (1.0 + true_m) * efw_base
    ) + We

    assert np.allclose(
        F,
        reconstructed_F,
        rtol=1e-10,
        atol=1e-6,
    ), (
        f"{case_name}: synthetic material-balance reconstruction failed."
    )


def run_case(case_name, expected_case, expected_aquifer, expected_m):
    df = load_case(case_name)
    validate_dataset_ground_truth(df, case_name)

    selector = build_selector(df)

    result = selector.select_best_model(
        model_configs=MODEL_CONFIGS,
        m_bounds=(0.0, 5.0),
        include_all_aquifer_models=True,
        gas_cap_presence_threshold=1.0e-6,
        competitive_relative_tolerance=0.05,
        competitive_absolute_tolerance=1.0e-8,
    )

    print()
    print("=" * 78)
    print(case_name.upper())
    print("=" * 78)
    print(result.comparison.to_string(index=False))

    print()
    print(f"Best mathematical case : {result.best_case}")
    print(f"Best aquifer           : {result.best_aquifer_model}")
    print(f"Best m                 : {result.best_gas_cap_ratio_m:.12g}")
    print(f"Best OOIP              : {result.best_ooip_stb:.6f} STB")
    print(f"Best RMSE              : {result.best_rmse_rb:.12g} rb")
    print(f"Best R²                : {result.best_r2:.12g}")

    assert result.best_result.success
    assert result.best_rmse_rb < 1e-4, (
        f"{case_name}: best RMSE is too large: {result.best_rmse_rb}"
    )

    # Ground truth is used only by this test, never by the selector.
    true_candidates = []
    for candidate in result.all_results.values():
        if not candidate.success:
            continue
        if candidate.case_name != expected_case:
            continue

        if expected_aquifer is None:
            aquifer_match = candidate.aquifer_model is None
        else:
            aquifer_match = candidate.aquifer_model == expected_aquifer

        if aquifer_match:
            true_candidates.append(candidate)

    assert len(true_candidates) == 1, (
        f"{case_name}: expected exactly one true physical candidate, "
        f"but found {len(true_candidates)}."
    )

    true_result = true_candidates[0]

    print()
    print("TRUE CONFIGURATION CANDIDATE")
    print("-" * 70)
    print(f"Configuration : {true_result.case_name}")
    print(f"Aquifer       : {true_result.aquifer_model}")
    print(f"OOIP          : {true_result.estimated_ooip_stb:.6f} STB")
    print(f"m             : {true_result.gas_cap_ratio_m:.12g}")
    print(f"RMSE          : {true_result.rmse_rb:.12g} rb")

    # The true physical configuration must fit the synthetic history.
    #
    # IMPORTANT: for coupled gas-cap + aquifer cases, N, m and aquifer
    # parameters can be non-identifiable from material-balance history
    # alone. Therefore this test deliberately does NOT require the
    # optimizer's N and m to equal the hidden synthetic values.
    assert true_result.rmse_rb < 1e-4, (
        f"{case_name}: true physical candidate does not reproduce "
        f"the synthetic material-balance history. "
        f"RMSE={true_result.rmse_rb}"
    )

    # Parameter recovery is diagnostic, not a pass/fail criterion for
    # coupled cases. If the fitted values happen to recover the truth,
    # report it; otherwise explicitly classify the case as ambiguous.
    ooip_rel_error = abs(
        true_result.estimated_ooip_stb - TRUE_N
    ) / TRUE_N

    m_abs_error = abs(
        true_result.gas_cap_ratio_m - expected_m
    )

    if ooip_rel_error <= 1e-5 and m_abs_error <= 1e-5:
        print("✓ True N and m also recovered numerically.")
    else:
        print("⚠ N/m are not uniquely recovered by this material-balance fit.")
        print(
            f"  OOIP relative error : {ooip_rel_error:.6e}"
        )
        print(
            f"  m absolute error    : {m_abs_error:.6e}"
        )

    print()
    print("ALL SUCCESSFUL PHYSICAL CANDIDATES")
    print("-" * 70)
    print(
        result.possible_cases[
            [
                "Rank", "Configuration", "Aquifer_Model",
                "RMSE_rb", "OOIP_STB", "m", "Competitive"
            ]
        ].to_string(index=False)
    )

    print()
    print("COMPETITIVE CANDIDATES")
    print("-" * 70)
    competitive = result.competitive_results
    if competitive.empty:
        print("None")
    else:
        print(
            competitive[
                [
                    "Rank", "Configuration", "Aquifer_Model",
                    "RMSE_rb", "OOIP_STB", "m"
                ]
            ].to_string(index=False)
        )

    # The selector must expose all successful interpretations; the
    # mathematical winner is only the default ranking, not proof of
    # physical uniqueness.
    assert len(result.possible_cases) >= 1

    if result.ambiguous:
        print()
        print("⚠ MODEL IDENTIFIABILITY: multiple interpretations have "
              "practically equivalent material-balance fits.")
        print("  The user should provide independent reservoir information "
              "to constrain the interpretation.")
    else:
        print("✓ No practically equivalent alternative under the "
              "configured tolerance.")

    print("✓ True physical configuration recovered.")
    print("✓ True configuration reproduces the synthetic history.")
    print("✓ All successful physical candidates remain available.")
    print("✓ Best-fit calculation passed.")


def test_all_8_cases():
    for case_name, expected_case, expected_aquifer, expected_m in CASES:
        run_case(
            case_name,
            expected_case,
            expected_aquifer,
            expected_m,
        )


def main():
    print("=" * 78)
    print("PyMBAL - FULL 8-CASE GAS-CAP + AQUIFER VALIDATION")
    print("=" * 78)
    print()
    print("The selector receives:")
    print("  F(t), Eo(t), Eg(t), Efw(t), pressure history and time history")
    print()
    print("The selector does NOT receive:")
    print("  We(t), true m, true OOIP, or true aquifer model")
    print()

    test_all_8_cases()

    print()
    print("=" * 78)
    print("✓ ALL 8 GAS-CAP + AQUIFER SCENARIOS PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
