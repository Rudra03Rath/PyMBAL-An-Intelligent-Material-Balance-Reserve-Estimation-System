"""
PyMBAL - AQUIFER MODEL SELECTION VALIDATION

Validates automatic selection of the correct aquifer model
using three synthetic material-balance reservoirs.

Expected:

    Schilthuis synthetic
        -> Schilthuis selected

    Fetkovich synthetic
        -> Fetkovich selected

    V-E-H synthetic
        -> V-E-H selected
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.influx.model_selection import (
    AquiferModelSelector,
)


# ======================================================================
# PATHS
# ======================================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

SCHILTHUIS_FILE = (
    PROJECT_ROOT
    / "data"
    / "sample_data"
    / "synthetic_water_influx_reservoir.csv"
)

FETKOVICH_FILE = (
    PROJECT_ROOT
    / "data"
    / "sample_data"
    / "synthetic_fetkovich_reservoir.csv"
)

VEH_FILE = (
    PROJECT_ROOT
    / "data"
    / "sample_data"
    / "synthetic_van_everdingen_hurst"
    / "synthetic_van_everdingen_hurst_reservoir.csv"
)


# ======================================================================
# KNOWN MODEL PARAMETERS
# ======================================================================

AQUIFER_PRESSURE = 4000.0

SCHILTHUIS_C = 2.0

AQUIFER_CT = 2.0e-5

FETKOVICH_J = 100.0
FETKOVICH_VI = 5_000_000.0

VEH_K = 100.0
VEH_PHI = 0.20
VEH_H = 100.0
VEH_MU = 0.50
VEH_RE = 1000.0
VEH_ANGLE = 360.0

TRUE_OOIP = 10_000_000.0


# ======================================================================
# TOLERANCES
# ======================================================================

WINNING_MODEL_RMSE_TOLERANCE = 1e-2

MIN_MODEL_SEPARATION_RB = 1.0e2

R2_TOLERANCE = 0.999999


# ======================================================================
# LOAD DATA
# ======================================================================

def load_dataset(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Synthetic dataset not found:\n{path}"
        )

    df = pd.read_csv(path)

    required = [
        "Days",
        "Pressure_psia",
        "F_rb",
        "Etotal_rb_per_STB",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Dataset is missing columns: {missing}"
        )

    return df


# ======================================================================
# BUILD SELECTOR
# ======================================================================

def build_selector(df):

    return AquiferModelSelector(
        time_days=df["Days"].to_numpy(),
        pressure_psia=df[
            "Pressure_psia"
        ].to_numpy(),
        F_rb=df[
            "F_rb"
        ].to_numpy(),
        E_rb_per_STB=df[
            "Etotal_rb_per_STB"
        ].to_numpy(),
    )


# ======================================================================
# COMMON CONFIGURATION
# ======================================================================

def all_model_configs():

    return {

        "schilthuis": {

            "aquifer_pressure_psia":
                AQUIFER_PRESSURE,

            "c_bounds":
                (1e-6, 100.0),
        },

        "fetkovich": {

            "initial_aquifer_pressure_psia":
                AQUIFER_PRESSURE,

            "total_compressibility_psi_inv":
                AQUIFER_CT,

            "productivity_index_bounds":
                (1.0, 1000.0),

            "aquifer_volume_bounds":
                (100_000.0, 100_000_000.0),
        },

        "van_everdingen_hurst": {

            "aquifer_porosity":
                VEH_PHI,

            "aquifer_thickness_ft":
                VEH_H,

            "total_compressibility_psi_inv":
                AQUIFER_CT,

            "water_viscosity_cp":
                VEH_MU,

            "reservoir_radius_ft":
                VEH_RE,

            "encroachment_angle_deg":
                VEH_ANGLE,

            "permeability_bounds":
                (1.0, 1000.0),
        },
    }


# ======================================================================
# PRINT COMPARISON
# ======================================================================

def print_comparison(result):

    print()

    display_columns = [
        "Model",
        "Success",
        "RMSE_rb",
        "MAE_rb",
        "R2",
        "OOIP_STB",
    ]

    print(
        result.comparison[
            display_columns
        ].to_string(index=False)
    )

    failed = result.comparison[
        ~result.comparison["Success"]
    ]

    if not failed.empty:

        print()
        print("FAILED CANDIDATES")
        print("-" * 70)

        for _, row in failed.iterrows():

            print(
                f"{row['Model']}: "
                f"{row['Error']}"
            )


# ======================================================================
# VALIDATION OF WINNER
# ======================================================================

def validate_winner(
    result,
    expected_model,
):

    assert result.best_result.success

    assert (
        result.best_rmse_rb
        < WINNING_MODEL_RMSE_TOLERANCE
    ), (
        f"Winning model RMSE is too large: "
        f"{result.best_rmse_rb}"
    )

    assert (
        result.best_r2
        > R2_TOLERANCE
    ), (
        f"Winning model R² is too low: "
        f"{result.best_r2}"
    )

    assert expected_model.lower() in (
        result.best_model.lower()
    ), (
        f"Expected {expected_model}, "
        f"but selected {result.best_model}"
    )

    valid = result.comparison[
        result.comparison["Success"]
    ]

    winning_rmse = result.best_rmse_rb

    competitors = valid[
        valid["Model"]
        != result.best_model
    ]

    if not competitors.empty:

        minimum_competitor_rmse = (
            competitors["RMSE_rb"].min()
        )

        assert (
            minimum_competitor_rmse
            > winning_rmse
        ), (
            "Another model has an equal or "
            "lower RMSE than the expected winner."
        )


# ======================================================================
# INPUT VALIDATION
# ======================================================================

def test_input_validation():

    print()
    print("=" * 70)
    print(
        "MODEL SELECTION INPUT VALIDATION"
    )
    print("=" * 70)

    try:

        AquiferModelSelector(
            time_days=[0, 10, 20],
            pressure_psia=[4000, 3900],
            F_rb=[0, 1000, 2000],
            E_rb_per_STB=[0, 0.01, 0.02],
        )

        raise AssertionError

    except ValueError:

        print(
            "✓ Mismatched input lengths rejected."
        )

    try:

        AquiferModelSelector(
            time_days=[1, 10, 20],
            pressure_psia=[4000, 3900, 3800],
            F_rb=[0, 1000, 2000],
            E_rb_per_STB=[0, 0.01, 0.02],
        )

        raise AssertionError

    except ValueError:

        print(
            "✓ Non-zero initial time rejected."
        )

    try:

        AquiferModelSelector(
            time_days=[0, 20, 10],
            pressure_psia=[4000, 3900, 3800],
            F_rb=[0, 1000, 2000],
            E_rb_per_STB=[0, 0.01, 0.02],
        )

        raise AssertionError

    except ValueError:

        print(
            "✓ Non-monotonic time rejected."
        )

    try:

        AquiferModelSelector(
            time_days=[0, 10, 20],
            pressure_psia=[4000, 3900, 3800],
            F_rb=[0, -1000, 2000],
            E_rb_per_STB=[0, 0.01, 0.02],
        )

        raise AssertionError

    except ValueError:

        print(
            "✓ Negative underground withdrawal rejected."
        )

    print()
    print(
        "✓ Model-selection input validation PASSED."
    )


# ======================================================================
# RESULT STRUCTURE
# ======================================================================

def test_result_structure():

    print()
    print("=" * 70)
    print(
        "MODEL SELECTION RESULT STRUCTURE"
    )
    print("=" * 70)

    df = load_dataset(
        SCHILTHUIS_FILE
    )

    selector = build_selector(df)

    result = selector.select_best_model(
        model_configs=all_model_configs()
    )

    assert result.best_model is not None

    assert result.best_result is not None

    assert isinstance(
        result.comparison,
        pd.DataFrame,
    )

    assert isinstance(
        result.all_results,
        dict,
    )

    required_columns = [
        "Model",
        "Success",
        "RMSE_rb",
        "MAE_rb",
        "R2",
        "OOIP_STB",
        "Error",
    ]

    for column in required_columns:

        assert column in (
            result.comparison.columns
        )

    assert len(result.comparison) == 3

    print(
        "✓ Result structure validation PASSED."
    )


# ======================================================================
# SCHILTHUIS
# ======================================================================

def test_schilthuis_dataset():

    print()
    print("=" * 70)
    print(
        "SCHILTHUIS DATASET MODEL SELECTION"
    )
    print("=" * 70)

    df = load_dataset(
        SCHILTHUIS_FILE
    )

    print(
        f"✓ Loaded dataset: "
        f"{SCHILTHUIS_FILE.name}"
    )

    selector = build_selector(df)

    result = selector.select_best_model(
        model_configs=all_model_configs()
    )

    print_comparison(result)

    print()
    print(
        f"Best-fit model : "
        f"{result.best_model}"
    )

    print(
        f"Best RMSE      : "
        f"{result.best_rmse_rb:.9f} rb"
    )

    print(
        f"Best R²        : "
        f"{result.best_r2:.9f}"
    )

    validate_winner(
        result,
        "Schilthuis",
    )

    print(
        "✓ Schilthuis model-selection check PASSED."
    )


# ======================================================================
# FETKOVICH
# ======================================================================

def test_fetkovich_dataset():

    print()
    print("=" * 70)
    print(
        "FETKOVICH DATASET MODEL SELECTION"
    )
    print("=" * 70)

    df = load_dataset(
        FETKOVICH_FILE
    )

    print(
        f"✓ Loaded dataset: "
        f"{FETKOVICH_FILE.name}"
    )

    selector = build_selector(df)

    result = selector.select_best_model(
        model_configs=all_model_configs()
    )

    print_comparison(result)

    print()
    print(
        f"Best-fit model : "
        f"{result.best_model}"
    )

    print(
        f"Best RMSE      : "
        f"{result.best_rmse_rb:.9f} rb"
    )

    print(
        f"Best R²        : "
        f"{result.best_r2:.9f}"
    )

    validate_winner(
        result,
        "Fetkovich",
    )

    print(
        "✓ Fetkovich model-selection check PASSED."
    )


# ======================================================================
# V-E-H
# ======================================================================

def test_veh_dataset():

    print()
    print("=" * 70)
    print(
        "VAN EVERDINGEN-HURST DATASET MODEL SELECTION"
    )
    print("=" * 70)

    df = load_dataset(
        VEH_FILE
    )

    print(
        f"✓ Loaded dataset: "
        f"{VEH_FILE.name}"
    )

    selector = build_selector(df)

    result = selector.select_best_model(
        model_configs=all_model_configs()
    )

    print_comparison(result)

    print()
    print(
        f"Best-fit model : "
        f"{result.best_model}"
    )

    print(
        f"Best RMSE      : "
        f"{result.best_rmse_rb:.9f} rb"
    )

    print(
        f"Best R²        : "
        f"{result.best_r2:.9f}"
    )

    validate_winner(
        result,
        "Van Everdingen-Hurst",
    )

    print(
        "✓ V-E-H model-selection check PASSED."
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print(
        "PyMBAL - AQUIFER MODEL SELECTION VALIDATION"
    )
    print("=" * 70)

    print()
    print("Candidate models:")
    print(
        "  1. Schilthuis Steady-State"
    )
    print(
        "  2. Fetkovich Finite Aquifer"
    )
    print(
        "  3. Van Everdingen-Hurst"
    )

    print()
    print("IMPORTANT:")
    print(
        "The selector receives no We history."
    )
    print(
        "Each model is calibrated from "
        "material-balance history only."
    )

    test_input_validation()

    test_result_structure()

    test_schilthuis_dataset()

    test_fetkovich_dataset()

    test_veh_dataset()

    print()
    print("=" * 70)
    print(
        "✓ ALL AQUIFER MODEL SELECTION CHECKS PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()