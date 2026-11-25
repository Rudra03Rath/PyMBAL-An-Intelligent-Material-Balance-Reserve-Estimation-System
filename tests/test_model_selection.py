"""
PyMBAL - MODEL SELECTION LAYER VALIDATION
=========================================

Tests the analysis-layer wrapper around the already validated
GasCapAquiferCalibrator.

Important:
---------
This test does NOT assume that material balance uniquely identifies
the true physical mechanism. It checks:

1. Constraint validation.
2. Automatic mode evaluates the available candidate space.
3. User constraints actually restrict the candidate space.
4. Aquifer-model constraints are propagated correctly.
5. Best-fit and competitive candidates remain exposed.
6. The layer returns a stable ModelSelectionResult.

The detailed physics/recovery of all eight synthetic scenarios remains
the responsibility of tests.test_8_coupled_scenarios.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.model_selection import (
    IdentifiabilityAssessment,
    ModelSelectionConstraints,
    ModelSelectionResult,
    ReservoirModelSelector,
)


ROOT = Path("data/sample_data/synthetic_8_cases")


# Physical calibration configurations used by the already validated
# coupled calibrator. The model-selection layer only transports these
# configurations; it does not implement the aquifer physics.
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


# ======================================================================
# DATA LOADING
# ======================================================================

CASE_FILES = {
    "case_1_no_gas_no_aquifer": (
        ROOT
        / "case_1_no_gas_no_aquifer"
        / "case_1_no_gas_no_aquifer.csv"
    ),
    "case_2_gas_cap_only": (
        ROOT
        / "case_2_gas_cap_only"
        / "case_2_gas_cap_only.csv"
    ),
    "case_3_schilthuis_only": (
        ROOT
        / "case_3_schilthuis_only"
        / "case_3_schilthuis_only.csv"
    ),
    "case_4_fetkovich_only": (
        ROOT
        / "case_4_fetkovich_only"
        / "case_4_fetkovich_only.csv"
    ),
    "case_5_veh_only": (
        ROOT
        / "case_5_veh_only"
        / "case_5_veh_only.csv"
    ),
    "case_6_gas_cap_schilthuis": (
        ROOT
        / "case_6_gas_cap_schilthuis"
        / "case_6_gas_cap_schilthuis.csv"
    ),
    "case_7_gas_cap_fetkovich": (
        ROOT
        / "case_7_gas_cap_fetkovich"
        / "case_7_gas_cap_fetkovich.csv"
    ),
    "case_8_gas_cap_veh": (
        ROOT
        / "case_8_gas_cap_veh"
        / "case_8_gas_cap_veh.csv"
    ),
}


def load_case(case_name):
    path = CASE_FILES[case_name]

    if not path.exists():
        raise FileNotFoundError(
            f"{case_name}: dataset not found:\n{path}"
        )

    df = pd.read_csv(path)

    required = [
        "F_rb",
        "Eo_rb_per_STB",
        "Eg_rb_per_STB",
        "Efw_base_rb_per_STB",
        "Efw_rb_per_STB",
        "Days",
        "Pressure_psia",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{case_name}: missing required columns: {missing}"
        )

    return df


def build_selector(df):
    return ReservoirModelSelector(
        F_rb=df["F_rb"].to_numpy(dtype=float),
        Eo_rb_per_STB=df["Eo_rb_per_STB"].to_numpy(dtype=float),
        Eg_rb_per_STB=df["Eg_rb_per_STB"].to_numpy(dtype=float),
        Efw_rb_per_STB=df["Efw_base_rb_per_STB"].to_numpy(dtype=float),
        time_days=df["Days"].to_numpy(dtype=float),
        pressure_psia=df["Pressure_psia"].to_numpy(dtype=float),
        model_configs=MODEL_CONFIGS,
    )


# ======================================================================
# CONSTRAINT VALIDATION
# ======================================================================

def test_constraint_validation():
    print("=" * 70)
    print("MODEL SELECTION CONSTRAINT VALIDATION")
    print("=" * 70)

    automatic = ModelSelectionConstraints()
    normalized = automatic.normalized()

    assert normalized.is_fully_automatic
    assert normalized.gas_cap_present is None
    assert normalized.aquifer_present is None
    assert normalized.aquifer_model is None

    print("✓ Fully automatic constraints accepted.")

    gas_present = ModelSelectionConstraints(
        gas_cap_present=True
    ).normalized()

    assert gas_present.gas_cap_present is True
    print("✓ Gas-cap-present constraint accepted.")

    gas_absent = ModelSelectionConstraints(
        gas_cap_present=False
    ).normalized()

    assert gas_absent.gas_cap_present is False
    print("✓ Gas-cap-absent constraint accepted.")

    aquifer_present = ModelSelectionConstraints(
        aquifer_present=True
    ).normalized()

    assert aquifer_present.aquifer_present is True
    print("✓ Aquifer-present constraint accepted.")

    aquifer_absent = ModelSelectionConstraints(
        aquifer_present=False
    ).normalized()

    assert aquifer_absent.aquifer_present is False
    print("✓ Aquifer-absent constraint accepted.")

    veh = ModelSelectionConstraints(
        aquifer_model="VEH"
    ).normalized()

    assert veh.aquifer_present is True
    assert veh.aquifer_model == "van_everdingen_hurst"

    print("✓ Aquifer-model alias normalization passed.")

    try:
        ModelSelectionConstraints(
            aquifer_present=False,
            aquifer_model="fetkovich",
        ).normalized()
    except ValueError:
        print("✓ Contradictory aquifer constraints rejected.")
    else:
        raise AssertionError(
            "Contradictory aquifer constraints were accepted."
        )

    print("✓ Constraint validation PASSED.")


# ======================================================================
# RESULT STRUCTURE
# ======================================================================

def test_result_structure(case_name="case_1_no_gas_no_aquifer"):
    print()
    print("=" * 70)
    print("MODEL SELECTION RESULT STRUCTURE")
    print("=" * 70)

    df = load_case(case_name)
    selector = build_selector(df)

    result = selector.select(
        constraints=ModelSelectionConstraints(
            gas_cap_present=False,
            aquifer_present=False,
        )
    )

    assert isinstance(result, ModelSelectionResult)
    assert isinstance(result.identifiability, IdentifiabilityAssessment)

    assert result.best_case == "No Gas Cap + No Aquifer"
    assert np.isfinite(result.best_rmse_rb)
    assert np.isfinite(result.best_ooip_stb)
    assert np.isfinite(result.best_r2)

    assert not result.comparison.empty
    assert not result.possible_cases.empty
    assert not result.competitive_results.empty

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
        "Competitive",
        "Rank",
    }

    assert required_columns.issubset(
        set(result.comparison.columns)
    )

    print("✓ ModelSelectionResult returned.")
    print("✓ IdentifiabilityAssessment returned.")
    print("✓ Comparison table structure passed.")
    print("✓ Possible-cases table exposed.")
    print("✓ Competitive-candidates table exposed.")
    print("✓ Result structure PASSED.")


# ======================================================================
# USER CONSTRAINT FILTERING
# ======================================================================

def test_constraint_filtering():
    print()
    print("=" * 70)
    print("USER CONSTRAINT FILTERING")
    print("=" * 70)

    # Each constraint is tested against a physically compatible
    # synthetic dataset. We must NOT ask for a gas cap in Case 1,
    # because Case 1 was deliberately generated with no gas cap.

    # --------------------------------------------------------------
    # No gas + no aquifer
    # --------------------------------------------------------------
    df = load_case("case_1_no_gas_no_aquifer")
    selector = build_selector(df)

    result = selector.select(
        constraints=ModelSelectionConstraints(
            gas_cap_present=False,
            aquifer_present=False,
        )
    )

    comparison = result.comparison

    assert comparison["Gas_Cap"].eq(False).all()
    assert comparison["Aquifer"].eq(False).all()
    assert comparison["Aquifer_Model"].eq("None").all()
    assert result.best_case == "No Gas Cap + No Aquifer"

    print("✓ No-gas/no-aquifer constraint filters candidates.")

    # --------------------------------------------------------------
    # Gas cap only
    # --------------------------------------------------------------
    df = load_case("case_2_gas_cap_only")
    selector = build_selector(df)

    result = selector.select(
        constraints=ModelSelectionConstraints(
            gas_cap_present=True,
            aquifer_present=False,
        )
    )

    comparison = result.comparison

    assert not comparison.empty
    assert comparison["Gas_Cap"].eq(True).all()
    assert comparison["Aquifer"].eq(False).all()
    assert comparison["Aquifer_Model"].eq("None").all()

    print("✓ Gas-cap/no-aquifer constraint filters candidates.")

    # --------------------------------------------------------------
    # Aquifer + specific model
    # --------------------------------------------------------------
    df = load_case("case_4_fetkovich_only")
    selector = build_selector(df)

    result = selector.select(
        constraints=ModelSelectionConstraints(
            gas_cap_present=False,
            aquifer_present=True,
            aquifer_model="fetkovich",
        )
    )

    comparison = result.comparison

    assert not comparison.empty
    assert comparison["Gas_Cap"].eq(False).all()
    assert comparison["Aquifer"].eq(True).all()
    assert comparison["Aquifer_Model"].eq(
        "Fetkovich Finite Aquifer"
    ).all()

    print("✓ Specific aquifer-model constraint filters candidates.")

    # --------------------------------------------------------------
    # Gas cap + specific aquifer
    # --------------------------------------------------------------
    df = load_case("case_6_gas_cap_schilthuis")
    selector = build_selector(df)

    result = selector.select(
        constraints=ModelSelectionConstraints(
            gas_cap_present=True,
            aquifer_present=True,
            aquifer_model="schilthuis",
        )
    )

    comparison = result.comparison

    assert not comparison.empty
    assert comparison["Gas_Cap"].eq(True).all()
    assert comparison["Aquifer"].eq(True).all()
    assert comparison["Aquifer_Model"].eq(
        "Schilthuis Steady-State"
    ).all()

    print("✓ Gas-cap + specific aquifer constraint filters candidates.")

    print("✓ User constraint filtering PASSED.")


# ======================================================================
# AUTOMATIC MODE
# ======================================================================

def test_automatic_candidate_space():
    print()
    print("=" * 70)
    print("AUTOMATIC MODEL-SELECTION CANDIDATE SPACE")
    print("=" * 70)

    df = load_case("case_1_no_gas_no_aquifer")
    selector = build_selector(df)

    result = selector.select(
        constraints=ModelSelectionConstraints()
    )

    comparison = result.comparison

    successful = comparison[
        comparison["Success"].astype(bool)
    ]

    # In automatic mode the engine should expose the full 8-model
    # candidate space, even if some individual candidates fail.
    assert len(comparison) == 8

    configurations = set(
        comparison["Configuration"].astype(str)
    )

    assert "No Gas Cap + No Aquifer" in configurations
    assert "Gas Cap + No Aquifer" in configurations
    assert "No Gas Cap + Aquifer" in configurations
    assert "Gas Cap + Aquifer" in configurations

    aquifer_models = set(
        comparison.loc[
            comparison["Aquifer"].astype(bool),
            "Aquifer_Model",
        ].astype(str)
    )

    expected_aquifer_models = {
        "Schilthuis Steady-State",
        "Fetkovich Finite Aquifer",
        "Van Everdingen-Hurst",
    }

    assert expected_aquifer_models.issubset(aquifer_models)

    assert len(successful) >= 1
    assert result.best_result.success

    print("✓ Full 8-candidate space evaluated.")
    print("✓ All four physical configuration families exposed.")
    print("✓ All three aquifer model types exposed.")
    print("✓ At least one successful physical candidate found.")
    print("✓ Automatic candidate-space test PASSED.")


# ======================================================================
# POSSIBLE + COMPETITIVE CASES
# ======================================================================

def test_possible_and_competitive_candidates():
    print()
    print("=" * 70)
    print("POSSIBLE / COMPETITIVE CANDIDATE EXPOSURE")
    print("=" * 70)

    df = load_case("case_1_no_gas_no_aquifer")
    selector = build_selector(df)

    result = selector.select(
        constraints=ModelSelectionConstraints()
    )

    possible = result.possible_cases
    competitive = result.competitive_results

    assert not possible.empty
    assert not competitive.empty

    assert set(
        competitive["Configuration"].astype(str)
    ).issubset(
        set(possible["Configuration"].astype(str))
    )

    best_rmse = possible["RMSE_rb"].min()

    assert (
        competitive["RMSE_rb"]
        <= (
            best_rmse * (1.0 + selector.DEFAULT_RELATIVE_TOLERANCE)
            + selector.DEFAULT_ABSOLUTE_TOLERANCE_RB
        )
    ).all()

    assert result.identifiability.status in {
        "UNIQUE",
        "AMBIGUOUS",
    }

    print(f"✓ Successful candidates exposed : {len(possible)}")
    print(f"✓ Competitive candidates exposed: {len(competitive)}")
    print(
        f"✓ Identifiability status        : "
        f"{result.identifiability.status}"
    )
    print("✓ Candidate exposure PASSED.")


# ======================================================================
# ALL SYNTHETIC CASES - STRUCTURAL CHECK
# ======================================================================

def test_all_8_synthetic_cases():
    print()
    print("=" * 70)
    print("ALL 8 SYNTHETIC CASES - MODEL SELECTION LAYER")
    print("=" * 70)

    for case_name in CASE_FILES:
        df = load_case(case_name)
        selector = build_selector(df)

        result = selector.select(
            constraints=ModelSelectionConstraints()
        )

        assert isinstance(result, ModelSelectionResult)
        assert len(result.comparison) == 8

        successful = result.comparison[
            result.comparison["Success"].astype(bool)
        ]

        assert not successful.empty
        assert result.best_result.success
        assert np.isfinite(result.best_rmse_rb)

        best_aquifer_model = "None"
        if result.best_case in {
            "No Gas Cap + Aquifer",
            "Gas Cap + Aquifer",
        }:
            best_aquifer_model = str(
                result.comparison.loc[
                    result.comparison["Configuration"] == result.best_case,
                    "Aquifer_Model",
                ].iloc[0]
            )

        best_display = result.best_case
        if best_aquifer_model != "None":
            best_display = (
                f"{result.best_case} "
                f"[{best_aquifer_model}]"
            )

        print(
            f"✓ {case_name}: "
            f"{len(successful)}/8 successful candidates, "
            f"best = {best_display}"
        )

    print("✓ All 8 synthetic cases passed the analysis-layer checks.")


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("PyMBAL - MODEL SELECTION LAYER VALIDATION")
    print("=" * 70)
    print()
    print(
        "This test validates the analysis layer, not the underlying "
        "reservoir physics."
    )
    print()

    test_constraint_validation()
    test_result_structure()
    test_constraint_filtering()
    test_automatic_candidate_space()
    test_possible_and_competitive_candidates()
    test_all_8_synthetic_cases()

    print()
    print("=" * 70)
    print("✓ ALL MODEL SELECTION LAYER CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
