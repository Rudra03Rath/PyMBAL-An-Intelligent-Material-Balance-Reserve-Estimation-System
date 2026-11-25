"""
PyMBAL - VOLUMETRIC ANALYSIS LAYER VALIDATION

Validates OOIP/GIIP calculations above the already validated
model-selection and coupled calibration layers.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.model_selection import (
    ModelSelectionConstraints,
    ReservoirModelSelector,
)
from src.analysis.volumetrics import (
    VolumetricAnalyzer,
    VolumetricResult,
    estimate_volumetrics,
)


ROOT = Path("data/sample_data/synthetic_8_cases")

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

CASE_FILES = {
    "case_1_no_gas_no_aquifer": ROOT / "case_1_no_gas_no_aquifer" / "case_1_no_gas_no_aquifer.csv",
    "case_2_gas_cap_only": ROOT / "case_2_gas_cap_only" / "case_2_gas_cap_only.csv",
    "case_3_schilthuis_only": ROOT / "case_3_schilthuis_only" / "case_3_schilthuis_only.csv",
    "case_4_fetkovich_only": ROOT / "case_4_fetkovich_only" / "case_4_fetkovich_only.csv",
    "case_5_veh_only": ROOT / "case_5_veh_only" / "case_5_veh_only.csv",
    "case_6_gas_cap_schilthuis": ROOT / "case_6_gas_cap_schilthuis" / "case_6_gas_cap_schilthuis.csv",
    "case_7_gas_cap_fetkovich": ROOT / "case_7_gas_cap_fetkovich" / "case_7_gas_cap_fetkovich.csv",
    "case_8_gas_cap_veh": ROOT / "case_8_gas_cap_veh" / "case_8_gas_cap_veh.csv",
}


def load_case(case_name):
    path = CASE_FILES[case_name]
    if not path.exists():
        raise FileNotFoundError(f"{case_name}: dataset not found:\n{path}")

    df = pd.read_csv(path)

    required = [
        "F_rb",
        "Eo_rb_per_STB",
        "Eg_rb_per_STB",
        "Efw_base_rb_per_STB",
        "Efw_rb_per_STB",
        "Days",
        "Pressure_psia",
        "Bo_rb_per_STB",
        "Bg_rb_per_scf",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{case_name}: missing required columns: {missing}")

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


def get_initial_pvt(df):
    return (
        float(df["Bo_rb_per_STB"].iloc[0]),
        float(df["Bg_rb_per_scf"].iloc[0]),
    )


def test_result_structure():
    print("=" * 70)
    print("VOLUMETRIC RESULT STRUCTURE")
    print("=" * 70)

    df = load_case("case_1_no_gas_no_aquifer")
    selector = build_selector(df)

    selection = selector.select(
        constraints=ModelSelectionConstraints(
            gas_cap_present=False,
            aquifer_present=False,
        )
    )

    Boi, Bgi = get_initial_pvt(df)
    result = estimate_volumetrics(selection, Boi, Bgi)

    assert isinstance(result, VolumetricResult)
    assert isinstance(result.candidates, pd.DataFrame)

    assert np.isfinite(result.best_ooip_stb)
    assert np.isfinite(result.best_giip_scf)

    required = {
        "Configuration",
        "Aquifer_Model",
        "Gas_Cap",
        "Aquifer",
        "OOIP_STB",
        "OOIP_MMSTB",
        "Gas_Cap_Ratio_m",
        "GIIP_scf",
        "GIIP_Bscf",
        "Competitive",
        "RMSE_rb",
        "R2",
    }
    assert required.issubset(result.candidates.columns)

    print("✓ VolumetricResult returned.")
    print("✓ Candidate volumetric table exposed.")
    print("✓ OOIP/GIIP result fields exposed.")
    print("✓ Result structure PASSED.")


def test_ooip_passthrough():
    print()
    print("=" * 70)
    print("OOIP CONSISTENCY")
    print("=" * 70)

    df = load_case("case_1_no_gas_no_aquifer")
    selector = build_selector(df)
    selection = selector.select(
        constraints=ModelSelectionConstraints(
            gas_cap_present=False,
            aquifer_present=False,
        )
    )

    Boi, Bgi = get_initial_pvt(df)
    result = VolumetricAnalyzer(selection, Boi, Bgi).analyze()

    assert np.isclose(
        result.best_ooip_stb,
        selection.best_ooip_stb,
        rtol=1e-12,
        atol=1e-6,
    )

    print("✓ Volumetric OOIP matches the selected material-balance OOIP.")
    print("✓ OOIP consistency PASSED.")


def test_giip_equation():
    print()
    print("=" * 70)
    print("GIIP EQUATION")
    print("=" * 70)

    df = load_case("case_2_gas_cap_only")
    selector = build_selector(df)
    selection = selector.select(
        constraints=ModelSelectionConstraints(
            gas_cap_present=True,
            aquifer_present=False,
        )
    )

    Boi, Bgi = get_initial_pvt(df)
    analyzer = VolumetricAnalyzer(selection, Boi, Bgi)

    N = 10_000_000.0
    m = 0.05

    expected = m * N * Boi / Bgi
    actual = analyzer.calculate_giip(N, m, True)

    assert np.isclose(actual, expected, rtol=1e-12, atol=1e-6)

    print("✓ GIIP = m × N × Boi / Bgi verified.")
    print("✓ GIIP equation PASSED.")


def test_no_gas_cap_giip_zero():
    print()
    print("=" * 70)
    print("NO-GAS-CAP GIIP")
    print("=" * 70)

    df = load_case("case_1_no_gas_no_aquifer")
    selector = build_selector(df)
    selection = selector.select(
        constraints=ModelSelectionConstraints(
            gas_cap_present=False,
            aquifer_present=False,
        )
    )

    Boi, Bgi = get_initial_pvt(df)
    analyzer = VolumetricAnalyzer(selection, Boi, Bgi)

    assert analyzer.calculate_giip(
        10_000_000.0,
        0.0,
        False,
    ) == 0.0

    result = analyzer.analyze()
    assert result.best_giip_scf == 0.0
    assert result.gas_cap_status == "Absent"

    print("✓ No-gas-cap interpretation returns GIIP = 0.")
    print("✓ No-gas-cap GIIP handling PASSED.")


def test_candidate_table_all_8():
    print()
    print("=" * 70)
    print("ALL 8 SYNTHETIC CASES - VOLUMETRIC LAYER")
    print("=" * 70)

    for case_name, path in CASE_FILES.items():
        df = load_case(case_name)
        selector = build_selector(df)

        selection = selector.select(
            constraints=ModelSelectionConstraints()
        )

        Boi, Bgi = get_initial_pvt(df)
        result = estimate_volumetrics(selection, Boi, Bgi)

        # Explicitly verify that GIIP is calculated and exposed
        # for downstream UI/reporting layers.
        assert np.isfinite(result.best_giip_scf)
        assert "GIIP_scf" in result.candidates.columns
        assert "GIIP_Bscf" in result.candidates.columns

        successful = result.candidates[
            result.candidates["Success"].astype(bool)
        ]

        assert len(successful) >= 1
        assert np.all(np.isfinite(successful["OOIP_STB"]))
        assert np.all(successful["OOIP_STB"] > 0.0)

        gas_rows = successful[
            successful["Gas_Cap"].astype(bool)
        ]

        no_gas_rows = successful[
            ~successful["Gas_Cap"].astype(bool)
        ]

        # A gas-cap candidate must have positive GIIP only when
        # its calibrated gas-cap ratio m is physically positive.
        physical_gas_rows = gas_rows[
            gas_rows["Gas_Cap_Ratio_m"] > 1.0e-6
        ]

        degenerate_gas_rows = gas_rows[
            gas_rows["Gas_Cap_Ratio_m"] <= 1.0e-6
        ]

        if not physical_gas_rows.empty:
            assert np.all(
                physical_gas_rows["GIIP_scf"].to_numpy(dtype=float) > 0.0
            )

        if not degenerate_gas_rows.empty:
            degenerate_giip = degenerate_gas_rows["GIIP_scf"].to_numpy(dtype=float)

            assert np.all(np.isfinite(degenerate_giip))
            assert np.all(degenerate_giip >= 0.0)

            # These candidates have m <= 1e-6, so their calculated GIIP
            # should be negligible compared with reservoir-scale GIIP.
            assert np.all(degenerate_giip < 1.0e5)

        if not no_gas_rows.empty:
            assert np.allclose(
                no_gas_rows["GIIP_scf"].to_numpy(dtype=float),
                0.0,
                atol=1.0e-10,
            )

        print(
            f"✓ {case_name}: "
            f"{len(successful)} successful volumetric candidates, "
            f"best OOIP = {result.ooip_mmstb:.4f} MMSTB, "
            f"best GIIP = {result.giip_bscf:.4f} Bscf"
        )

    print("✓ All 8 synthetic cases passed volumetric checks.")


def test_competitive_range():
    print()
    print("=" * 70)
    print("COMPETITIVE OOIP / GIIP RANGE")
    print("=" * 70)

    df = load_case("case_1_no_gas_no_aquifer")
    selector = build_selector(df)

    selection = selector.select(
        constraints=ModelSelectionConstraints()
    )

    Boi, Bgi = get_initial_pvt(df)
    result = estimate_volumetrics(selection, Boi, Bgi)

    assert result.competitive_ooip_min_stb <= result.competitive_ooip_max_stb

    if (
        result.competitive_giip_min_scf is not None
        and result.competitive_giip_max_scf is not None
    ):
        assert (
            result.competitive_giip_min_scf
            <= result.competitive_giip_max_scf
        )

    competitive = result.candidates[
        result.candidates["Competitive"].astype(bool)
    ]

    assert not competitive.empty

    print(
        "✓ Competitive OOIP range: "
        f"{result.competitive_ooip_min_stb / 1e6:.4f}–"
        f"{result.competitive_ooip_max_stb / 1e6:.4f} MMSTB"
    )

    if result.competitive_giip_range_bscf is None:
        print("✓ No competitive gas-cap candidate; GIIP range is N/A.")
    else:
        lo, hi = result.competitive_giip_range_bscf
        print(f"✓ Competitive GIIP range: {lo:.4f}–{hi:.4f} Bscf")

    print("✓ Competitive volumetric range PASSED.")


def test_input_validation():
    print()
    print("=" * 70)
    print("VOLUMETRIC INPUT VALIDATION")
    print("=" * 70)

    df = load_case("case_1_no_gas_no_aquifer")
    selector = build_selector(df)
    selection = selector.select(
        constraints=ModelSelectionConstraints(
            gas_cap_present=False,
            aquifer_present=False,
        )
    )

    try:
        VolumetricAnalyzer(selection, 0.0, 0.00082)
    except ValueError:
        pass
    else:
        raise AssertionError("Zero Boi was accepted.")

    try:
        VolumetricAnalyzer(selection, 1.42, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Zero Bgi was accepted.")

    print("✓ Invalid initial PVT values rejected.")
    print("✓ Input validation PASSED.")


def main():
    print("=" * 70)
    print("PyMBAL - VOLUMETRIC ANALYSIS LAYER VALIDATION")
    print("=" * 70)
    print()
    print(
        "This test validates OOIP/GIIP interpretation above the "
        "already validated model-selection and calibration layers."
    )

    test_result_structure()
    test_ooip_passthrough()
    test_giip_equation()
    test_no_gas_cap_giip_zero()
    test_competitive_range()
    test_input_validation()
    test_candidate_table_all_8()

    print()
    print("=" * 70)
    print("✓ ALL VOLUMETRIC ANALYSIS CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
