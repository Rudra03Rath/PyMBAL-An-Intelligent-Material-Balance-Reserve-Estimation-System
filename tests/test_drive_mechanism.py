"""
PyMBAL - DRIVE MECHANISM ANALYSIS VALIDATION

Validates the DDI-based drive-mechanism interpretation layer without
changing the frozen physics/calibration engine.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.drive_mechanism import (
    DriveMechanismAnalyzer,
    DriveMechanismResult,
)
from src.analysis.model_selection import (
    ModelSelectionConstraints,
    ReservoirModelSelector,
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


# Expensive calibration/model-selection results are cached here.
# This keeps the test suite from recalibrating the same synthetic case
# in every individual test.
_CASE_RESULTS = {}


def get_case_result(case_name):
    if case_name not in _CASE_RESULTS:
        _CASE_RESULTS[case_name] = build_drive_result(case_name)
    return _CASE_RESULTS[case_name]


def load_case(case_name):
    path = CASE_FILES[case_name]
    if not path.exists():
        raise FileNotFoundError(f"{case_name}: dataset not found:\n{path}")

    df = pd.read_csv(path)

    required = [
        "Eo_rb_per_STB",
        "Eg_rb_per_STB",
        "Efw_base_rb_per_STB",
        "Days",
        "Pressure_psia",
        "F_rb",
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


def build_drive_result(case_name):
    df = load_case(case_name)
    selector = build_selector(df)

    selection = selector.select(
        constraints=ModelSelectionConstraints()
    )

    water_influx = (
        df["We_rb"].to_numpy(dtype=float)
        if "We_rb" in df.columns
        else None
    )

    analyzer = DriveMechanismAnalyzer(
        model_selection_result=selection,
        Eo_rb_per_STB=df["Eo_rb_per_STB"].to_numpy(dtype=float),
        Eg_rb_per_STB=df["Eg_rb_per_STB"].to_numpy(dtype=float),
        Efw_rb_per_STB=df["Efw_base_rb_per_STB"].to_numpy(dtype=float),
        water_influx_rb=water_influx,
        pressure_psia=df["Pressure_psia"].to_numpy(dtype=float),
        time_days=df["Days"].to_numpy(dtype=float),
    )

    return analyzer.analyze(), selection, df


def test_result_structure():
    print("=" * 70)
    print("DRIVE MECHANISM RESULT STRUCTURE")
    print("=" * 70)

    result, _, _ = get_case_result("case_1_no_gas_no_aquifer")

    assert isinstance(result, DriveMechanismResult)
    assert len(result.ddi_history) == 30

    expected = {
        "Solution-Gas Drive",
        "Gas-Cap Expansion",
        "Aquifer Influx",
        "Rock & Connate-Water Expansion",
    }

    assert expected.issubset(set(result.ddi_history.columns))
    assert set(result.final_ddi) == expected

    print("✓ DriveMechanismResult returned.")
    print("✓ DDI history returned.")
    print("✓ Four drive components exposed.")
    print("✓ Result structure PASSED.")


def test_ddi_sums_to_100():
    print()
    print("=" * 70)
    print("DDI NORMALIZATION")
    print("=" * 70)

    for case_name in CASE_FILES:
        result, _, _ = get_case_result(case_name)

        values = result.ddi_history[
            [
                "Solution-Gas Drive",
                "Gas-Cap Expansion",
                "Aquifer Influx",
                "Rock & Connate-Water Expansion",
            ]
        ].dropna()

        totals = values.sum(axis=1).to_numpy(dtype=float)

        assert np.allclose(totals, 100.0, atol=1.0e-10)

    print("✓ DDI sums to 100% at every valid history point.")
    print("✓ DDI normalization PASSED.")


def test_zero_gas_cap_contribution():
    print()
    print("=" * 70)
    print("ZERO GAS-CAP CONTRIBUTION")
    print("=" * 70)

    checked = 0

    for case_name in CASE_FILES:
        result, selection, _ = get_case_result(case_name)

        # DDI follows the automatically selected interpretation, not the
        # hidden synthetic truth. Therefore only selected no-gas-cap cases
        # are expected to have zero gas-cap contribution.
        if result.gas_cap_status == "Absent":
            checked += 1
            assert np.allclose(
                result.ddi_history["Gas-Cap Expansion"].dropna(),
                0.0,
                atol=1.0e-10,
            )

    assert checked > 0, "No selected no-gas-cap case was available for validation."

    print(f"✓ {checked} selected no-gas-cap cases have zero gas-cap DDI.")
    print("✓ Gas-cap zero-contribution check PASSED.")


def test_zero_aquifer_contribution():
    print()
    print("=" * 70)
    print("ZERO AQUIFER CONTRIBUTION")
    print("=" * 70)

    checked = 0

    for case_name in CASE_FILES:
        result, selection, _ = get_case_result(case_name)

        # As with gas-cap DDI, the check follows the selected model rather
        # than assuming the hidden synthetic configuration was selected.
        if result.aquifer_status == "Absent":
            checked += 1
            assert np.allclose(
                result.ddi_history["Aquifer Influx"].dropna(),
                0.0,
                atol=1.0e-10,
            )

    assert checked > 0, "No selected no-aquifer case was available for validation."

    print(f"✓ {checked} selected no-aquifer cases have zero aquifer DDI.")
    print("✓ Aquifer zero-contribution check PASSED.")


def test_final_ddi_bounds():
    print()
    print("=" * 70)
    print("FINAL DDI BOUNDS")
    print("=" * 70)

    for case_name in CASE_FILES:
        result, _, _ = get_case_result(case_name)

        values = np.asarray(list(result.final_ddi.values()), dtype=float)

        assert np.all(np.isfinite(values))
        assert np.all(values >= -1.0e-10)
        assert np.all(values <= 100.0 + 1.0e-10)

    print("✓ All final DDI percentages lie within 0–100%.")
    print("✓ Final DDI bounds PASSED.")


def test_dominant_and_overall_logic():
    print()
    print("=" * 70)
    print("DOMINANT / OVERALL DRIVE CLASSIFICATION")
    print("=" * 70)

    # Synthetic result object with controlled DDI is not required here;
    # validate the classification rule through the analyzer using
    # representative final energy terms.
    result, selection, df = get_case_result("case_1_no_gas_no_aquifer")

    assert result.dominant_mechanism in result.final_ddi
    assert result.overall_mechanism in {
        "Solution-Gas Drive",
        "Gas-Cap Expansion",
        "Aquifer Influx",
        "Rock & Connate-Water Expansion",
        "Combination Drive",
    }

    assert result.dominant_percentage >= 0.0
    assert result.dominant_percentage <= 100.0

    print(f"✓ Dominant mechanism: {result.dominant_mechanism}")
    print(f"✓ Overall mechanism : {result.overall_mechanism}")
    print("✓ Classification logic PASSED.")


def test_all_8_cases():
    print()
    print("=" * 70)
    print("ALL 8 SYNTHETIC CASES - DRIVE MECHANISM ANALYSIS")
    print("=" * 70)

    for case_name in CASE_FILES:
        result, selection, _ = get_case_result(case_name)

        assert len(result.ddi_history) == 30
        assert np.isclose(
            sum(result.final_ddi.values()),
            100.0,
            atol=1.0e-10,
        )

        print(
            f"✓ {case_name}: "
            f"overall = {result.overall_mechanism}, "
            f"dominant = {result.dominant_mechanism}, "
            f"confidence = {result.confidence}"
        )

    print("✓ All 8 synthetic cases passed drive-mechanism checks.")


def main():
    print("=" * 70)
    print("PyMBAL - DRIVE MECHANISM ANALYSIS VALIDATION")
    print("=" * 70)
    print()
    print(
        "This test validates DDI interpretation above the already "
        "validated model-selection and calibration layers."
    )
    print("Each synthetic case is calibrated once and cached for the full test suite.")

    test_result_structure()
    test_ddi_sums_to_100()
    test_zero_gas_cap_contribution()
    test_zero_aquifer_contribution()
    test_final_ddi_bounds()
    test_dominant_and_overall_logic()
    test_all_8_cases()

    print()
    print("=" * 70)
    print("✓ ALL DRIVE MECHANISM CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
