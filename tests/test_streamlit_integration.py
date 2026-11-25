from pathlib import Path

import pandas as pd

from app import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_FILEPATH,
    run_pymbal_pipeline,
)


def main():
    print("=" * 70)
    print("PyMBAL - STREAMLIT INTEGRATION VALIDATION")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("UI BACKEND CONNECTION")
    print("=" * 70)

    assert Path(DEFAULT_FILEPATH).exists()
    assert Path(DEFAULT_CONFIG_PATH).exists()

    print("✓ Default reservoir dataset found.")
    print("✓ Reservoir configuration found.")

    print("\n" + "=" * 70)
    print("PIPELINE ACCESS")
    print("=" * 70)

    result = run_pymbal_pipeline(
        filepath=DEFAULT_FILEPATH,
        config_path=DEFAULT_CONFIG_PATH,
        forecast_horizon_days=500.0,
        forecast_points=20,
    )

    assert result is not None
    assert result.report is not None
    assert result.dataframe is not None

    print("✓ Streamlit can access the PyMBAL pipeline.")
    print("✓ Pipeline result returned.")

    print("\n" + "=" * 70)
    print("REPORT CONNECTION")
    print("=" * 70)

    report = result.report
    summary = report.executive_summary()

    required_summary_fields = [
        "selected_configuration",
        "selected_aquifer_model",
        "identifiability_status",
        "dominant_drive",
        "dominant_drive_percentage",
        "overall_drive",
        "ooip_stb",
        "giip_scf",
        "selected_dca_model",
        "eur_stb",
        "remaining_reserves_stb",
    ]

    for field in required_summary_fields:
        assert field in summary

    print("✓ Executive summary fields available.")
    print("✓ Report layer connected.")

    print("\n" + "=" * 70)
    print("TABLE CONNECTION")
    print("=" * 70)

    assert isinstance(
        report.possible_candidates(),
        pd.DataFrame,
    )

    assert isinstance(
        report.competitive_candidates(),
        pd.DataFrame,
    )

    assert isinstance(
        report.volumetric_candidates(),
        pd.DataFrame,
    )

    assert isinstance(
        report.dca_comparison(),
        pd.DataFrame,
    )

    assert isinstance(
        report.reserve_candidates(),
        pd.DataFrame,
    )

    print("✓ Model-selection tables available.")
    print("✓ Volumetric table available.")
    print("✓ DCA comparison available.")
    print("✓ Reserves table available.")

    print("\n" + "=" * 70)
    print("VISUALIZATION DATA CONNECTION")
    print("=" * 70)

    ddi_history = report.ddi_history()
    final_ddi = report.final_ddi()

    assert ddi_history is not None
    assert final_ddi is not None

    print("✓ DDI history available.")
    print("✓ Final DDI percentages available.")

    print("\n" + "=" * 70)
    print("FORECAST CONNECTION")
    print("=" * 70)

    forecast = result.material_balance_forecast

    assert forecast is not None
    assert forecast.forecast_history is not None

    print("✓ Material-balance forecast available.")

    print("\n" + "=" * 70)
    print("✓ ALL STREAMLIT INTEGRATION CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()