"""
PyMBAL - FULL PIPELINE INTEGRATION VALIDATION

This test executes the real analytical layers in sequence on one real
synthetic reservoir case and verifies that their public result objects
remain consistent as information flows through the pipeline.

Pipeline:
    CSV
      -> Model Selection
      -> Drive Mechanism
      -> Volumetrics
      -> Material-Balance Forecasting
      -> DCA Forecast Scenarios
      -> Reserves
      -> Reporting

This is deliberately an integration test, not a collection of mocked
layer tests.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.model_selection import (
    ModelSelectionConstraints,
    ModelSelectionResult,
    ReservoirModelSelector,
)
from src.analysis.drive_mechanism import (
    DriveMechanismAnalyzer,
    DriveMechanismResult,
)
from src.analysis.volumetrics import (
    VolumetricResult,
    estimate_volumetrics,
)
from src.analysis.reserves import (
    ReserveResult,
    estimate_reserves,
)
from src.forecasting.forecasting import (
    ForecastingResult,
    MaterialBalanceForecaster,
)
from src.forecasting.scenarios import (
    ForecastScenarioEngine,
    ForecastScenarioResult,
)
from src.reporting.report import (
    PyMBALReport,
    build_report,
)


# ======================================================================
# CONFIGURATION
# ======================================================================

DATA_ROOT = Path("data/sample_data/synthetic_8_cases")

CASE_NAME = "case_6_gas_cap_schilthuis"

CASE_FILE = (
    DATA_ROOT
    / CASE_NAME
    / f"{CASE_NAME}.csv"
)

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
# DATA
# ======================================================================

def load_case():
    if not CASE_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{CASE_FILE}"
        )

    df = pd.read_csv(CASE_FILE)

    required = [
        "Days",
        "Pressure_psia",
        "F_rb",
        "Eo_rb_per_STB",
        "Eg_rb_per_STB",
        "Efw_base_rb_per_STB",
        "Efw_rb_per_STB",
        "We_rb",
        "Bo_rb_per_STB",
        "Bg_rb_per_scf",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"{CASE_NAME}: missing required columns: {missing}"
        )

    return df


# ======================================================================
# PIPELINE BUILDERS
# ======================================================================

def build_model_selection(df):
    selector = ReservoirModelSelector(
        F_rb=df["F_rb"].to_numpy(dtype=float),
        Eo_rb_per_STB=df["Eo_rb_per_STB"].to_numpy(dtype=float),
        Eg_rb_per_STB=df["Eg_rb_per_STB"].to_numpy(dtype=float),
        Efw_rb_per_STB=df["Efw_base_rb_per_STB"].to_numpy(dtype=float),
        time_days=df["Days"].to_numpy(dtype=float),
        pressure_psia=df["Pressure_psia"].to_numpy(dtype=float),
        model_configs=MODEL_CONFIGS,
    )

    return selector.select(
        constraints=ModelSelectionConstraints()
    )


def build_drive(selection, df):
    analyzer = DriveMechanismAnalyzer(
        model_selection_result=selection,
        Eo_rb_per_STB=df["Eo_rb_per_STB"].to_numpy(dtype=float),
        Eg_rb_per_STB=df["Eg_rb_per_STB"].to_numpy(dtype=float),
        Efw_rb_per_STB=df["Efw_base_rb_per_STB"].to_numpy(dtype=float),
        water_influx_rb=df["We_rb"].to_numpy(dtype=float),
        pressure_psia=df["Pressure_psia"].to_numpy(dtype=float),
        time_days=df["Days"].to_numpy(dtype=float),
    )

    return analyzer.analyze()


def build_volumetrics(selection, df):
    Boi = float(df["Bo_rb_per_STB"].iloc[0])
    Bgi = float(df["Bg_rb_per_scf"].iloc[0])

    return estimate_volumetrics(
        model_selection_result=selection,
        Boi_rb_per_STB=Boi,
        Bgi_rb_per_scf=Bgi,
    )


def build_material_balance_forecast(selection, df):
    historical_time = df["Days"].to_numpy(dtype=float)
    last_time = float(historical_time[-1])

    # A controlled future pressure/PVT path is supplied explicitly.
    # The forecasting engine intentionally does not invent future PVT.
    forecast_time = np.array(
        [
            last_time + 100.0,
            last_time + 200.0,
            last_time + 300.0,
            last_time + 400.0,
            last_time + 500.0,
        ],
        dtype=float,
    )

    last_pressure = float(df["Pressure_psia"].iloc[-1])

    forecast_pressure = np.linspace(
        last_pressure - 200.0,
        last_pressure - 1000.0,
        len(forecast_time),
    )

    last_eo = float(df["Eo_rb_per_STB"].iloc[-1])
    last_eg = float(df["Eg_rb_per_STB"].iloc[-1])
    last_efw = float(df["Efw_base_rb_per_STB"].iloc[-1])
    last_we = float(df["We_rb"].iloc[-1])

    forecast_eo = np.linspace(
        last_eo,
        last_eo * 1.20,
        len(forecast_time),
    )

    forecast_eg = np.linspace(
        last_eg,
        last_eg * 1.20,
        len(forecast_time),
    )

    forecast_efw = np.linspace(
        last_efw,
        last_efw * 1.20,
        len(forecast_time),
    )

    forecast_we = np.linspace(
        last_we,
        last_we * 1.20,
        len(forecast_time),
    )

    forecaster = MaterialBalanceForecaster(
        model_selection_result=selection,
        historical_time_days=historical_time,
        historical_pressure_psia=df["Pressure_psia"].to_numpy(dtype=float),
        historical_F_rb=df["F_rb"].to_numpy(dtype=float),
        historical_Eo_rb_per_STB=df["Eo_rb_per_STB"].to_numpy(dtype=float),
        historical_Eg_rb_per_STB=df["Eg_rb_per_STB"].to_numpy(dtype=float),
        historical_Efw_rb_per_STB=df[
            "Efw_base_rb_per_STB"
        ].to_numpy(dtype=float),
        historical_We_rb=df["We_rb"].to_numpy(dtype=float),
    )

    return forecaster.forecast(
        forecast_time_days=forecast_time,
        forecast_pressure_psia=forecast_pressure,
        forecast_Eo_rb_per_STB=forecast_eo,
        forecast_Eg_rb_per_STB=forecast_eg,
        forecast_Efw_rb_per_STB=forecast_efw,
        forecast_We_rb=forecast_we,
    )


def build_dca_forecast(df):
    """
    Build the real DCA scenario result.

    Production-rate history is a controlled synthetic production history
    because the material-balance synthetic reservoir files contain
    reservoir-state histories rather than a production-rate column.
    """
    time = df["Days"].to_numpy(dtype=float)

    # A smooth declining oil-rate history tied to the synthetic
    # reservoir time axis. This is intentionally independent from
    # material-balance calculations.
    rate = (
        1200.0
        * np.exp(-0.0009 * time)
        + 35.0
    )

    last_time = float(time[-1])

    forecast_time = np.linspace(
        last_time + 30.0,
        last_time + 1500.0,
        40,
    )

    engine = ForecastScenarioEngine(
        time_days=time,
        rate=rate,
        rate_unit="STB/day",
    )

    return engine.analyze(
        forecast_time_days=forecast_time
    ), time, rate


def build_reserves(volumetrics, dca_result, time, rate):
    return estimate_reserves(
        volumetric_result=volumetrics,
        forecast_scenario_result=dca_result,
        historical_time_days=time,
        historical_rate=rate,
    )


# ======================================================================
# INTEGRATION TEST
# ======================================================================

def test_full_pipeline():
    df = load_case()

    selection = build_model_selection(df)
    assert isinstance(selection, ModelSelectionResult)

    drive = build_drive(selection, df)
    assert isinstance(drive, DriveMechanismResult)

    volumetrics = build_volumetrics(selection, df)
    assert isinstance(volumetrics, VolumetricResult)

    material_balance_forecast = build_material_balance_forecast(
        selection,
        df,
    )
    assert isinstance(material_balance_forecast, ForecastingResult)

    dca_result, production_time, production_rate = build_dca_forecast(df)
    assert isinstance(dca_result, ForecastScenarioResult)

    reserves = build_reserves(
        volumetrics=volumetrics,
        dca_result=dca_result,
        time=production_time,
        rate=production_rate,
    )
    assert isinstance(reserves, ReserveResult)

    report = build_report(
        model_selection=selection,
        drive_mechanism=drive,
        volumetrics=volumetrics,
        forecasting=dca_result,
        reserves=reserves,
    )
    assert isinstance(report, PyMBALReport)

    # --------------------------------------------------------------
    # Cross-layer consistency
    # --------------------------------------------------------------

    assert np.isfinite(selection.best_ooip_stb)
    assert np.isclose(
        volumetrics.best_ooip_stb,
        selection.best_ooip_stb,
        rtol=1.0e-10,
        atol=1.0e-6,
    )

    assert np.isfinite(volumetrics.best_giip_scf)
    assert len(drive.ddi_history) == len(df)

    ddi_values = drive.ddi_history[
        [
            "Solution-Gas Drive",
            "Gas-Cap Expansion",
            "Aquifer Influx",
            "Rock & Connate-Water Expansion",
        ]
    ].dropna()

    assert not ddi_values.empty
    assert np.allclose(
        ddi_values.sum(axis=1).to_numpy(dtype=float),
        100.0,
        atol=1.0e-10,
    )

    assert material_balance_forecast.forecast_status == "SUCCESS"
    assert len(material_balance_forecast.forecast_history) == 5
    assert np.all(
        np.isfinite(
            material_balance_forecast.forecast_history["F_rb"]
        )
    )

    assert set(dca_result.dca_results) == {
        "Exponential",
        "Harmonic",
        "Hyperbolic",
    }

    assert dca_result.best_dca_result.success

    assert np.isfinite(reserves.best_remaining_reserves_stb)
    assert reserves.best_remaining_reserves_stb >= 0.0
    assert reserves.best_remaining_reserves_stb <= (
        volumetrics.best_ooip_stb
        - reserves.historical_np_stb
        + 1.0e-6
    )

    # --------------------------------------------------------------
    # Reporting layer must expose the same upstream values.
    # --------------------------------------------------------------

    summary = report.executive_summary()

    assert np.isclose(
        summary["ooip_stb"],
        volumetrics.best_ooip_stb,
    )

    assert np.isclose(
        summary["giip_scf"],
        volumetrics.best_giip_scf,
    )

    assert np.isclose(
        summary["remaining_reserves_stb"],
        reserves.best_remaining_reserves_stb,
    )

    assert summary["selected_dca_model"] == dca_result.best_dca_model

    assert set(report.final_ddi()) == {
        "Solution-Gas Drive",
        "Gas-Cap Expansion",
        "Aquifer Influx",
        "Rock & Connate-Water Expansion",
    }

    serialized = report.to_dict()

    assert "executive_summary" in serialized
    assert "model_selection" in serialized
    assert "drive_mechanism" in serialized
    assert "volumetrics" in serialized
    assert "forecasting" in serialized
    assert "reserves" in serialized
    assert "ddi_history" in serialized
    assert "dca_comparison" in serialized
    assert "reserve_candidates" in serialized


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("PyMBAL - FULL PIPELINE INTEGRATION VALIDATION")
    print("=" * 70)
    print()
    print(
        "This test executes the real analytical layers sequentially "
        "on one synthetic reservoir case."
    )
    print()
    print(f"Case: {CASE_NAME}")
    print()

    print("=" * 70)
    print("FULL PIPELINE")
    print("=" * 70)

    test_full_pipeline()

    print("✓ CSV → Model Selection")
    print("✓ Model Selection → Drive Mechanism")
    print("✓ Model Selection → Volumetrics")
    print("✓ Model Selection → Material-Balance Forecasting")
    print("✓ Production History → Three Arps DCA Models")
    print("✓ Volumetrics + DCA → Reserves")
    print("✓ All analytical results → Reporting")
    print()
    print("✓ Cross-layer consistency checks PASSED")
    print("✓ Structured report serialization PASSED")
    print()
    print("=" * 70)
    print("✓ ALL FULL PIPELINE INTEGRATION CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
