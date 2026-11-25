"""
PyMBAL - Main Application

Refactored application architecture
------------------------------------

The reservoir calculations remain in the validated src/ modules.

This file is now responsible for orchestration only:

    Load
      ↓
    Validate
      ↓
    Clean
      ↓
    PVT
      ↓
    Reservoir State
      ↓
    Model Selection
      ↓
    Drive Mechanism / DDI
      ↓
    Volumetrics / OOIP / GIIP
      ↓
    Material-Balance Forecast
      ↓
    DCA Forecast Scenarios
      ↓
    Technical Reserves
      ↓
    PyMBALReport
      ↓
    Streamlit / CLI

Important:
- No reservoir physics is implemented here.
- Model-selection constraints are passed to the validated selector.
- Future PVT is explicitly parameterized rather than silently invented.
- DCA uses an actual cumulative-oil-production column when available.
- The report layer remains read-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import tempfile

import numpy as np
import pandas as pd

from src.utils.loader import DataLoader
from src.utils.validator import DataValidator
from src.utils.cleaner import DataCleaner

from src.pvt import PVTEngine

from src.core.reservoir import ReservoirModel
from src.core.material_balance import MaterialBalanceEngine
from src.core.expansion import calculate_total_expansion
from src.core.underground_withdrawal import calculate_underground_withdrawal

from src.analysis.model_selection import (
    ModelSelectionConstraints,
    ReservoirModelSelector,
)
from src.analysis.drive_mechanism import (
    DriveMechanismAnalyzer,
)
from src.analysis.volumetrics import (
    estimate_volumetrics,
)
from src.analysis.reserves import (
    estimate_reserves,
)

from src.forecasting.forecasting import (
    forecast_material_balance,
)
from src.forecasting.scenarios import (
    ForecastScenarioEngine,
)

from src.reporting.report import (
    PyMBALReport,
    build_report,
)


# ======================================================================
# PROJECT CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "reservoir_config"
    / "reservoir_parameters.json"
)

DEFAULT_FILEPATH = (
    PROJECT_ROOT
    / "data"
    / "sample_data"
    / "synthetic_8_cases"
    / "case_6_gas_cap_schilthuis"
    / "case_6_gas_cap_schilthuis.csv"
)

PVT_MODE = "HYBRID"
ALLOW_PVT_EXTRAPOLATION = True

# The selector needs fixed physical quantities for the aquifer models.
# These defaults are the same physical search configuration used by the
# validated full-pipeline test. If an aquifer section exists in the JSON,
# those values override the defaults.
DEFAULT_MODEL_CONFIGS = {
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
# PIPELINE RESULT
# ======================================================================

@dataclass
class PyMBALPipelineResult:
    """
    Complete application result.

    ``report`` is the public reporting object used by Streamlit and
    future export/reporting code.

    The additional fields preserve the detailed intermediate outputs
    required by diagnostics and visualization.
    """

    report: PyMBALReport

    dataframe: pd.DataFrame
    reservoir_state: object

    material_balance_result: pd.DataFrame
    material_balance_analysis: dict

    material_balance_forecast: object

    production_time_days: np.ndarray
    production_rate_stb_per_day: np.ndarray


# ======================================================================
# INPUT HELPERS
# ======================================================================

def _load_dataframe(filepath: str | Path) -> pd.DataFrame:
    """Load one reservoir dataset using the project's DataLoader."""

    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{path}"
        )

    return DataLoader(str(path)).load()


def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Run the existing base-data validation and cleaning workflow."""

    DataValidator.validate_base_columns(df)
    DataValidator.validate_missing_values(df)
    DataValidator.validate_negative_values(df)
    DataValidator.validate_pvt_properties(df)

    return DataCleaner.clean(df)


def _prepare_model_configs(
    reservoir: ReservoirModel,
) -> dict:
    """
    Build aquifer calibration configuration.

    Defaults are intentionally explicit. If the reservoir JSON contains
    an ``aquifer`` section, supplied values override the defaults.
    """

    configs = {
        name: dict(values)
        for name, values in DEFAULT_MODEL_CONFIGS.items()
    }

    aquifer = reservoir.config.get("aquifer", {})

    initial_pressure = float(
        reservoir.config["reservoir"]["initial_pressure_psia"]
    )

    initial_aquifer_pressure = float(
        aquifer.get(
            "aquifer_pressure_psia",
            initial_pressure,
        )
    )

    configs["schilthuis"].update(
        {
            "aquifer_pressure_psia": initial_aquifer_pressure,
            "c_bounds": tuple(
                aquifer.get(
                    "c_bounds",
                    configs["schilthuis"]["c_bounds"],
                )
            ),
        }
    )

    configs["fetkovich"].update(
        {
            "initial_aquifer_pressure_psia": float(
                aquifer.get(
                    "initial_aquifer_pressure_psia",
                    initial_aquifer_pressure,
                )
            ),
            "total_compressibility_psi_inv": float(
                aquifer.get(
                    "total_compressibility_psi_inv",
                    configs["fetkovich"][
                        "total_compressibility_psi_inv"
                    ],
                )
            ),
            "productivity_index_bounds": tuple(
                aquifer.get(
                    "productivity_index_bounds",
                    configs["fetkovich"][
                        "productivity_index_bounds"
                    ],
                )
            ),
            "aquifer_volume_bounds": tuple(
                aquifer.get(
                    "aquifer_volume_bounds",
                    configs["fetkovich"][
                        "aquifer_volume_bounds"
                    ],
                )
            ),
        }
    )

    configs["van_everdingen_hurst"].update(
        {
            "aquifer_porosity": float(
                aquifer.get(
                    "aquifer_porosity",
                    configs["van_everdingen_hurst"][
                        "aquifer_porosity"
                    ],
                )
            ),
            "aquifer_thickness_ft": float(
                aquifer.get(
                    "aquifer_thickness_ft",
                    configs["van_everdingen_hurst"][
                        "aquifer_thickness_ft"
                    ],
                )
            ),
            "total_compressibility_psi_inv": float(
                aquifer.get(
                    "total_compressibility_psi_inv",
                    configs["van_everdingen_hurst"][
                        "total_compressibility_psi_inv"
                    ],
                )
            ),
            "water_viscosity_cp": float(
                aquifer.get(
                    "water_viscosity_cp",
                    configs["van_everdingen_hurst"][
                        "water_viscosity_cp"
                    ],
                )
            ),
            "reservoir_radius_ft": float(
                aquifer.get(
                    "reservoir_radius_ft",
                    configs["van_everdingen_hurst"][
                        "reservoir_radius_ft"
                    ],
                )
            ),
            "encroachment_angle_deg": float(
                aquifer.get(
                    "encroachment_angle_deg",
                    configs["van_everdingen_hurst"][
                        "encroachment_angle_deg"
                    ],
                )
            ),
            "permeability_bounds": tuple(
                aquifer.get(
                    "permeability_bounds",
                    configs["van_everdingen_hurst"][
                        "permeability_bounds"
                    ],
                )
            ),
        }
    )

    return configs


def _build_constraints(
    gas_cap_present: Optional[bool],
    aquifer_present: Optional[bool],
    aquifer_model: Optional[str],
) -> ModelSelectionConstraints:
    """Build validated user/model-selection constraints."""

    return ModelSelectionConstraints(
        gas_cap_present=gas_cap_present,
        aquifer_present=aquifer_present,
        aquifer_model=aquifer_model,
    ).normalized()


def _derive_production_rate(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Obtain oil-rate history for DCA/reserves.

    Preferred input:
        Np_STB = cumulative oil production.

    The DCA engine works with rate, so cumulative production is
    differentiated against the supplied Days history.

    If a daily oil-rate column already exists, it is used directly.
    """

    time = df["Days"].to_numpy(dtype=float)

    if "Oil_Rate_STB_per_day" in df.columns:
        rate = df["Oil_Rate_STB_per_day"].to_numpy(dtype=float)

    elif "Qo_STB_per_day" in df.columns:
        rate = df["Qo_STB_per_day"].to_numpy(dtype=float)

    elif "Oil_Rate_STB_day" in df.columns:
        rate = df["Oil_Rate_STB_day"].to_numpy(dtype=float)

    elif "Np_STB" in df.columns:
        cumulative = df["Np_STB"].to_numpy(dtype=float)

        if len(cumulative) < 2:
            raise ValueError(
                "At least two Np_STB points are required to derive "
                "an oil-rate history for DCA."
            )

        rate = np.gradient(cumulative, time)

    else:
        raise ValueError(
            "DCA/reserves require either an oil-rate column "
            "('Oil_Rate_STB_per_day', 'Qo_STB_per_day', or "
            "'Oil_Rate_STB_day') or cumulative oil production "
            "column 'Np_STB'."
        )

    if not np.all(np.isfinite(rate)):
        raise ValueError(
            "Derived oil-rate history contains non-finite values."
        )

    # Small numerical negatives from differentiation are clipped.
    rate = np.maximum(rate, 0.0)

    if np.all(rate <= 0.0):
        raise ValueError(
            "Oil-rate history contains no positive production rate."
        )

    return time, rate


# ======================================================================
# MAIN REUSABLE PIPELINE
# ======================================================================

def run_pymbal_pipeline(
    filepath: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    pvt_mode: str = PVT_MODE,
    allow_pvt_extrapolation: bool = ALLOW_PVT_EXTRAPOLATION,
    gas_cap_present: Optional[bool] = None,
    aquifer_present: Optional[bool] = None,
    aquifer_model: Optional[str] = None,
    gas_cap_ratio_m: Optional[float] = None,
    forecast_horizon_days: float = 500.0,
    forecast_points: int = 20,
    future_pressure_decline_fraction: float = 0.25,
    future_expansion_factor: float = 1.20,
) -> PyMBALPipelineResult:
    """
    Execute the complete PyMBAL analytical pipeline.

    Parameters
    ----------
    filepath:
        Reservoir/material-balance dataset.

    config_path:
        Reservoir configuration JSON.

    pvt_mode:
        ``laboratory``, ``correlation`` or ``hybrid``.

    gas_cap_present / aquifer_present / aquifer_model:
        Independent user knowledge. ``None`` means unknown and leaves
        the model-selection layer fully automatic.

    forecast_horizon_days:
        Future reservoir-state forecast horizon.

    forecast_points:
        Number of future reservoir-state forecast points.

    future_pressure_decline_fraction:
        Fraction of the last historical pressure used as total future
        pressure decline.

        Example:
            0.25 -> final forecast pressure is 75% of current pressure.

    future_expansion_factor:
        Multiplier applied to the last historical expansion values at
        the forecast endpoint. Linear interpolation is used between
        the current state and the endpoint.

    Notes
    -----
    Future PVT is an explicit forecasting assumption. It is not inferred
    from DCA and is not part of the historical calibration.
    """

    # ------------------------------------------------------------------
    # 1. Load / validate / clean
    # ------------------------------------------------------------------

    df = _load_dataframe(filepath)
    df = _validate_and_clean(df)

    # ------------------------------------------------------------------
    # 2. PVT
    # ------------------------------------------------------------------

    pvt_engine = PVTEngine(
        mode=pvt_mode,
        allow_extrapolation=allow_pvt_extrapolation,
    )

    df = pvt_engine.calculate(df)

    # ------------------------------------------------------------------
    # 2.5. Underground withdrawal (F)
    #
    # F depends only on production history and pressure-dependent PVT.
    # It must be calculated BEFORE model selection because the selector
    # calibrates reservoir interpretations against F.
    #
    # F = Np*Bo + (Gp - Np*Rs)*Bg + Wp*Bw
    # ------------------------------------------------------------------

    df = calculate_underground_withdrawal(df)

    # ------------------------------------------------------------------
    # 3. Reservoir state
    # ------------------------------------------------------------------

    reservoir = ReservoirModel(config_path)

    initial_pressure = float(
        reservoir.config["reservoir"]["initial_pressure_psia"]
    )

    initial_pvt = pvt_engine.get_initial_pvt(
        df=df,
        initial_pressure=initial_pressure,
    )

    state = reservoir.get_initial_state(
        initial_pvt=initial_pvt,
    )

    # ------------------------------------------------------------------
    # 4. Expansion basis for coupled model selection
    #
    # The selector must receive an m-independent Efw history.
    # Therefore we calculate the expansion terms with m = 0 and
    # expose that Efw history separately as Efw_base_rb_per_STB.
    #
    # The selected gas-cap ratio m is NOT applied here. It is
    # determined by the model-selection layer.
    # ------------------------------------------------------------------

    expansion_basis = calculate_total_expansion(
        df,
        Boi=state.Boi_rb_per_STB,
        Bgi=state.Bgi_rb_per_scf,
        Rsi=state.Rsi_scf_per_STB,
        Pi=state.initial_pressure_psia,
        Swi=state.initial_water_saturation,
        cf=state.formation_compressibility_psi_inv,
        cw=state.water_compressibility_psi_inv,
        m=0.0,
    )

    df["Eo_rb_per_STB"] = expansion_basis[
        "Eo_rb_per_STB"
    ].to_numpy(dtype=float)

    df["Eg_rb_per_STB"] = expansion_basis[
        "Eg_rb_per_STB"
    ].to_numpy(dtype=float)

    df["Efw_base_rb_per_STB"] = expansion_basis[
        "Efw_rb_per_STB"
    ].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # 5. Model selection
    # ------------------------------------------------------------------

    # F_rb is generated above from the PVT + production history.
    # The remaining columns are the expansion histories required by
    # the coupled model selector.
    required_selection_columns = [
        "F_rb",
        "Eo_rb_per_STB",
        "Eg_rb_per_STB",
        "Efw_base_rb_per_STB",
        "Days",
        "Pressure_psia",
    ]

    missing = [
        column
        for column in required_selection_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Columns required for model selection are missing: "
            + ", ".join(missing)
        )

    model_configs = _prepare_model_configs(reservoir)

    constraints = _build_constraints(
        gas_cap_present=gas_cap_present,
        aquifer_present=aquifer_present,
        aquifer_model=aquifer_model,
    )

    selector = ReservoirModelSelector(
        F_rb=df["F_rb"].to_numpy(dtype=float),
        Eo_rb_per_STB=df[
            "Eo_rb_per_STB"
        ].to_numpy(dtype=float),
        Eg_rb_per_STB=df[
            "Eg_rb_per_STB"
        ].to_numpy(dtype=float),
        Efw_rb_per_STB=df[
            "Efw_base_rb_per_STB"
        ].to_numpy(dtype=float),
        time_days=df["Days"].to_numpy(dtype=float),
        pressure_psia=df[
            "Pressure_psia"
        ].to_numpy(dtype=float),
        model_configs=model_configs,
    )

    selection = selector.select(
        constraints=constraints,
        gas_cap_ratio_m=gas_cap_ratio_m,
    )


    # ------------------------------------------------------------------
    # Apply the selected gas-cap ratio to the reservoir state.
    #
    # The model-selection layer estimates m when it is unknown.
    # MaterialBalanceEngine subsequently recalculates the final
    # m-dependent expansion terms from reservoir_state.gas_cap_ratio.
    # Therefore the selected m must replace the configuration default
    # before MBE analysis.
    # ------------------------------------------------------------------

    selected_m = float(
        selection.best_gas_cap_ratio_m
    )

    if not np.isfinite(selected_m) or selected_m < 0.0:
        raise ValueError(
            "Model selection returned an invalid gas-cap ratio m."
        )

    state.gas_cap_ratio = selected_m

    # ------------------------------------------------------------------
    # 6. Selected calibrated water influx
    #
    # This preserves the critical connection from the existing app:
    #
    # model selection -> selected calibrated We -> MBE engine.
    # ------------------------------------------------------------------

    best_calibration = selection.best_result

    predicted_we = np.asarray(
        best_calibration.predicted_we_rb,
        dtype=float,
    )

    if len(predicted_we) != len(df):
        raise ValueError(
            "Selected calibration returned a We history with a length "
            "different from the reservoir history."
        )

    if not np.all(np.isfinite(predicted_we)):
        raise ValueError(
            "Selected calibration returned non-finite We values."
        )

    predicted_we = np.maximum(
        predicted_we,
        0.0,
    )

    df["We_rb"] = predicted_we

    time_days = df["Days"].to_numpy(dtype=float)

    if len(time_days) >= 2:
        df["Influx_Rate_rb_per_day"] = np.gradient(
            predicted_we,
            time_days,
        )
    else:
        df["Influx_Rate_rb_per_day"] = 0.0

    df["Calibrated_OOIP_STB"] = selection.best_ooip_stb

    for parameter_name, value in (
        best_calibration.aquifer_parameters.items()
    ):
        df[
            "Calibrated_" + str(parameter_name)
        ] = value

    # ------------------------------------------------------------------
    # 7. Existing core material-balance engine
    #
    # This remains in the pipeline so the refactored app preserves the
    # original MBE result and Havlena-Ode calculation.
    # ------------------------------------------------------------------

    mbe = MaterialBalanceEngine(
        reservoir_state=state,
    )

    material_balance_result, material_balance_analysis = (
        mbe.analyze(df)
    )

    # ------------------------------------------------------------------
    # 8. Drive mechanism / DDI
    # ------------------------------------------------------------------

    drive_analyzer = DriveMechanismAnalyzer(
        model_selection_result=selection,
        Eo_rb_per_STB=df[
            "Eo_rb_per_STB"
        ].to_numpy(dtype=float),
        Eg_rb_per_STB=df[
            "Eg_rb_per_STB"
        ].to_numpy(dtype=float),
        Efw_rb_per_STB=df[
            "Efw_base_rb_per_STB"
        ].to_numpy(dtype=float),
        water_influx_rb=predicted_we,
        pressure_psia=df[
            "Pressure_psia"
        ].to_numpy(dtype=float),
        time_days=time_days,
    )

    drive_result = drive_analyzer.analyze()

    # ------------------------------------------------------------------
    # 9. Volumetrics / OOIP / GIIP
    # ------------------------------------------------------------------

    volumetric_result = estimate_volumetrics(
        model_selection_result=selection,
        Boi_rb_per_STB=float(
            initial_pvt["Bo_rb_per_STB"]
        ),
        Bgi_rb_per_scf=float(
            initial_pvt["Bg_rb_per_scf"]
        ),
    )

    # ------------------------------------------------------------------
    # 10. Material-balance reservoir-state forecast
    #
    # Future PVT is explicitly defined by a simple linear endpoint
    # assumption. This is a forecasting assumption, not calibration.
    # ------------------------------------------------------------------

    if forecast_horizon_days <= 0.0:
        raise ValueError(
            "forecast_horizon_days must be positive."
        )

    if forecast_points < 2:
        raise ValueError(
            "forecast_points must be at least 2."
        )

    if not 0.0 < future_pressure_decline_fraction < 1.0:
        raise ValueError(
            "future_pressure_decline_fraction must be between 0 and 1."
        )

    if future_expansion_factor <= 0.0:
        raise ValueError(
            "future_expansion_factor must be positive."
        )

    last_time = float(time_days[-1])
    last_pressure = float(df["Pressure_psia"].iloc[-1])

    forecast_time = np.linspace(
        last_time + forecast_horizon_days / forecast_points,
        last_time + forecast_horizon_days,
        forecast_points,
    )

    final_pressure = (
        last_pressure
        * (1.0 - future_pressure_decline_fraction)
    )

    forecast_pressure = np.linspace(
        last_pressure,
        final_pressure,
        forecast_points,
    )

    last_eo = float(df["Eo_rb_per_STB"].iloc[-1])
    last_eg = float(df["Eg_rb_per_STB"].iloc[-1])
    last_efw = float(df["Efw_base_rb_per_STB"].iloc[-1])
    last_we = float(predicted_we[-1])

    forecast_eo = np.linspace(
        last_eo,
        last_eo * future_expansion_factor,
        forecast_points,
    )

    forecast_eg = np.linspace(
        last_eg,
        last_eg * future_expansion_factor,
        forecast_points,
    )

    forecast_efw = np.linspace(
        last_efw,
        last_efw * future_expansion_factor,
        forecast_points,
    )

    # If an aquifer is absent in the selected physical interpretation,
    # future We must remain zero. Otherwise use a transparent linear
    # continuation of the last calibrated cumulative We.
    if bool(selection.best_result.aquifer_present):
        forecast_we = np.linspace(
            last_we,
            last_we * future_expansion_factor,
            forecast_points,
        )
    else:
        forecast_we = np.zeros(
            forecast_points,
            dtype=float,
        )

    material_balance_forecast = forecast_material_balance(
        model_selection_result=selection,
        historical_time_days=time_days,
        historical_pressure_psia=df[
            "Pressure_psia"
        ].to_numpy(dtype=float),
        historical_F_rb=df[
            "F_rb"
        ].to_numpy(dtype=float),
        historical_Eo_rb_per_STB=df[
            "Eo_rb_per_STB"
        ].to_numpy(dtype=float),
        historical_Eg_rb_per_STB=df[
            "Eg_rb_per_STB"
        ].to_numpy(dtype=float),
        historical_Efw_rb_per_STB=df[
            "Efw_base_rb_per_STB"
        ].to_numpy(dtype=float),
        forecast_time_days=forecast_time,
        forecast_pressure_psia=forecast_pressure,
        forecast_Eo_rb_per_STB=forecast_eo,
        forecast_Eg_rb_per_STB=forecast_eg,
        forecast_Efw_rb_per_STB=forecast_efw,
        historical_We_rb=predicted_we,
        forecast_We_rb=forecast_we,
    )

    # ------------------------------------------------------------------
    # 11. Production DCA
    # ------------------------------------------------------------------

    production_time, production_rate = _derive_production_rate(df)

    dca_forecast_time = np.linspace(
        float(production_time[-1]) + 30.0,
        float(production_time[-1]) + forecast_horizon_days,
        max(20, forecast_points),
    )

    dca_engine = ForecastScenarioEngine(
        time_days=production_time,
        rate=production_rate,
        rate_unit="STB/day",
    )

    dca_result = dca_engine.analyze(
        forecast_time_days=dca_forecast_time,
    )

    # ------------------------------------------------------------------
    # 12. Technical reserves
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("DEBUG - RESERVE VALIDATION")
    print("=" * 70)

    historical_np = float(df["Np_STB"].iloc[-1])
    selected_ooip = float(volumetric_result.best_ooip_stb)

    print(f"Historical final Np_STB : {historical_np:,.2f}")
    print(f"Selected OOIP           : {selected_ooip:,.2f}")
    print(f"Np / OOIP               : {historical_np / selected_ooip:.3f}")
    print(f"Gas-cap ratio m         : {volumetric_result.best_candidate.gas_cap_ratio_m}")
    print(f"Selected configuration  : {volumetric_result.best_candidate.configuration}")
    print(f"Aquifer model           : {volumetric_result.best_candidate.aquifer_model}")
    print(f"R2                      : {volumetric_result.best_candidate.r2}")
    print("=" * 70)

    reserve_result = estimate_reserves(
        volumetric_result=volumetric_result,
        forecast_scenario_result=dca_result,
        historical_time_days=production_time,
        historical_rate=production_rate,
    )

    # ------------------------------------------------------------------
    # 12. Final report
    # ------------------------------------------------------------------

    application_warnings = []

    if selection.identifiability.ambiguous:
        application_warnings.append(
            "Multiple material-balance interpretations remain "
            "competitive. Independent reservoir information should "
            "be used before treating the mathematical best case as "
            "the definitive physical interpretation."
        )

    application_notes = [
        "Future reservoir-state PVT/pressure paths are explicit "
        "forecast assumptions and are not historical calibration results.",
        "DCA model selection is mathematical fit ranking only.",
        "Technical remaining reserves are not an economic or "
        "reserves-classification determination.",
    ]

    report = build_report(
        model_selection=selection,
        drive_mechanism=drive_result,
        volumetrics=volumetric_result,
        forecasting=dca_result,
        reserves=reserve_result,
        warnings=application_warnings,
        notes=application_notes,
    )

    return PyMBALPipelineResult(
        report=report,
        dataframe=df,
        reservoir_state=state,
        material_balance_result=material_balance_result,
        material_balance_analysis=material_balance_analysis,
        material_balance_forecast=material_balance_forecast,
        production_time_days=production_time,
        production_rate_stb_per_day=production_rate,
    )


# ======================================================================
# LEGACY COMPATIBILITY WRAPPER
# ======================================================================

def run_material_balance():
    """
    Backward-compatible CLI wrapper.

    Returns the original three core outputs:
        material_balance_result,
        material_balance_analysis,
        reservoir_state

    The complete new pipeline is also executed.
    """

    pipeline = run_pymbal_pipeline(
        filepath=DEFAULT_FILEPATH,
        config_path=DEFAULT_CONFIG_PATH,
    )

    report = pipeline.report
    summary = report.executive_summary()

    print("=" * 70)
    print(
        "PyMBAL: Intelligent Material Balance "
        "& Reserve Evaluation System"
    )
    print("=" * 70)

    print("\nFINAL PYMBAL SUMMARY")
    print("-" * 70)

    print(
        f"Configuration       : "
        f"{summary['selected_configuration']}"
    )

    print(
        f"Aquifer model       : "
        f"{summary['selected_aquifer_model'] or 'None'}"
    )

    print(
        f"Identifiability     : "
        f"{summary['identifiability_status']}"
    )

    print(
        f"Dominant drive      : "
        f"{summary['dominant_drive']} "
        f"({summary['dominant_drive_percentage']:.2f}%)"
    )

    print(
        f"Overall drive       : "
        f"{summary['overall_drive']}"
    )

    print(
        f"OOIP                : "
        f"{summary['ooip_stb']:,.2f} STB"
    )

    print(
        f"GIIP                : "
        f"{summary['giip_scf']:,.2f} scf"
    )

    print(
        f"Best DCA            : "
        f"{summary['selected_dca_model']}"
    )

    print(
        f"EUR                 : "
        f"{summary['eur_stb']:,.2f} STB"
    )

    print(
        f"Remaining technical : "
        f"{summary['remaining_reserves_stb']:,.2f} STB"
    )

    print()
    print(
        "✓ Complete PyMBAL analytical pipeline completed."
    )

    return (
        pipeline.material_balance_result,
        pipeline.material_balance_analysis,
        pipeline.reservoir_state,
    )


# ======================================================================
# STREAMLIT APPLICATION
# ======================================================================

def _render_streamlit():
    """
    Render the first integrated Streamlit UI.

    The UI is deliberately thin: all reservoir calculations happen in
    run_pymbal_pipeline().
    """

    import streamlit as st

    st.set_page_config(
        page_title="PyMBAL",
        page_icon="🛢️",
        layout="wide",
    )

    st.title(
        "PyMBAL"
    )

    st.caption(
        "Intelligent Material Balance & Reserve Evaluation System"
    )

    st.sidebar.header("Input")

    uploaded_file = st.sidebar.file_uploader(
        "Reservoir history CSV",
        type=["csv"],
    )

    config_path = st.sidebar.text_input(
        "Reservoir configuration",
        value=str(
            DEFAULT_CONFIG_PATH.relative_to(PROJECT_ROOT)
        ),
    )

    st.sidebar.header("Model Constraints")

    gas_cap_choice = st.sidebar.selectbox(
        "Gas-cap presence",
        [
            "Unknown",
            "Present",
            "Absent",
        ],
        index=0,
    )

    aquifer_choice = st.sidebar.selectbox(
        "Aquifer presence",
        [
            "Unknown",
            "Present",
            "Absent",
        ],
        index=0,
    )

    aquifer_model_choice = st.sidebar.selectbox(
        "Aquifer model",
        [
            "Unknown",
            "Schilthuis Steady-State",
            "Fetkovich Finite Aquifer",
            "Van Everdingen-Hurst",
        ],
        index=0,
    )

    st.sidebar.header("Forecast Assumptions")

    forecast_horizon = st.sidebar.number_input(
        "Forecast horizon (days)",
        min_value=30.0,
        value=500.0,
        step=30.0,
    )

    pressure_decline = st.sidebar.slider(
        "Future pressure decline fraction",
        min_value=0.05,
        max_value=0.80,
        value=0.25,
        step=0.05,
    )

    expansion_factor = st.sidebar.number_input(
        "Future expansion endpoint multiplier",
        min_value=1.01,
        max_value=5.0,
        value=1.20,
        step=0.05,
    )

    if uploaded_file is None:
        st.info(
            "Upload a reservoir-history CSV from the sidebar to run "
            "the complete PyMBAL pipeline."
        )

        st.markdown(
            """
### Pipeline

**Data → Model Selection → DDI → OOIP/GIIP → Forecast → DCA → Reserves → Report**

The model-selection layer remains automatic when the corresponding
reservoir information is left as **Unknown**.
"""
        )
        return

    def _constraint(value):
        if value == "Unknown":
            return None
        return value == "Present"

    gas_cap_present = _constraint(gas_cap_choice)
    aquifer_present = _constraint(aquifer_choice)

    gas_cap_ratio_m = st.sidebar.number_input(
        "Gas Cap Ratio (m) - optional",
        min_value=0.0,
        value=0.0,
        step=0.05,
        help="Leave as 0 for automatic estimation. Provide m only if known from reservoir studies.",
    )

    if gas_cap_ratio_m == 0.0:
        gas_cap_ratio_m = None

    if aquifer_model_choice == "Unknown":
        aquifer_model = None
    else:
        aquifer_model = aquifer_model_choice

    # A specific aquifer model itself implies aquifer presence.
    if aquifer_model is not None:
        aquifer_present = True

    # A temporary file lets the existing DataLoader remain unchanged.
    with tempfile.NamedTemporaryFile(
        suffix=".csv",
        delete=False,
    ) as temporary:
        temporary.write(
            uploaded_file.getvalue()
        )
        uploaded_path = temporary.name

    try:
        with st.spinner(
            "Running PyMBAL analytical pipeline..."
        ):
            pipeline = run_pymbal_pipeline(
                filepath=uploaded_path,
                config_path=(
                    Path(config_path)
                    if Path(config_path).is_absolute()
                    else PROJECT_ROOT / config_path
                ),
                gas_cap_present=gas_cap_present,
                aquifer_present=aquifer_present,
                aquifer_model=aquifer_model,
                gas_cap_ratio_m=gas_cap_ratio_m,
                forecast_horizon_days=forecast_horizon,
                future_pressure_decline_fraction=pressure_decline,
                future_expansion_factor=expansion_factor,
            )
    except Exception as exc:
        st.error(
            f"PyMBAL pipeline failed: {exc}"
        )
        st.exception(exc)
        return
    finally:
        try:
            Path(uploaded_path).unlink(missing_ok=True)
        except Exception:
            pass

    report = pipeline.report
    summary = report.executive_summary()

    # --------------------------------------------------------------
    # Executive metrics
    # --------------------------------------------------------------

    st.subheader("Executive Summary")

    columns = st.columns(4)

    columns[0].metric(
        "OOIP",
        f"{summary['ooip_stb'] / 1e6:.3f} MMSTB",
    )

    columns[1].metric(
        "GIIP",
        f"{summary['giip_scf'] / 1e9:.3f} Bscf",
    )

    columns[2].metric(
        "Remaining Technical Reserves",
        f"{summary['remaining_reserves_stb'] / 1e6:.3f} MMSTB",
    )

    columns[3].metric(
        "Best DCA",
        summary["selected_dca_model"],
    )

    st.divider()

    # --------------------------------------------------------------
    # Model interpretation
    # --------------------------------------------------------------

    st.subheader("Reservoir Interpretation")

    left, right = st.columns(2)

    with left:
        st.write(
            "**Selected configuration:**",
            summary["selected_configuration"],
        )

        st.write(
            "**Aquifer model:**",
            summary["selected_aquifer_model"] or "None",
        )

        st.write(
            "**Identifiability:**",
            summary["identifiability_status"],
        )

    with right:
        st.write(
            "**Dominant drive:**",
            f"{summary['dominant_drive']} "
            f"({summary['dominant_drive_percentage']:.2f}%)",
        )

        st.write(
            "**Overall drive:**",
            summary["overall_drive"],
        )

    if report.is_ambiguous:
        st.warning(
            "Multiple material-balance interpretations remain "
            "competitive. The mathematical best case should not be "
            "treated as uniquely proven without independent reservoir "
            "information."
        )

    # --------------------------------------------------------------
    # Tabs
    # --------------------------------------------------------------

    tabs = st.tabs(
        [
            "Model Selection",
            "Drive Mechanism",
            "Volumetrics",
            "Forecast",
            "DCA",
            "Reserves",
            "Report",
        ]
    )

    with tabs[0]:
        st.subheader("Possible Interpretations")
        st.dataframe(
            report.possible_candidates(),
            use_container_width=True,
        )

        st.subheader("Competitive Interpretations")
        st.dataframe(
            report.competitive_candidates(),
            use_container_width=True,
        )

    with tabs[1]:
        st.subheader("Final DDI")

        ddi = report.final_ddi()

        ddi_df = pd.DataFrame(
            {
                "Drive Mechanism": list(ddi.keys()),
                "Contribution (%)": list(ddi.values()),
            }
        )

        st.dataframe(
            ddi_df,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("DDI History")
        st.line_chart(
            report.ddi_history()
        )

    with tabs[2]:
        st.subheader("OOIP / GIIP Candidates")
        st.dataframe(
            report.volumetric_candidates(),
            use_container_width=True,
        )

        st.write(
            f"**Selected OOIP:** "
            f"{report.ooip_stb / 1e6:.4f} MMSTB"
        )

        st.write(
            f"**Selected GIIP:** "
            f"{report.giip_scf / 1e9:.4f} Bscf"
        )

    with tabs[3]:
        st.subheader("Material-Balance Reservoir-State Forecast")

        st.dataframe(
            pipeline.material_balance_forecast.forecast_history,
            use_container_width=True,
        )

        forecast_history = (
            pipeline.material_balance_forecast.forecast_history
        )

        chart_df = forecast_history[
            [
                "Time_days",
                "Pressure_psia",
            ]
        ].set_index("Time_days")

        st.line_chart(
            chart_df
        )

    with tabs[4]:
        st.subheader("DCA Model Comparison")

        st.dataframe(
            report.dca_comparison(),
            use_container_width=True,
        )

        st.write(
            f"**Selected mathematical DCA:** "
            f"{report.selected_dca_model}"
        )

        best = report.forecasting.best_dca_result

        dca_plot = pd.DataFrame(
            {
                "Historical": pd.Series(
                    best.historical_rate,
                    index=best.historical_time,
                ),
                "Fitted": pd.Series(
                    best.fitted_rate,
                    index=best.historical_time,
                ),
            }
        )

        forecast_plot = pd.DataFrame(
            {
                "Forecast": pd.Series(
                    best.forecast_rate,
                    index=best.forecast_time,
                )
            }
        )

        st.line_chart(
            pd.concat(
                [dca_plot, forecast_plot],
                axis=0,
            )
        )

    with tabs[5]:
        st.subheader("Technical Reserves")

        st.dataframe(
            report.reserve_candidates(),
            use_container_width=True,
        )

        st.write(
            f"**Historical production:** "
            f"{report.reserves.historical_np_stb / 1e6:.4f} MMSTB"
        )

        st.write(
            f"**EUR:** "
            f"{report.eur_stb / 1e6:.4f} MMSTB"
        )

        st.write(
            f"**Remaining technical reserves:** "
            f"{report.remaining_reserves_stb / 1e6:.4f} MMSTB"
        )

    with tabs[6]:
        st.subheader("Engineering Report")

        st.json(
            report.to_dict()
        )

        st.subheader("Warnings")

        warnings = report.all_warnings()

        if warnings:
            for warning in warnings:
                st.warning(warning)
        else:
            st.success("No warnings.")

        st.subheader("Notes")

        for note in report.notes:
            st.info(note)


# ======================================================================
# ENTRY POINT
# ======================================================================

def main():
    """
    Streamlit is the primary UI.

    Running ``python app.py`` remains supported for the existing CLI
    workflow and is useful for backend debugging.
    """

    try:
        import streamlit  # noqa: F401
    except ImportError:
        run_material_balance()
        return

    # Streamlit sets this environment variable when executing the app.
    import os

    if (
        os.environ.get("STREAMLIT_SERVER_PORT")
        or os.environ.get("STREAMLIT_RUNTIME")
        or "streamlit" in os.environ.get("_", "").lower()
    ):
        _render_streamlit()
    else:
        run_material_balance()


if __name__ == "__main__":
    main()
