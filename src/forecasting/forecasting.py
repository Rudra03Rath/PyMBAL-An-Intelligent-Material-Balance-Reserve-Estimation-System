"""
PyMBAL - Material-Balance Forecasting Engine

Forecasting layer above the already validated model-selection and
volumetric-analysis layers.

This module does NOT:
    - recalibrate reservoir models
    - perform model selection
    - invent future PVT properties
    - perform production-rate forecasting

Instead, it accepts a user-specified future pressure/PVT path and
projects the material-balance reservoir state using the selected
OOIP and gas-cap ratio.

Core equation:

    F_future = We_future
               + N [Eo_future + m Eg_future + (1 + m) Efw_future]

This is deliberately separated from production-rate forecasting.
A later Forecast Scenarios / DCA layer can convert reservoir-state
forecasts into production forecasts when sufficient production/PVT
information is available.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.analysis.model_selection import ModelSelectionResult


# ======================================================================
# RESULT CONTAINER
# ======================================================================

@dataclass
class ForecastingResult:
    """
    Stable forecasting result for app.py, reporting and Streamlit.
    """

    historical_history: pd.DataFrame
    forecast_history: pd.DataFrame

    best_ooip_stb: float
    gas_cap_ratio_m: float

    forecast_points: int
    forecast_horizon_days: float

    initial_pressure_psia: float
    final_forecast_pressure_psia: float

    initial_f_rb: float
    final_forecast_f_rb: float

    forecast_status: str
    warnings: tuple

    @property
    def pressure_decline_psia(self) -> float:
        """Total pressure decline over the forecast period."""
        return float(
            self.initial_pressure_psia
            - self.final_forecast_pressure_psia
        )

    @property
    def pressure_decline_percent(self) -> float:
        """Pressure decline as a percentage of initial pressure."""
        if self.initial_pressure_psia <= 0.0:
            return np.nan

        return float(
            100.0
            * self.pressure_decline_psia
            / self.initial_pressure_psia
        )

    @property
    def f_increase_rb(self) -> float:
        """Increase in underground withdrawal over the forecast."""
        return float(
            self.final_forecast_f_rb
            - self.initial_f_rb
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "best_ooip_stb": self.best_ooip_stb,
            "gas_cap_ratio_m": self.gas_cap_ratio_m,
            "forecast_points": self.forecast_points,
            "forecast_horizon_days": self.forecast_horizon_days,
            "initial_pressure_psia": self.initial_pressure_psia,
            "final_forecast_pressure_psia": self.final_forecast_pressure_psia,
            "pressure_decline_psia": self.pressure_decline_psia,
            "pressure_decline_percent": self.pressure_decline_percent,
            "initial_f_rb": self.initial_f_rb,
            "final_forecast_f_rb": self.final_forecast_f_rb,
            "f_increase_rb": self.f_increase_rb,
            "forecast_status": self.forecast_status,
            "warnings": list(self.warnings),
        }


# ======================================================================
# FORECASTING ENGINE
# ======================================================================

class MaterialBalanceForecaster:
    """
    Project future material-balance reservoir state.

    Parameters
    ----------
    model_selection_result:
        Result returned by ReservoirModelSelector.select().

    historical_time_days:
        Historical time history.

    historical_pressure_psia:
        Historical reservoir pressure history.

    historical_F_rb:
        Historical underground withdrawal history.

    historical_Eo_rb_per_STB:
        Historical oil/solution-gas expansion history.

    historical_Eg_rb_per_STB:
        Historical gas-cap expansion history.

    historical_Efw_rb_per_STB:
        Historical rock + connate-water expansion history.

    historical_We_rb:
        Historical cumulative water influx history. Optional.

    Notes
    -----
    The future PVT/expansion path is supplied explicitly.

    The engine intentionally does not extrapolate Eo, Eg or Efw because
    doing so without a PVT model would be physically unjustified.
    """

    def __init__(
        self,
        model_selection_result: ModelSelectionResult,
        historical_time_days,
        historical_pressure_psia,
        historical_F_rb,
        historical_Eo_rb_per_STB,
        historical_Eg_rb_per_STB,
        historical_Efw_rb_per_STB,
        historical_We_rb=None,
    ):
        if not isinstance(
            model_selection_result,
            ModelSelectionResult,
        ):
            raise TypeError(
                "model_selection_result must be a ModelSelectionResult."
            )

        self.model_selection_result = model_selection_result

        self.time = np.asarray(
            historical_time_days,
            dtype=float,
        )

        self.pressure = np.asarray(
            historical_pressure_psia,
            dtype=float,
        )

        self.F = np.asarray(
            historical_F_rb,
            dtype=float,
        )

        self.Eo = np.asarray(
            historical_Eo_rb_per_STB,
            dtype=float,
        )

        self.Eg = np.asarray(
            historical_Eg_rb_per_STB,
            dtype=float,
        )

        self.Efw = np.asarray(
            historical_Efw_rb_per_STB,
            dtype=float,
        )

        self.We = (
            None
            if historical_We_rb is None
            else np.asarray(
                historical_We_rb,
                dtype=float,
            )
        )

        self.N = float(
            model_selection_result.best_ooip_stb
        )

        self.m = float(
            model_selection_result.best_gas_cap_ratio_m
        )

        self._validate_historical_inputs()

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def _validate_historical_inputs(self):
        arrays = {
            "historical_time_days": self.time,
            "historical_pressure_psia": self.pressure,
            "historical_F_rb": self.F,
            "historical_Eo_rb_per_STB": self.Eo,
            "historical_Eg_rb_per_STB": self.Eg,
            "historical_Efw_rb_per_STB": self.Efw,
        }

        lengths = [
            len(values)
            for values in arrays.values()
        ]

        if len(set(lengths)) != 1:
            raise ValueError(
                "All historical histories must have identical lengths."
            )

        if len(self.time) < 2:
            raise ValueError(
                "At least two historical data points are required."
            )

        for name, values in arrays.items():
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{name} contains NaN or infinite values."
                )

        if self.We is not None:
            if len(self.We) != len(self.time):
                raise ValueError(
                    "historical_We_rb must have the same length as "
                    "the historical histories."
                )

            if not np.all(np.isfinite(self.We)):
                raise ValueError(
                    "historical_We_rb contains NaN or infinite values."
                )

            if np.any(self.We < 0.0):
                raise ValueError(
                    "historical_We_rb cannot contain negative values."
                )

        if not np.isclose(self.time[0], 0.0):
            raise ValueError(
                "Historical time must start at zero."
            )

        if np.any(np.diff(self.time) <= 0.0):
            raise ValueError(
                "Historical time must be strictly increasing."
            )

        if np.any(self.pressure <= 0.0):
            raise ValueError(
                "Historical pressure must be positive."
            )

        if np.any(self.F < 0.0):
            raise ValueError(
                "Historical F cannot be negative."
            )

        for name, values in {
            "historical_Eo_rb_per_STB": self.Eo,
            "historical_Eg_rb_per_STB": self.Eg,
            "historical_Efw_rb_per_STB": self.Efw,
        }.items():
            if np.any(values < 0.0):
                raise ValueError(
                    f"{name} cannot contain negative values."
                )

        if not np.isfinite(self.N) or self.N <= 0.0:
            raise ValueError(
                "Selected OOIP must be finite and positive."
            )

        if not np.isfinite(self.m) or self.m < 0.0:
            raise ValueError(
                "Selected gas-cap ratio m must be finite and non-negative."
            )

    # ==================================================================
    # HISTORICAL MATERIAL-BALANCE CHECK
    # ==================================================================

    def _historical_material_balance_prediction(self):
        """
        Reconstruct historical F from the selected model.

        F = We + N[Eo + mEg + (1+m)Efw]

        If historical We is unavailable, We is assumed to be zero only
        for this diagnostic reconstruction.
        """

        if self.We is None:
            We = np.zeros_like(self.F)
        else:
            We = self.We

        predicted_F = (
            We
            + self.N
            * (
                self.Eo
                + self.m * self.Eg
                + (1.0 + self.m) * self.Efw
            )
        )

        return predicted_F

    # ==================================================================
    # FUTURE INPUT VALIDATION
    # ==================================================================

    @staticmethod
    def _validate_future_inputs(
        forecast_time_days,
        forecast_pressure_psia,
        forecast_Eo_rb_per_STB,
        forecast_Eg_rb_per_STB,
        forecast_Efw_rb_per_STB,
        forecast_We_rb,
        last_historical_time,
        last_historical_pressure,
    ):
        arrays = {
            "forecast_time_days": np.asarray(
                forecast_time_days,
                dtype=float,
            ),
            "forecast_pressure_psia": np.asarray(
                forecast_pressure_psia,
                dtype=float,
            ),
            "forecast_Eo_rb_per_STB": np.asarray(
                forecast_Eo_rb_per_STB,
                dtype=float,
            ),
            "forecast_Eg_rb_per_STB": np.asarray(
                forecast_Eg_rb_per_STB,
                dtype=float,
            ),
            "forecast_Efw_rb_per_STB": np.asarray(
                forecast_Efw_rb_per_STB,
                dtype=float,
            ),
        }

        if forecast_We_rb is not None:
            arrays["forecast_We_rb"] = np.asarray(
                forecast_We_rb,
                dtype=float,
            )

        lengths = [
            len(values)
            for values in arrays.values()
        ]

        if len(set(lengths)) != 1:
            raise ValueError(
                "All forecast histories must have identical lengths."
            )

        if lengths[0] == 0:
            raise ValueError(
                "Forecast histories cannot be empty."
            )

        for name, values in arrays.items():
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{name} contains NaN or infinite values."
                )

        time = arrays["forecast_time_days"]
        pressure = arrays["forecast_pressure_psia"]

        if np.any(time <= last_historical_time):
            raise ValueError(
                "Every forecast time must be greater than the last "
                "historical time."
            )

        if np.any(np.diff(time) <= 0.0):
            raise ValueError(
                "Forecast time must be strictly increasing."
            )

        if np.any(pressure <= 0.0):
            raise ValueError(
                "Forecast pressure must be positive."
            )

        # The first forecast point must represent the first future
        # state, not a duplicate historical point.
        if np.isclose(
            time[0],
            last_historical_time,
        ):
            raise ValueError(
                "The first forecast time cannot duplicate the last "
                "historical time."
            )

        for name in (
            "forecast_Eo_rb_per_STB",
            "forecast_Eg_rb_per_STB",
            "forecast_Efw_rb_per_STB",
        ):
            if np.any(arrays[name] < 0.0):
                raise ValueError(
                    f"{name} cannot contain negative values."
                )

        if forecast_We_rb is not None:
            if np.any(arrays["forecast_We_rb"] < 0.0):
                raise ValueError(
                    "forecast_We_rb cannot contain negative values."
                )

        # A future reservoir pressure above the last historical pressure
        # is allowed mathematically, but is flagged rather than silently
        # rejected because pressure restoration can be physically possible.
        warnings = []

        if np.any(
            pressure > last_historical_pressure
        ):
            warnings.append(
                "Forecast pressure rises above the final historical "
                "pressure."
            )

        return arrays, tuple(warnings)

    # ==================================================================
    # FORECAST
    # ==================================================================

    def forecast(
        self,
        forecast_time_days,
        forecast_pressure_psia,
        forecast_Eo_rb_per_STB,
        forecast_Eg_rb_per_STB,
        forecast_Efw_rb_per_STB,
        forecast_We_rb=None,
    ):
        """
        Project future material-balance reservoir state.

        Parameters
        ----------
        forecast_time_days:
            Absolute future time values.

        forecast_pressure_psia:
            Future reservoir pressure path.

        forecast_Eo_rb_per_STB:
            Future oil/solution-gas expansion history.

        forecast_Eg_rb_per_STB:
            Future gas-cap expansion history.

        forecast_Efw_rb_per_STB:
            Future rock + connate-water expansion history.

        forecast_We_rb:
            Future cumulative aquifer influx history. If omitted,
            zero water influx is assumed.

        Returns
        -------
        ForecastingResult
        """

        arrays, warnings = self._validate_future_inputs(
            forecast_time_days=forecast_time_days,
            forecast_pressure_psia=forecast_pressure_psia,
            forecast_Eo_rb_per_STB=forecast_Eo_rb_per_STB,
            forecast_Eg_rb_per_STB=forecast_Eg_rb_per_STB,
            forecast_Efw_rb_per_STB=forecast_Efw_rb_per_STB,
            forecast_We_rb=forecast_We_rb,
            last_historical_time=float(self.time[-1]),
            last_historical_pressure=float(self.pressure[-1]),
        )

        time = arrays["forecast_time_days"]
        pressure = arrays["forecast_pressure_psia"]
        Eo = arrays["forecast_Eo_rb_per_STB"]
        Eg = arrays["forecast_Eg_rb_per_STB"]
        Efw = arrays["forecast_Efw_rb_per_STB"]

        if forecast_We_rb is None:
            We = np.zeros_like(time)
        else:
            We = arrays["forecast_We_rb"]

        # --------------------------------------------------------------
        # Material-balance forecast
        #
        # F = We + N[Eo + mEg + (1+m)Efw]
        # --------------------------------------------------------------

        F = (
            We
            + self.N
            * (
                Eo
                + self.m * Eg
                + (1.0 + self.m) * Efw
            )
        )

        # --------------------------------------------------------------
        # Forecast depletion indicator
        # --------------------------------------------------------------

        pressure_decline = (
            self.pressure[-1]
            - pressure
        )

        pressure_decline_percent = (
            100.0
            * pressure_decline
            / self.pressure[-1]
        )

        # --------------------------------------------------------------
        # Construct forecast table
        # --------------------------------------------------------------

        history = pd.DataFrame(
            {
                "Time_days": time,
                "Pressure_psia": pressure,
                "Eo_rb_per_STB": Eo,
                "Eg_rb_per_STB": Eg,
                "Efw_rb_per_STB": Efw,
                "We_rb": We,
                "F_rb": F,
                "Pressure_Decline_psia": pressure_decline,
                "Pressure_Decline_percent": (
                    pressure_decline_percent
                ),
            }
        )

        historical_We = (
            np.zeros_like(self.time)
            if self.We is None
            else self.We
        )

        historical_F_prediction = (
            historical_We
            + self.N
            * (
                self.Eo
                + self.m * self.Eg
                + (1.0 + self.m) * self.Efw
            )
        )

        historical_history = pd.DataFrame(
            {
                "Time_days": self.time,
                "Pressure_psia": self.pressure,
                "F_rb": self.F,
                "Eo_rb_per_STB": self.Eo,
                "Eg_rb_per_STB": self.Eg,
                "Efw_rb_per_STB": self.Efw,
                "We_rb": historical_We,
                "F_predicted_rb": historical_F_prediction,
            }
        )

        return ForecastingResult(
            historical_history=historical_history,
            forecast_history=history,
            best_ooip_stb=self.N,
            gas_cap_ratio_m=self.m,
            forecast_points=len(time),
            forecast_horizon_days=float(
                time[-1] - self.time[-1]
            ),
            initial_pressure_psia=float(
                self.pressure[-1]
            ),
            final_forecast_pressure_psia=float(
                pressure[-1]
            ),
            initial_f_rb=float(
                self.F[-1]
            ),
            final_forecast_f_rb=float(
                F[-1]
            ),
            forecast_status="SUCCESS",
            warnings=warnings,
        )


# ======================================================================
# CONVENIENCE FUNCTION
# ======================================================================

def forecast_material_balance(
    model_selection_result: ModelSelectionResult,
    historical_time_days,
    historical_pressure_psia,
    historical_F_rb,
    historical_Eo_rb_per_STB,
    historical_Eg_rb_per_STB,
    historical_Efw_rb_per_STB,
    forecast_time_days,
    forecast_pressure_psia,
    forecast_Eo_rb_per_STB,
    forecast_Eg_rb_per_STB,
    forecast_Efw_rb_per_STB,
    historical_We_rb=None,
    forecast_We_rb=None,
):
    """
    Convenience wrapper around MaterialBalanceForecaster.
    """

    forecaster = MaterialBalanceForecaster(
        model_selection_result=model_selection_result,
        historical_time_days=historical_time_days,
        historical_pressure_psia=historical_pressure_psia,
        historical_F_rb=historical_F_rb,
        historical_Eo_rb_per_STB=historical_Eo_rb_per_STB,
        historical_Eg_rb_per_STB=historical_Eg_rb_per_STB,
        historical_Efw_rb_per_STB=historical_Efw_rb_per_STB,
        historical_We_rb=historical_We_rb,
    )

    return forecaster.forecast(
        forecast_time_days=forecast_time_days,
        forecast_pressure_psia=forecast_pressure_psia,
        forecast_Eo_rb_per_STB=forecast_Eo_rb_per_STB,
        forecast_Eg_rb_per_STB=forecast_Eg_rb_per_STB,
        forecast_Efw_rb_per_STB=forecast_Efw_rb_per_STB,
        forecast_We_rb=forecast_We_rb,
    )


__all__ = [
    "ForecastingResult",
    "MaterialBalanceForecaster",
    "forecast_material_balance",
]