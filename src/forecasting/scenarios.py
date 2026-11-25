"""
PyMBAL - Forecast Scenario Engine

This module sits above the material-balance forecasting engine.

Responsibilities
----------------
1. Generate controlled pressure-depletion scenarios.
2. Fit the three classical Arps decline-curve models:
       - Exponential
       - Harmonic
       - Hyperbolic
3. Compare historical DCA fits.
4. Generate future production-rate forecasts.
5. Calculate cumulative production and EUR.

Important
---------
DCA is a production-behavior model.

It is intentionally kept separate from the material-balance physics
implemented in forecasting.py.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ======================================================================
# CONSTANTS
# ======================================================================

DCA_MODEL_NAMES = (
    "Exponential",
    "Harmonic",
    "Hyperbolic",
)


# ======================================================================
# DCA RESULT
# ======================================================================

@dataclass
class DCAFitResult:
    """
    Result of fitting one Arps decline model.
    """

    model_name: str
    qi: float
    Di: float
    b: float

    rmse: float
    mae: float
    r2: float

    historical_time: np.ndarray
    historical_rate: np.ndarray
    fitted_rate: np.ndarray

    forecast_time: np.ndarray
    forecast_rate: np.ndarray

    historical_cumulative: np.ndarray
    forecast_cumulative: np.ndarray

    eur: float

    success: bool
    error: Optional[str] = None

    def summary(self) -> Dict:
        return {
            "model": self.model_name,
            "qi": float(self.qi),
            "Di": float(self.Di),
            "b": float(self.b),
            "RMSE": float(self.rmse),
            "MAE": float(self.mae),
            "R2": float(self.r2),
            "EUR": float(self.eur),
            "success": bool(self.success),
            "error": self.error,
        }


# ======================================================================
# SCENARIO RESULT
# ======================================================================

@dataclass
class ForecastScenarioResult:
    """
    Collection of DCA results and pressure scenarios.
    """

    dca_results: Dict[str, DCAFitResult]
    pressure_scenarios: pd.DataFrame

    best_dca_model: str
    best_dca_result: DCAFitResult

    warnings: tuple = ()

    def comparison_table(self) -> pd.DataFrame:
        rows = []

        for model_name, result in self.dca_results.items():
            rows.append(
                {
                    "Model": model_name,
                    "qi": result.qi,
                    "Di": result.Di,
                    "b": result.b,
                    "RMSE": result.rmse,
                    "MAE": result.mae,
                    "R2": result.r2,
                    "EUR": result.eur,
                    "Success": result.success,
                }
            )

        table = pd.DataFrame(rows)

        if not table.empty:
            table = table.sort_values(
                by=["RMSE", "MAE"],
                ascending=[True, True],
            ).reset_index(drop=True)

            table.insert(
                0,
                "Rank",
                np.arange(1, len(table) + 1),
            )

        return table


# ======================================================================
# FORECAST SCENARIO ENGINE
# ======================================================================

class ForecastScenarioEngine:
    """
    Generate pressure scenarios and perform DCA-based production
    forecasting.

    Parameters
    ----------
    time_days:
        Historical production time.

    rate:
        Historical production rate corresponding to time_days.

    rate_unit:
        Descriptive unit for the production rate. It does not affect
        calculations.

    Notes
    -----
    Time must be non-negative and strictly increasing.

    Rate must be positive because Arps decline fitting is intended
    for positive production-rate histories.
    """

    def __init__(
        self,
        time_days: Sequence[float],
        rate: Sequence[float],
        rate_unit: str = "STB/day",
    ):
        self.time = np.asarray(
            time_days,
            dtype=float,
        )

        self.rate = np.asarray(
            rate,
            dtype=float,
        )

        self.rate_unit = rate_unit

        self._validate_inputs()

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def _validate_inputs(self):
        if len(self.time) != len(self.rate):
            raise ValueError(
                "time_days and rate must have identical lengths."
            )

        if len(self.time) < 5:
            raise ValueError(
                "At least five production-rate data points are "
                "required for DCA."
            )

        if not np.all(np.isfinite(self.time)):
            raise ValueError(
                "time_days contains NaN or infinite values."
            )

        if not np.all(np.isfinite(self.rate)):
            raise ValueError(
                "rate contains NaN or infinite values."
            )

        if np.any(self.time < 0.0):
            raise ValueError(
                "time_days cannot contain negative values."
            )

        if np.any(np.diff(self.time) <= 0.0):
            raise ValueError(
                "time_days must be strictly increasing."
            )

        if np.any(self.rate <= 0.0):
            raise ValueError(
                "Production rates must be positive."
            )

    # ==================================================================
    # ARPS MODELS
    # ==================================================================

    @staticmethod
    def exponential_rate(
        t,
        qi,
        Di,
    ):
        """
        Exponential decline.

        q = qi * exp(-Di*t)
        """
        return qi * np.exp(-Di * t)

    @staticmethod
    def harmonic_rate(
        t,
        qi,
        Di,
    ):
        """
        Harmonic decline.

        q = qi / (1 + Di*t)
        """
        return qi / (1.0 + Di * t)

    @staticmethod
    def hyperbolic_rate(
        t,
        qi,
        Di,
        b,
    ):
        """
        Hyperbolic decline.

        q = qi / (1 + b*Di*t)^(1/b)
        """
        return qi / np.power(
            1.0 + b * Di * t,
            1.0 / b,
        )

    # ==================================================================
    # CUMULATIVE PRODUCTION
    # ==================================================================

    @staticmethod
    def exponential_cumulative(
        t,
        qi,
        Di,
    ):
        """
        Cumulative production from exponential decline.
        """
        if Di <= 0.0:
            return qi * t

        return (
            qi
            / Di
            * (
                1.0
                - np.exp(-Di * t)
            )
        )

    @staticmethod
    def harmonic_cumulative(
        t,
        qi,
        Di,
    ):
        """
        Cumulative production from harmonic decline.
        """
        if Di <= 0.0:
            return qi * t

        return (
            qi
            / Di
            * np.log(1.0 + Di * t)
        )

    @staticmethod
    def hyperbolic_cumulative(
        t,
        qi,
        Di,
        b,
    ):
        """
        Cumulative production from hyperbolic decline.

        Handles the b -> 1 limit separately.
        """

        if Di <= 0.0:
            return qi * t

        if np.isclose(
            b,
            1.0,
            atol=1.0e-8,
        ):
            return (
                qi
                / Di
                * np.log(1.0 + Di * t)
            )

        exponent = (
            (b - 1.0)
            / b
        )

        return (
            qi
            / (
                (1.0 - b) * Di
            )
            * (
                1.0
                - np.power(
                    1.0 + b * Di * t,
                    exponent,
                )
            )
        )

    # ==================================================================
    # METRICS
    # ==================================================================

    @staticmethod
    def _metrics(
        observed,
        predicted,
    ):
        residual = observed - predicted

        rmse = float(
            np.sqrt(
                np.mean(
                    residual ** 2
                )
            )
        )

        mae = float(
            np.mean(
                np.abs(residual)
            )
        )

        ss_res = float(
            np.sum(
                residual ** 2
            )
        )

        ss_tot = float(
            np.sum(
                (
                    observed
                    - np.mean(observed)
                )
                ** 2
            )
        )

        if np.isclose(
            ss_tot,
            0.0,
        ):
            r2 = 1.0 if np.isclose(
                ss_res,
                0.0,
            ) else -np.inf
        else:
            r2 = 1.0 - (
                ss_res
                / ss_tot
            )

        return rmse, mae, float(r2)

    # ==================================================================
    # FIT ONE MODEL
    # ==================================================================

    def _fit_exponential(
        self,
        forecast_time,
    ):
        qi0 = max(
            float(self.rate[0]),
            1.0e-12,
        )

        Di0 = 1.0e-3

        popt, _ = curve_fit(
            self.exponential_rate,
            self.time,
            self.rate,
            p0=[
                qi0,
                Di0,
            ],
            bounds=(
                [
                    1.0e-12,
                    0.0,
                ],
                [
                    np.inf,
                    np.inf,
                ],
            ),
            maxfev=50000,
        )

        qi, Di = popt

        fitted = self.exponential_rate(
            self.time,
            qi,
            Di,
        )

        forecast = self.exponential_rate(
            forecast_time,
            qi,
            Di,
        )

        cumulative = (
            self.exponential_cumulative(
                forecast_time,
                qi,
                Di,
            )
        )

        rmse, mae, r2 = self._metrics(
            self.rate,
            fitted,
        )

        return DCAFitResult(
            model_name="Exponential",
            qi=float(qi),
            Di=float(Di),
            b=0.0,
            rmse=rmse,
            mae=mae,
            r2=r2,
            historical_time=self.time.copy(),
            historical_rate=self.rate.copy(),
            fitted_rate=fitted,
            forecast_time=forecast_time.copy(),
            forecast_rate=forecast,
            historical_cumulative=np.zeros_like(
                self.time
            ),
            forecast_cumulative=cumulative,
            eur=float(cumulative[-1]),
            success=True,
        )

    # ------------------------------------------------------------------

    def _fit_harmonic(
        self,
        forecast_time,
    ):
        qi0 = max(
            float(self.rate[0]),
            1.0e-12,
        )

        Di0 = 1.0e-3

        popt, _ = curve_fit(
            self.harmonic_rate,
            self.time,
            self.rate,
            p0=[
                qi0,
                Di0,
            ],
            bounds=(
                [
                    1.0e-12,
                    0.0,
                ],
                [
                    np.inf,
                    np.inf,
                ],
            ),
            maxfev=50000,
        )

        qi, Di = popt

        fitted = self.harmonic_rate(
            self.time,
            qi,
            Di,
        )

        forecast = self.harmonic_rate(
            forecast_time,
            qi,
            Di,
        )

        cumulative = (
            self.harmonic_cumulative(
                forecast_time,
                qi,
                Di,
            )
        )

        rmse, mae, r2 = self._metrics(
            self.rate,
            fitted,
        )

        return DCAFitResult(
            model_name="Harmonic",
            qi=float(qi),
            Di=float(Di),
            b=1.0,
            rmse=rmse,
            mae=mae,
            r2=r2,
            historical_time=self.time.copy(),
            historical_rate=self.rate.copy(),
            fitted_rate=fitted,
            forecast_time=forecast_time.copy(),
            forecast_rate=forecast,
            historical_cumulative=np.zeros_like(
                self.time
            ),
            forecast_cumulative=cumulative,
            eur=float(cumulative[-1]),
            success=True,
        )

    # ------------------------------------------------------------------

    def _fit_hyperbolic(
        self,
        forecast_time,
    ):
        qi0 = max(
            float(self.rate[0]),
            1.0e-12,
        )

        Di0 = 1.0e-3
        b0 = 0.5

        popt, _ = curve_fit(
            self.hyperbolic_rate,
            self.time,
            self.rate,
            p0=[
                qi0,
                Di0,
                b0,
            ],
            bounds=(
                [
                    1.0e-12,
                    0.0,
                    1.0e-6,
                ],
                [
                    np.inf,
                    np.inf,
                    1.999999,
                ],
            ),
            maxfev=100000,
        )

        qi, Di, b = popt

        fitted = self.hyperbolic_rate(
            self.time,
            qi,
            Di,
            b,
        )

        forecast = self.hyperbolic_rate(
            forecast_time,
            qi,
            Di,
            b,
        )

        cumulative = (
            self.hyperbolic_cumulative(
                forecast_time,
                qi,
                Di,
                b,
            )
        )

        rmse, mae, r2 = self._metrics(
            self.rate,
            fitted,
        )

        return DCAFitResult(
            model_name="Hyperbolic",
            qi=float(qi),
            Di=float(Di),
            b=float(b),
            rmse=rmse,
            mae=mae,
            r2=r2,
            historical_time=self.time.copy(),
            historical_rate=self.rate.copy(),
            fitted_rate=fitted,
            forecast_time=forecast_time.copy(),
            forecast_rate=forecast,
            historical_cumulative=np.zeros_like(
                self.time
            ),
            forecast_cumulative=cumulative,
            eur=float(cumulative[-1]),
            success=True,
        )

    # ==================================================================
    # PUBLIC DCA FITTING
    # ==================================================================

    def fit_dca_models(
        self,
        forecast_time_days: Sequence[float],
    ) -> Dict[str, DCAFitResult]:
        """
        Fit exponential, harmonic and hyperbolic DCA models.
        """

        forecast_time = np.asarray(
            forecast_time_days,
            dtype=float,
        )

        if len(forecast_time) == 0:
            raise ValueError(
                "forecast_time_days cannot be empty."
            )

        if not np.all(
            np.isfinite(forecast_time)
        ):
            raise ValueError(
                "forecast_time_days contains "
                "NaN or infinite values."
            )

        if np.any(
            forecast_time < self.time[-1]
        ):
            raise ValueError(
                "Forecast time cannot precede "
                "the final historical time."
            )

        if np.any(
            np.diff(forecast_time) <= 0.0
        ):
            raise ValueError(
                "Forecast time must be strictly increasing."
            )

        results = {}

        fitters = {
            "Exponential": self._fit_exponential,
            "Harmonic": self._fit_harmonic,
            "Hyperbolic": self._fit_hyperbolic,
        }

        for model_name, fitter in fitters.items():
            try:
                results[model_name] = fitter(
                    forecast_time
                )

            except Exception as exc:
                results[model_name] = DCAFitResult(
                    model_name=model_name,
                    qi=np.nan,
                    Di=np.nan,
                    b=np.nan,
                    rmse=np.inf,
                    mae=np.inf,
                    r2=-np.inf,
                    historical_time=self.time.copy(),
                    historical_rate=self.rate.copy(),
                    fitted_rate=np.full_like(
                        self.rate,
                        np.nan,
                    ),
                    forecast_time=forecast_time.copy(),
                    forecast_rate=np.full_like(
                        forecast_time,
                        np.nan,
                    ),
                    historical_cumulative=np.zeros_like(
                        self.time
                    ),
                    forecast_cumulative=np.full_like(
                        forecast_time,
                        np.nan,
                    ),
                    eur=np.nan,
                    success=False,
                    error=str(exc),
                )

        return results

    # ==================================================================
    # PRESSURE SCENARIOS
    # ==================================================================

    def generate_pressure_scenarios(
        self,
        forecast_time_days: Sequence[float],
        final_pressure_psia: float,
    ) -> pd.DataFrame:
        """
        Generate three simple pressure scenarios.

        Base:
            User-specified final pressure.

        Low depletion:
            25% of the base pressure decline.

        High depletion:
            175% of the base pressure decline.

        This method intentionally generates only pressure paths.
        PVT expansion terms should be evaluated separately.
        """

        forecast_time = np.asarray(
            forecast_time_days,
            dtype=float,
        )

        if len(forecast_time) == 0:
            raise ValueError(
                "forecast_time_days cannot be empty."
            )

        if np.any(
            np.diff(forecast_time) <= 0.0
        ):
            raise ValueError(
                "forecast_time_days must be strictly increasing."
            )

        if not np.isfinite(
            final_pressure_psia
        ):
            raise ValueError(
                "final_pressure_psia must be finite."
            )

        if final_pressure_psia <= 0.0:
            raise ValueError(
                "final_pressure_psia must be positive."
            )

        p0 = float(
            self.rate[0]
        )

        # Pressure itself is not available in this class.
        # Therefore this method is deliberately left as a generic
        # scenario utility and uses the supplied final pressure only
        # as the endpoint. A separate reservoir-pressure scenario
        # interface will later consume the actual historical pressure.
        #
        # The production-rate engine should not infer reservoir
        # pressure from production rate.

        raise NotImplementedError(
            "Pressure scenario generation requires historical "
            "reservoir-pressure data. Use the material-balance "
            "forecasting engine for pressure-based reservoir-state "
            "scenarios."
        )

    # ==================================================================
    # COMPLETE SCENARIO ANALYSIS
    # ==================================================================

    def analyze(
        self,
        forecast_time_days: Sequence[float],
    ) -> ForecastScenarioResult:
        """
        Fit all three DCA models and select the mathematical best fit.

        Selection is based on minimum RMSE.

        This is a mathematical fit ranking only. It should not be
        interpreted as an engineering recommendation.
        """

        results = self.fit_dca_models(
            forecast_time_days
        )

        successful = [
            result
            for result in results.values()
            if result.success
            and np.isfinite(result.rmse)
        ]

        if not successful:
            raise RuntimeError(
                "No DCA model could be fitted successfully."
            )

        successful.sort(
            key=lambda result: (
                result.rmse,
                result.mae,
            )
        )

        best = successful[0]

        return ForecastScenarioResult(
            dca_results=results,
            pressure_scenarios=pd.DataFrame(),
            best_dca_model=best.model_name,
            best_dca_result=best,
            warnings=(
                "DCA model selection is based on historical "
                "fit quality only. Engineering validation is "
                "required before using a model for reserves.",
            ),
        )


# ======================================================================
# CONVENIENCE FUNCTION
# ======================================================================

def compare_dca_models(
    time_days,
    rate,
    forecast_time_days,
    rate_unit="STB/day",
):
    """
    Convenience wrapper for comparing all three Arps models.
    """

    engine = ForecastScenarioEngine(
        time_days=time_days,
        rate=rate,
        rate_unit=rate_unit,
    )

    return engine.analyze(
        forecast_time_days=forecast_time_days,
    )


__all__ = [
    "DCA_MODEL_NAMES",
    "DCAFitResult",
    "ForecastScenarioResult",
    "ForecastScenarioEngine",
    "compare_dca_models",
]