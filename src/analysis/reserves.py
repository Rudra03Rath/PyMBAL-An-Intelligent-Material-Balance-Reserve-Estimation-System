"""
PyMBAL - Technical Reserves Estimation Layer

Consumes the already validated volumetric and DCA/forecast-scenario
results and converts them into a transparent technical remaining
recoverable-volume estimate.

This layer deliberately does NOT:
    - perform material-balance calibration
    - perform model selection
    - fit DCA models
    - perform economic-limit calculations
    - claim SEC/PUD/proved reserves

For an oil-rate forecast, the layer uses:

    Np_hist = cumulative historical oil production

    EUR_raw = DCA cumulative production at the forecast endpoint

    Remaining_raw = max(EUR_raw - Np_hist, 0)

The raw DCA result is constrained by the selected OOIP:

    Remaining_material_balance_limit =
        max(OOIP - Np_hist, 0)

    Technical_remaining =
        min(Remaining_raw, Remaining_material_balance_limit)

This prevents a production-rate fit from implying recovery greater than
the oil originally in place. Economic and reserves-classification
criteria can be added later as a separate layer.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from src.analysis.volumetrics import VolumetricResult
from src.forecasting.scenarios import ForecastScenarioResult, DCAFitResult


@dataclass(frozen=True)
class ReserveCandidate:
    """Reserve interpretation for one DCA model."""

    model_name: str
    success: bool

    qi: float
    Di: float
    b: float

    historical_np_stb: float
    eur_raw_stb: float
    remaining_raw_stb: float

    technical_eur_stb: float
    technical_reserves_stb: float

    historical_recovery_factor_percent: float
    ultimate_recovery_factor_percent: float

    constrained_by_ooip: bool
    error: Optional[str] = None


@dataclass
class ReserveResult:
    """Stable reserves result for app.py, reporting and Streamlit."""

    best_candidate: ReserveCandidate
    candidates: pd.DataFrame

    selected_dca_model: str

    historical_np_stb: float
    best_eur_stb: float
    best_remaining_reserves_stb: float

    best_ooip_stb: float
    historical_recovery_factor_percent: float
    ultimate_recovery_factor_percent: float

    warnings: tuple = ()

    @property
    def best_eur_mmstb(self) -> float:
        return self.best_eur_stb / 1.0e6

    @property
    def best_remaining_reserves_mmstb(self) -> float:
        return self.best_remaining_reserves_stb / 1.0e6

    @property
    def best_ooip_mmstb(self) -> float:
        return self.best_ooip_stb / 1.0e6

    @property
    def is_ooip_constrained(self) -> bool:
        return bool(self.best_candidate.constrained_by_ooip)

    def summary(self) -> Dict[str, Any]:
        return {
            "selected_dca_model": self.selected_dca_model,
            "historical_np_stb": self.historical_np_stb,
            "best_eur_stb": self.best_eur_stb,
            "best_eur_mmstb": self.best_eur_mmstb,
            "best_remaining_reserves_stb": self.best_remaining_reserves_stb,
            "best_remaining_reserves_mmstb": self.best_remaining_reserves_mmstb,
            "best_ooip_stb": self.best_ooip_stb,
            "best_ooip_mmstb": self.best_ooip_mmstb,
            "historical_recovery_factor_percent": (
                self.historical_recovery_factor_percent
            ),
            "ultimate_recovery_factor_percent": (
                self.ultimate_recovery_factor_percent
            ),
            "ooip_constrained": self.is_ooip_constrained,
            "warnings": list(self.warnings),
        }


class ReservesEstimator:
    """
    Estimate technical remaining oil recovery from a ForecastScenarioResult.

    Parameters
    ----------
    volumetric_result:
        Already validated VolumetricResult containing selected OOIP.

    forecast_scenario_result:
        Already validated ForecastScenarioResult containing the three
        Arps DCA fits.

    historical_time_days:
        Historical production time.

    historical_rate:
        Historical production rate in STB/day.

    Notes
    -----
    Historical cumulative production is calculated independently with
    trapezoidal integration of the observed rate history. This is
    preferable to treating the DCA fitted cumulative as historical
    production.

    The selected DCA model is the mathematical best DCA fit already
    exposed by ForecastScenarioResult. No second DCA selection is done.
    """

    def __init__(
        self,
        volumetric_result: VolumetricResult,
        forecast_scenario_result: ForecastScenarioResult,
        historical_time_days: Sequence[float],
        historical_rate: Sequence[float],
    ):
        if not isinstance(volumetric_result, VolumetricResult):
            raise TypeError(
                "volumetric_result must be a VolumetricResult."
            )

        if not isinstance(
            forecast_scenario_result,
            ForecastScenarioResult,
        ):
            raise TypeError(
                "forecast_scenario_result must be a ForecastScenarioResult."
            )

        self.volumetric_result = volumetric_result
        self.forecast_scenario_result = forecast_scenario_result

        self.time = np.asarray(
            historical_time_days,
            dtype=float,
        )
        self.rate = np.asarray(
            historical_rate,
            dtype=float,
        )

        self.ooip = float(
            volumetric_result.best_ooip_stb
        )

        self._validate_inputs()

    def _validate_inputs(self):
        if len(self.time) != len(self.rate):
            raise ValueError(
                "historical_time_days and historical_rate must have "
                "identical lengths."
            )

        if len(self.time) < 2:
            raise ValueError(
                "At least two historical production points are required."
            )

        if not np.all(np.isfinite(self.time)):
            raise ValueError(
                "historical_time_days contains NaN or infinite values."
            )

        if not np.all(np.isfinite(self.rate)):
            raise ValueError(
                "historical_rate contains NaN or infinite values."
            )

        if np.any(self.time < 0.0):
            raise ValueError(
                "historical_time_days cannot contain negative values."
            )

        if np.any(np.diff(self.time) <= 0.0):
            raise ValueError(
                "historical_time_days must be strictly increasing."
            )

        if np.any(self.rate < 0.0):
            raise ValueError(
                "historical_rate cannot contain negative values."
            )

        if not np.isfinite(self.ooip) or self.ooip <= 0.0:
            raise ValueError(
                "Selected OOIP must be finite and positive."
            )

    @staticmethod
    def _historical_cumulative(
        time: np.ndarray,
        rate: np.ndarray,
    ) -> float:
        """Integrate observed production rate with the trapezoidal rule."""
        return float(
            np.trapezoid(rate, time)
            if hasattr(np, "trapezoid")
            else np.trapz(rate, time)
        )

    def _candidate_from_dca(
        self,
        result: DCAFitResult,
        historical_np_stb: float,
    ) -> ReserveCandidate:
        if not result.success:
            return ReserveCandidate(
                model_name=result.model_name,
                success=False,
                qi=np.nan,
                Di=np.nan,
                b=np.nan,
                historical_np_stb=historical_np_stb,
                eur_raw_stb=np.nan,
                remaining_raw_stb=np.nan,
                technical_eur_stb=np.nan,
                technical_reserves_stb=np.nan,
                historical_recovery_factor_percent=np.nan,
                ultimate_recovery_factor_percent=np.nan,
                constrained_by_ooip=False,
                error=result.error,
            )

        eur_raw = float(result.eur)

        if not np.isfinite(eur_raw) or eur_raw < 0.0:
            return ReserveCandidate(
                model_name=result.model_name,
                success=False,
                qi=float(result.qi),
                Di=float(result.Di),
                b=float(result.b),
                historical_np_stb=historical_np_stb,
                eur_raw_stb=eur_raw,
                remaining_raw_stb=np.nan,
                technical_eur_stb=np.nan,
                technical_reserves_stb=np.nan,
                historical_recovery_factor_percent=np.nan,
                ultimate_recovery_factor_percent=np.nan,
                constrained_by_ooip=False,
                error="DCA EUR is not finite and non-negative.",
            )

        remaining_raw = max(
            eur_raw - historical_np_stb,
            0.0,
        )

        material_balance_remaining = max(
            self.ooip - historical_np_stb,
            0.0,
        )

        technical_reserves = min(
            remaining_raw,
            material_balance_remaining,
        )

        technical_eur = (
            historical_np_stb
            + technical_reserves
        )

        historical_rf = (
            100.0
            * historical_np_stb
            / self.ooip
        )

        ultimate_rf = (
            100.0
            * technical_eur
            / self.ooip
        )

        constrained = bool(
            remaining_raw
            > material_balance_remaining
            + max(1.0e-10, 1.0e-10 * self.ooip)
        )

        return ReserveCandidate(
            model_name=result.model_name,
            success=True,
            qi=float(result.qi),
            Di=float(result.Di),
            b=float(result.b),
            historical_np_stb=historical_np_stb,
            eur_raw_stb=eur_raw,
            remaining_raw_stb=remaining_raw,
            technical_eur_stb=technical_eur,
            technical_reserves_stb=technical_reserves,
            historical_recovery_factor_percent=historical_rf,
            ultimate_recovery_factor_percent=ultimate_rf,
            constrained_by_ooip=constrained,
            error=None,
        )

    def estimate(self) -> ReserveResult:
        """
        Calculate reserve estimates for all successful DCA models.

        The mathematically best DCA model remains the selected model.
        The other successful DCA estimates remain visible for uncertainty
        and reporting rather than being discarded.
        """
        historical_np = self._historical_cumulative(
            self.time,
            self.rate,
        )

        if historical_np > self.ooip:
            raise ValueError(
                "Historical cumulative production exceeds selected OOIP."
            )

        candidates = []

        for result in self.forecast_scenario_result.dca_results.values():
            candidates.append(
                self._candidate_from_dca(
                    result=result,
                    historical_np_stb=historical_np,
                )
            )

        successful = [
            candidate
            for candidate in candidates
            if candidate.success
            and np.isfinite(candidate.technical_reserves_stb)
        ]

        if not successful:
            raise RuntimeError(
                "No successful DCA candidate is available for reserves "
                "estimation."
            )

        selected_name = str(
            self.forecast_scenario_result.best_dca_model
        )

        selected = [
            candidate
            for candidate in successful
            if candidate.model_name == selected_name
        ]

        if not selected:
            # Defensive fallback; should not happen if ForecastScenarioResult
            # is internally consistent.
            selected_candidate = min(
                successful,
                key=lambda candidate: candidate.remaining_raw_stb,
            )
            selected_name = selected_candidate.model_name
        else:
            selected_candidate = selected[0]

        rows = []
        for candidate in candidates:
            rows.append(
                {
                    "Model": candidate.model_name,
                    "Success": candidate.success,
                    "qi": candidate.qi,
                    "Di": candidate.Di,
                    "b": candidate.b,
                    "Historical_Np_STB": candidate.historical_np_stb,
                    "EUR_Raw_STB": candidate.eur_raw_stb,
                    "EUR_Raw_MMSTB": (
                        candidate.eur_raw_stb / 1.0e6
                        if np.isfinite(candidate.eur_raw_stb)
                        else np.nan
                    ),
                    "Remaining_Raw_STB": candidate.remaining_raw_stb,
                    "Remaining_Raw_MMSTB": (
                        candidate.remaining_raw_stb / 1.0e6
                        if np.isfinite(candidate.remaining_raw_stb)
                        else np.nan
                    ),
                    "Technical_EUR_STB": candidate.technical_eur_stb,
                    "Technical_EUR_MMSTB": (
                        candidate.technical_eur_stb / 1.0e6
                        if np.isfinite(candidate.technical_eur_stb)
                        else np.nan
                    ),
                    "Technical_Reserves_STB": (
                        candidate.technical_reserves_stb
                    ),
                    "Technical_Reserves_MMSTB": (
                        candidate.technical_reserves_stb / 1.0e6
                        if np.isfinite(candidate.technical_reserves_stb)
                        else np.nan
                    ),
                    "Historical_RF_percent": (
                        candidate.historical_recovery_factor_percent
                    ),
                    "Ultimate_RF_percent": (
                        candidate.ultimate_recovery_factor_percent
                    ),
                    "OOIP_Constrained": (
                        candidate.constrained_by_ooip
                    ),
                    "Selected": (
                        candidate.model_name == selected_name
                    ),
                    "Error": candidate.error,
                }
            )

        table = pd.DataFrame(rows)

        warnings = []

        if any(
            candidate.constrained_by_ooip
            for candidate in successful
        ):
            warnings.append(
                "At least one DCA EUR exceeded the selected OOIP and "
                "was constrained by the material-balance OOIP limit."
            )

        warnings.append(
            "This is a technical remaining-recoverable estimate, not an "
            "economic or reserves-classification determination."
        )

        return ReserveResult(
            best_candidate=selected_candidate,
            candidates=table,
            selected_dca_model=selected_name,
            historical_np_stb=historical_np,
            best_eur_stb=selected_candidate.technical_eur_stb,
            best_remaining_reserves_stb=(
                selected_candidate.technical_reserves_stb
            ),
            best_ooip_stb=self.ooip,
            historical_recovery_factor_percent=(
                selected_candidate.historical_recovery_factor_percent
            ),
            ultimate_recovery_factor_percent=(
                selected_candidate.ultimate_recovery_factor_percent
            ),
            warnings=tuple(warnings),
        )


def estimate_reserves(
    volumetric_result: VolumetricResult,
    forecast_scenario_result: ForecastScenarioResult,
    historical_time_days: Sequence[float],
    historical_rate: Sequence[float],
) -> ReserveResult:
    """Convenience wrapper around ReservesEstimator."""
    return ReservesEstimator(
        volumetric_result=volumetric_result,
        forecast_scenario_result=forecast_scenario_result,
        historical_time_days=historical_time_days,
        historical_rate=historical_rate,
    ).estimate()


__all__ = [
    "ReserveCandidate",
    "ReserveResult",
    "ReservesEstimator",
    "estimate_reserves",
]
