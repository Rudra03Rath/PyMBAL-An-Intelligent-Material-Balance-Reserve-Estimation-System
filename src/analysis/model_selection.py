
"""
PyMBAL - Reservoir Model Selection & User Constraint Layer

High-level interpretation layer above
src.influx.gas_cap_aquifer_calibration.

No reservoir physics is duplicated here. The coupled calibrator remains
the single source of truth for calibration.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.influx.gas_cap_aquifer_calibration import (
    GasCapAquiferCalibrator,
    GasCapAquiferSelectionResult,
)


@dataclass(frozen=True)
class ModelSelectionConstraints:
    """Optional independent reservoir information."""

    gas_cap_present: Optional[bool] = None
    aquifer_present: Optional[bool] = None
    aquifer_model: Optional[str] = None
    gas_cap_ratio_m: Optional[float] = None

    def normalized(self):
        if self.gas_cap_present not in (None, True, False):
            raise ValueError("gas_cap_present must be True, False, or None.")
        if self.aquifer_present not in (None, True, False):
            raise ValueError("aquifer_present must be True, False, or None.")

        model = self.aquifer_model
        aquifer = self.aquifer_present

        if model is not None:
            aliases = {
                "schilthuis": "schilthuis",
                "schilthuis steady-state": "schilthuis",
                "fetkovich": "fetkovich",
                "fetkovich finite aquifer": "fetkovich",
                "veh": "van_everdingen_hurst",
                "van_everdingen_hurst": "van_everdingen_hurst",
                "van-everdingen-hurst": "van_everdingen_hurst",
                "van everdingen-hurst": "van_everdingen_hurst",
            }
            key = str(model).strip().lower()
            if key not in aliases:
                raise ValueError(f"Unsupported aquifer_model: {model}")
            model = aliases[key]

            if aquifer is False:
                raise ValueError(
                    "aquifer_model cannot be specified when aquifer_present=False."
                )
            aquifer = True

        return ModelSelectionConstraints(
            gas_cap_present=self.gas_cap_present,
            aquifer_present=aquifer,
            aquifer_model=model,
        )

    @property
    def is_fully_automatic(self):
        return (
            self.gas_cap_present is None
            and self.aquifer_present is None
            and self.aquifer_model is None
        )


@dataclass(frozen=True)
class IdentifiabilityAssessment:
    status: str
    message: str
    competitive_count: int
    best_case: str
    competitive_cases: Tuple[str, ...] = ()

    @property
    def ambiguous(self):
        return self.status == "AMBIGUOUS"

    @property
    def unique(self):
        return self.status == "UNIQUE"


@dataclass
class ModelSelectionResult:
    """Stable result object for app.py, Streamlit and reporting."""

    selection: GasCapAquiferSelectionResult
    constraints: ModelSelectionConstraints
    identifiability: IdentifiabilityAssessment

    @property
    def best_case(self):
        return self.selection.best_case

    @property
    def best_result(self):
        return self.selection.best_result

    @property
    def best_rmse_rb(self):
        return self.selection.best_rmse_rb

    @property
    def best_r2(self):
        return self.selection.best_r2

    @property
    def best_ooip_stb(self):
        return self.selection.best_ooip_stb

    @property
    def best_gas_cap_ratio_m(self):
        return self.selection.best_gas_cap_ratio_m

    @property
    def best_aquifer_model(self):
        return self.selection.best_aquifer_model

    @property
    def comparison(self):
        return self.selection.comparison

    @property
    def all_results(self):
        return self.selection.all_results

    @property
    def possible_cases(self):
        return self.selection.possible_cases

    @property
    def competitive_results(self):
        return self.selection.competitive_results

    @property
    def ambiguous(self):
        return self.identifiability.ambiguous

    def summary(self) -> Dict[str, Any]:
        return {
            "best_case": self.best_case,
            "best_aquifer_model": self.best_aquifer_model,
            "best_ooip_stb": self.best_ooip_stb,
            "best_gas_cap_ratio_m": self.best_gas_cap_ratio_m,
            "best_rmse_rb": self.best_rmse_rb,
            "best_r2": self.best_r2,
            "identifiability_status": self.identifiability.status,
            "competitive_count": self.identifiability.competitive_count,
            "competitive_cases": list(self.identifiability.competitive_cases),
            "constraints": {
                "gas_cap_present": self.constraints.gas_cap_present,
                "aquifer_present": self.constraints.aquifer_present,
                "aquifer_model": self.constraints.aquifer_model,
            },
        }


class ReservoirModelSelector:
    """
    High-level model-selection service.

    It delegates calibration to GasCapAquiferCalibrator and only handles
    constraints, interpretation and standardized results.
    """

    DEFAULT_M_BOUNDS = (0.0, 10.0)
    DEFAULT_RELATIVE_TOLERANCE = 0.05
    DEFAULT_ABSOLUTE_TOLERANCE_RB = 1.0e-8

    def __init__(
        self,
        F_rb,
        Eo_rb_per_STB,
        Eg_rb_per_STB,
        Efw_rb_per_STB,
        time_days,
        pressure_psia,
        model_configs=None,
    ):
        """
        Build the model-selection service.

        ``model_configs`` contains the physical parameter bounds/configuration
        required by the underlying aquifer calibration models. It is supplied
        by the caller (app.py/tests) rather than duplicated in this analysis
        layer.
        """
        self.model_configs = dict(model_configs or {})

        self.calibrator = GasCapAquiferCalibrator(
            F_rb=F_rb,
            Eo_rb_per_STB=Eo_rb_per_STB,
            Eg_rb_per_STB=Eg_rb_per_STB,
            Efw_rb_per_STB=Efw_rb_per_STB,
            time_days=time_days,
            pressure_psia=pressure_psia,
        )

    def select(
        self,
        *,
        constraints=None,
        model_configs=None,
        m_bounds=DEFAULT_M_BOUNDS,
        include_all_aquifer_models=True,
        competitive_relative_tolerance=DEFAULT_RELATIVE_TOLERANCE,
        competitive_absolute_tolerance=DEFAULT_ABSOLUTE_TOLERANCE_RB,
        gas_cap_presence_threshold=1.0e-6,
        gas_cap_ratio_m=None,
    ):
        constraints = (
            constraints or ModelSelectionConstraints()
        ).normalized()

        effective_model_configs = (
            self.model_configs
            if model_configs is None
            else dict(model_configs)
        )

        selection = self.calibrator.select_best_model(
            model_configs=effective_model_configs,
            m_bounds=tuple(m_bounds),
            include_all_aquifer_models=include_all_aquifer_models,
            gas_cap_presence_threshold=gas_cap_presence_threshold,
            gas_cap_present=constraints.gas_cap_present,
            aquifer_present=constraints.aquifer_present,
            aquifer_model=constraints.aquifer_model,
            competitive_relative_tolerance=competitive_relative_tolerance,
            competitive_absolute_tolerance=competitive_absolute_tolerance,
            fixed_gas_cap_ratio_m=gas_cap_ratio_m,
        )


        assessment = self._assess_identifiability(selection)

        return ModelSelectionResult(
            selection=selection,
            constraints=constraints,
            identifiability=assessment,
        )
        

    @staticmethod
    def _assess_identifiability(selection):
        comparison = selection.comparison.copy()

        # Primary source of truth: calibrator competitive flag.
        # Fallback: selection.competitive_results for older result objects.
        competitive = comparison[
            comparison["Competitive"].astype(bool)
            & comparison["Success"].astype(bool)
        ].copy()

        if len(competitive) == 0 and hasattr(selection, "competitive_results"):
            try:
                competitive = selection.competitive_results.copy()
            except Exception:
                pass

        cases = tuple(
            (
                competitive["Configuration"].astype(str)
                + np.where(
                    competitive["Aquifer_Model"].astype(str).str.lower() == "none",
                    "",
                    " [" + competitive["Aquifer_Model"].astype(str) + "]",
                )
            ).tolist()
        )

        if len(competitive) <= 1:
            return IdentifiabilityAssessment(
                status="UNIQUE",
                message=(
                    "One candidate clearly dominates under the configured "
                    "competitive-fit tolerance. This is a mathematical "
                    "selection result, not proof that every reservoir "
                    "parameter is independently identifiable."
                ),
                competitive_count=len(competitive),
                best_case=selection.best_case,
                competitive_cases=cases,
            )

        return IdentifiabilityAssessment(
            status="AMBIGUOUS",
            message=(
                "Multiple physically admissible interpretations produce "
                "practically equivalent material-balance fits. Material "
                "balance alone does not uniquely identify the drive "
                "mechanism for this history. Independent reservoir "
                "information should be used to constrain the interpretation."
            ),
            competitive_count=len(competitive),
            best_case=selection.best_case,
            competitive_cases=cases,
        )

    @staticmethod
    def describe_constraints(constraints=None):
        c = (constraints or ModelSelectionConstraints()).normalized()

        gas = (
            "Unknown" if c.gas_cap_present is None
            else "Present" if c.gas_cap_present
            else "Absent"
        )
        aquifer = (
            "Unknown" if c.aquifer_present is None
            else "Present" if c.aquifer_present
            else "Absent"
        )
        model = c.aquifer_model or "Automatic"

        return (
            f"Gas cap: {gas} | "
            f"Aquifer: {aquifer} | "
            f"Aquifer model: {model}"
        )

    @staticmethod
    def available_constraint_values():
        return {
            "gas_cap_present": (None, True, False),
            "aquifer_present": (None, True, False),
            "aquifer_model": (
                None,
                "schilthuis",
                "fetkovich",
                "van_everdingen_hurst",
            ),
        }


def build_model_selector(
    F_rb,
    Eo_rb_per_STB,
    Eg_rb_per_STB,
    Efw_rb_per_STB,
    time_days,
    pressure_psia,
    model_configs=None,
):
    return ReservoirModelSelector(
        F_rb=F_rb,
        Eo_rb_per_STB=Eo_rb_per_STB,
        Eg_rb_per_STB=Eg_rb_per_STB,
        Efw_rb_per_STB=Efw_rb_per_STB,
        time_days=time_days,
        pressure_psia=pressure_psia,
        model_configs=model_configs,
    )


__all__ = [
    "ModelSelectionConstraints",
    "IdentifiabilityAssessment",
    "ModelSelectionResult",
    "ReservoirModelSelector",
    "build_model_selector",
]
