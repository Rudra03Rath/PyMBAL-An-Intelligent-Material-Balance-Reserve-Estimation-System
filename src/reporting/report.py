"""
PyMBAL - Reporting Layer

This module is the final read-only aggregation layer above the validated
analysis/forecasting modules.

Responsibilities
----------------
- Collect validated result objects into one stable report object.
- Preserve model-selection ambiguity and competitive candidates.
- Expose drive-mechanism, volumetric, DCA and reserves summaries.
- Collect warnings/evidence without performing new reservoir calculations.
- Provide structured tables suitable for Streamlit, export and reporting.

This module deliberately does NOT:
- calibrate reservoir models,
- perform material-balance calculations,
- fit DCA models,
- calculate OOIP/GIIP,
- calculate reserves,
- generate plots.

The reporting layer is therefore an orchestration/serialization layer only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from src.analysis.model_selection import ModelSelectionResult
from src.analysis.drive_mechanism import DriveMechanismResult
from src.analysis.volumetrics import VolumetricResult
from src.analysis.reserves import ReserveResult
from src.forecasting.scenarios import ForecastScenarioResult


# ======================================================================
# REPORT RESULT
# ======================================================================

@dataclass
class PyMBALReport:
    """
    Complete engineering interpretation assembled from validated results.

    All component result objects are retained so the UI/reporting layer
    can access detailed information without recalculating anything.
    """

    model_selection: ModelSelectionResult
    drive_mechanism: DriveMechanismResult
    volumetrics: VolumetricResult
    forecasting: ForecastScenarioResult
    reserves: ReserveResult

    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def selected_configuration(self) -> str:
        return str(self.model_selection.best_case)

    @property
    def selected_aquifer_model(self) -> Optional[str]:
        model = self.model_selection.best_aquifer_model
        if model is None:
            return None
        return str(model)

    @property
    def identifiability_status(self) -> str:
        return str(self.model_selection.identifiability.status)

    @property
    def is_ambiguous(self) -> bool:
        return self.model_selection.ambiguous

    @property
    def selected_dca_model(self) -> str:
        return str(self.forecasting.best_dca_model)

    @property
    def ooip_stb(self) -> float:
        return float(self.volumetrics.best_ooip_stb)

    @property
    def giip_scf(self) -> float:
        return float(self.volumetrics.best_giip_scf)

    @property
    def remaining_reserves_stb(self) -> float:
        return float(self.reserves.best_remaining_reserves_stb)

    @property
    def eur_stb(self) -> float:
        return float(self.reserves.best_eur_stb)

    def executive_summary(self) -> Dict[str, Any]:
        """
        Return the compact set of values intended for the top of a report
        or Streamlit dashboard.
        """
        return {
            "selected_configuration": self.selected_configuration,
            "selected_aquifer_model": self.selected_aquifer_model,
            "identifiability_status": self.identifiability_status,
            "competitive_candidate_count": (
                self.model_selection.identifiability.competitive_count
            ),
            "dominant_drive": self.drive_mechanism.dominant_mechanism,
            "dominant_drive_percentage": (
                self.drive_mechanism.dominant_percentage
            ),
            "overall_drive": self.drive_mechanism.overall_mechanism,
            "drive_confidence": self.drive_mechanism.confidence,
            "ooip_stb": self.ooip_stb,
            "ooip_mmstb": self.volumetrics.ooip_mmstb,
            "giip_scf": self.giip_scf,
            "giip_bscf": self.volumetrics.giip_bscf,
            "selected_dca_model": self.selected_dca_model,
            "eur_stb": self.eur_stb,
            "eur_mmstb": self.reserves.best_eur_mmstb,
            "remaining_reserves_stb": self.remaining_reserves_stb,
            "remaining_reserves_mmstb": (
                self.reserves.best_remaining_reserves_mmstb
            ),
            "historical_recovery_factor_percent": (
                self.reserves.historical_recovery_factor_percent
            ),
            "ultimate_recovery_factor_percent": (
                self.reserves.ultimate_recovery_factor_percent
            ),
        }

    def model_selection_summary(self) -> Dict[str, Any]:
        return self.model_selection.summary()

    def drive_mechanism_summary(self) -> Dict[str, Any]:
        return self.drive_mechanism.summary()

    def volumetric_summary(self) -> Dict[str, Any]:
        return self.volumetrics.summary()

    def forecasting_summary(self) -> Dict[str, Any]:
        best = self.forecasting.best_dca_result

        return {
            "best_dca_model": self.forecasting.best_dca_model,
            "best_dca": best.summary(),
            "dca_comparison": self.forecasting.comparison_table(),
            "pressure_scenarios": self.forecasting.pressure_scenarios.copy(),
            "warnings": list(self.forecasting.warnings),
        }

    def reserves_summary(self) -> Dict[str, Any]:
        return self.reserves.summary()

    def competitive_candidates(self) -> pd.DataFrame:
        """
        Return the complete successful competitive model table.

        The reporting layer does not recompute competitiveness. It simply
        exposes the table already produced by model selection.
        """
        table = self.model_selection.comparison.copy()

        if "Competitive" not in table.columns:
            return pd.DataFrame()

        if "Success" in table.columns:
            table = table[
                table["Competitive"].astype(bool)
                & table["Success"].astype(bool)
            ].copy()
        else:
            table = table[
                table["Competitive"].astype(bool)
            ].copy()

        return table.reset_index(drop=True)

    def possible_candidates(self) -> pd.DataFrame:
        """
        Return all successful physical candidates exposed by model selection.
        """
        table = self.model_selection.possible_cases

        if table is None:
            return pd.DataFrame()

        if isinstance(table, pd.DataFrame):
            return table.copy()

        return pd.DataFrame(table)

    def volumetric_candidates(self) -> pd.DataFrame:
        return self.volumetrics.candidates.copy()

    def dca_comparison(self) -> pd.DataFrame:
        return self.forecasting.comparison_table().copy()

    def reserve_candidates(self) -> pd.DataFrame:
        return self.reserves.candidates.copy()

    def ddi_history(self) -> pd.DataFrame:
        return self.drive_mechanism.ddi_history.copy()

    def final_ddi(self) -> Dict[str, float]:
        return dict(self.drive_mechanism.final_ddi)

    def all_warnings(self) -> tuple[str, ...]:
        """
        Consolidate warnings from all validated layers.

        Duplicate warning strings are removed while preserving order.
        """
        values: List[str] = []

        values.extend(str(x) for x in self.warnings)
        values.extend(str(x) for x in self.forecasting.warnings)
        values.extend(str(x) for x in self.reserves.warnings)

        seen = set()
        unique = []

        for value in values:
            if value not in seen:
                seen.add(value)
                unique.append(value)

        return tuple(unique)

    def all_evidence(self) -> tuple[str, ...]:
        return tuple(
            str(x)
            for x in self.drive_mechanism.evidence
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the report into a JSON-friendly structured dictionary.

        DataFrames are converted to record lists. Detailed result objects
        remain represented through their public summaries.
        """
        return {
            "executive_summary": self.executive_summary(),
            "model_selection": self.model_selection_summary(),
            "drive_mechanism": self.drive_mechanism_summary(),
            "volumetrics": self.volumetric_summary(),
            "forecasting": self.forecasting_summary(),
            "reserves": self.reserves_summary(),
            "possible_candidates": (
                self.possible_candidates().to_dict(orient="records")
            ),
            "competitive_candidates": (
                self.competitive_candidates().to_dict(orient="records")
            ),
            "volumetric_candidates": (
                self.volumetric_candidates().to_dict(orient="records")
            ),
            "dca_comparison": (
                self.dca_comparison().to_dict(orient="records")
            ),
            "reserve_candidates": (
                self.reserve_candidates().to_dict(orient="records")
            ),
            "ddi_history": (
                self.ddi_history().to_dict(orient="records")
            ),
            "final_ddi": self.final_ddi(),
            "warnings": list(self.all_warnings()),
            "evidence": list(self.all_evidence()),
            "notes": list(self.notes),
        }


# ======================================================================
# REPORT BUILDER
# ======================================================================

class PyMBALReportBuilder:
    """
    Assemble a PyMBALReport from already validated result objects.

    No calculations are performed here.
    """

    def __init__(
        self,
        model_selection: ModelSelectionResult,
        drive_mechanism: DriveMechanismResult,
        volumetrics: VolumetricResult,
        forecasting: ForecastScenarioResult,
        reserves: ReserveResult,
    ):
        self.model_selection = model_selection
        self.drive_mechanism = drive_mechanism
        self.volumetrics = volumetrics
        self.forecasting = forecasting
        self.reserves = reserves

        self._validate_components()

    def _validate_components(self):
        if not isinstance(
            self.model_selection,
            ModelSelectionResult,
        ):
            raise TypeError(
                "model_selection must be a ModelSelectionResult."
            )

        if not isinstance(
            self.drive_mechanism,
            DriveMechanismResult,
        ):
            raise TypeError(
                "drive_mechanism must be a DriveMechanismResult."
            )

        if not isinstance(
            self.volumetrics,
            VolumetricResult,
        ):
            raise TypeError(
                "volumetrics must be a VolumetricResult."
            )

        if not isinstance(
            self.forecasting,
            ForecastScenarioResult,
        ):
            raise TypeError(
                "forecasting must be a ForecastScenarioResult."
            )

        if not isinstance(
            self.reserves,
            ReserveResult,
        ):
            raise TypeError(
                "reserves must be a ReserveResult."
            )

    def build(
        self,
        *,
        warnings=(),
        notes=(),
    ) -> PyMBALReport:
        """
        Build the final report object.

        Optional warnings/notes are application-level messages and do not
        alter any analytical result.
        """
        return PyMBALReport(
            model_selection=self.model_selection,
            drive_mechanism=self.drive_mechanism,
            volumetrics=self.volumetrics,
            forecasting=self.forecasting,
            reserves=self.reserves,
            warnings=tuple(str(x) for x in warnings),
            notes=tuple(str(x) for x in notes),
        )


def build_report(
    model_selection: ModelSelectionResult,
    drive_mechanism: DriveMechanismResult,
    volumetrics: VolumetricResult,
    forecasting: ForecastScenarioResult,
    reserves: ReserveResult,
    *,
    warnings=(),
    notes=(),
) -> PyMBALReport:
    """
    Convenience wrapper for PyMBALReportBuilder.
    """
    return PyMBALReportBuilder(
        model_selection=model_selection,
        drive_mechanism=drive_mechanism,
        volumetrics=volumetrics,
        forecasting=forecasting,
        reserves=reserves,
    ).build(
        warnings=warnings,
        notes=notes,
    )


__all__ = [
    "PyMBALReport",
    "PyMBALReportBuilder",
    "build_report",
]
