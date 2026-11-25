"""
PyMBAL - Drive Mechanism Analysis

Interpretation layer above the validated material-balance/model-selection
workflow.

This module does not perform gas-cap or aquifer calibration. It receives
the selected model interpretation and the corresponding material-balance
histories, then calculates Drive Distribution Index (DDI) histories and
summarizes the dominant/combined drive mechanism.

DDI components:
    Solution-gas/oil expansion : Eo
    Gas-cap expansion         : m * Eg
    Rock + connate-water      : Efw
    Aquifer influx             : We / N

The four components are normalized to 100% at each valid history point.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriveMechanismResult:
    """Stable result object for app.py, reporting and visualization."""

    ddi_history: pd.DataFrame
    final_ddi: Dict[str, float]
    dominant_mechanism: str
    overall_mechanism: str
    confidence: str
    gas_cap_status: str
    aquifer_status: str
    aquifer_model: str
    identifiability_status: str
    competitive_drive_interpretations: Tuple[str, ...]
    evidence: Tuple[str, ...]

    @property
    def dominant_percentage(self) -> float:
        return float(max(self.final_ddi.values()))

    def summary(self) -> Dict[str, Any]:
        return {
            "final_ddi": dict(self.final_ddi),
            "dominant_mechanism": self.dominant_mechanism,
            "overall_mechanism": self.overall_mechanism,
            "confidence": self.confidence,
            "gas_cap_status": self.gas_cap_status,
            "aquifer_status": self.aquifer_status,
            "aquifer_model": self.aquifer_model,
            "identifiability_status": self.identifiability_status,
            "competitive_drive_interpretations": list(
                self.competitive_drive_interpretations
            ),
            "evidence": list(self.evidence),
        }


class DriveMechanismAnalyzer:
    """
    Calculate drive-distribution percentages for a selected
    material-balance interpretation.

    Parameters
    ----------
    model_selection_result:
        A ModelSelectionResult produced by ReservoirModelSelector.
    Eo_rb_per_STB, Eg_rb_per_STB, Efw_rb_per_STB:
        Expansion histories. Efw_rb_per_STB must be the Efw history
        corresponding to the selected interpretation.
    water_influx_rb:
        Cumulative water influx history We(t). Required for an aquifer
        contribution to be quantified. If omitted, aquifer contribution
        cannot be calculated and a ValueError is raised when the selected
        model contains an aquifer.
    N_stb:
        OOIP used to convert We to an equivalent reservoir expansion term.
        Defaults to the selected model's OOIP.
    m:
        Gas-cap ratio. Defaults to the selected model's calibrated m.
    """

    SOLUTION_GAS = "Solution-Gas Drive"
    GAS_CAP = "Gas-Cap Expansion"
    AQUIFER = "Aquifer Influx"
    ROCK_WATER = "Rock & Connate-Water Expansion"

    def __init__(
        self,
        model_selection_result,
        Eo_rb_per_STB,
        Eg_rb_per_STB,
        Efw_rb_per_STB,
        water_influx_rb=None,
        N_stb=None,
        m=None,
        pressure_psia=None,
        time_days=None,
        dominance_margin=10.0,
        dominance_threshold=40.0,
        zero_tolerance=1.0e-12,
    ):
        self.result = model_selection_result

        self.Eo = np.asarray(Eo_rb_per_STB, dtype=float)
        self.Eg = np.asarray(Eg_rb_per_STB, dtype=float)
        self.Efw = np.asarray(Efw_rb_per_STB, dtype=float)

        self.We = (
            None
            if water_influx_rb is None
            else np.asarray(water_influx_rb, dtype=float)
        )

        self.N = (
            float(N_stb)
            if N_stb is not None
            else float(model_selection_result.best_ooip_stb)
        )

        self.m = (
            float(m)
            if m is not None
            else float(model_selection_result.best_gas_cap_ratio_m)
        )

        self.pressure = (
            None
            if pressure_psia is None
            else np.asarray(pressure_psia, dtype=float)
        )

        self.time = (
            None
            if time_days is None
            else np.asarray(time_days, dtype=float)
        )

        self.dominance_margin = float(dominance_margin)
        self.dominance_threshold = float(dominance_threshold)
        self.zero_tolerance = float(zero_tolerance)

        self._validate_inputs()

    def _validate_inputs(self):
        n = len(self.Eo)

        if n == 0:
            raise ValueError("Expansion histories cannot be empty.")

        if len(self.Eg) != n or len(self.Efw) != n:
            raise ValueError(
                "Eo, Eg and Efw histories must have identical lengths."
            )

        if self.We is not None and len(self.We) != n:
            raise ValueError(
                "water_influx_rb must have the same length as expansion histories."
            )

        if self.pressure is not None and len(self.pressure) != n:
            raise ValueError(
                "pressure_psia must have the same length as expansion histories."
            )

        if self.time is not None and len(self.time) != n:
            raise ValueError(
                "time_days must have the same length as expansion histories."
            )

        if self.N <= 0.0 or not np.isfinite(self.N):
            raise ValueError("N_stb must be finite and positive.")

        if self.m < 0.0 or not np.isfinite(self.m):
            raise ValueError("m must be finite and non-negative.")

        for name, values in {
            "Eo": self.Eo,
            "Eg": self.Eg,
            "Efw": self.Efw,
        }.items():
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains non-finite values.")
            if np.any(values < -self.zero_tolerance):
                raise ValueError(f"{name} cannot contain negative values.")

        if self.We is not None:
            if not np.all(np.isfinite(self.We)):
                raise ValueError("water_influx_rb contains non-finite values.")
            if np.any(self.We < -self.zero_tolerance):
                raise ValueError("water_influx_rb cannot contain negative values.")

    def _selected_case_text(self) -> str:
        case = str(self.result.best_case)
        aquifer = str(self.result.best_aquifer_model or "None")

        if aquifer.lower() == "none":
            return case

        return f"{case} [{aquifer}]"

    def _statuses(self):
        case = self.result.best_case

        gas_cap = (
            "Present"
            if "Gas Cap" in case and "No Gas Cap" not in case
            else "Absent"
        )

        aquifer = (
            "Present"
            if "Aquifer" in case and "No Aquifer" not in case
            else "Absent"
        )

        aquifer_model = (
            str(self.result.best_aquifer_model)
            if aquifer == "Present"
            else "None"
        )

        return gas_cap, aquifer, aquifer_model

    def _confidence(self) -> str:
        status = str(self.result.identifiability.status).upper()

        if status == "UNIQUE":
            return "High"

        if status == "AMBIGUOUS":
            return "Moderate"

        return "Low"

    def _classify_overall(self, final_ddi: Dict[str, float]):
        ordered = sorted(
            final_ddi.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        top_name, top_value = ordered[0]
        second_value = ordered[1][1]

        if top_value >= self.dominance_threshold and (
            top_value - second_value >= self.dominance_margin
        ):
            return top_name

        return "Combination Drive"

    def analyze(self) -> DriveMechanismResult:
        gas_cap_status, aquifer_status, aquifer_model = self._statuses()

        if aquifer_status == "Present" and self.We is None:
            raise ValueError(
                "water_influx_rb is required to quantify an aquifer contribution "
                "for an aquifer-bearing selected model."
            )

        solution = np.maximum(self.Eo, 0.0)
        gas_cap = np.maximum(self.m * self.Eg, 0.0)
        rock_water = np.maximum(self.Efw, 0.0)

        if self.We is None:
            aquifer = np.zeros_like(solution)
        else:
            aquifer = np.maximum(self.We / self.N, 0.0)

        total = solution + gas_cap + rock_water + aquifer

        valid = total > self.zero_tolerance
        if not np.any(valid):
            raise ValueError(
                "Drive-distribution denominator is zero for the complete history."
            )

        ddi = {
            self.SOLUTION_GAS: np.zeros_like(total),
            self.GAS_CAP: np.zeros_like(total),
            self.AQUIFER: np.zeros_like(total),
            self.ROCK_WATER: np.zeros_like(total),
        }

        for key, values in (
            (self.SOLUTION_GAS, solution),
            (self.GAS_CAP, gas_cap),
            (self.AQUIFER, aquifer),
            (self.ROCK_WATER, rock_water),
        ):
            ddi[key][valid] = 100.0 * values[valid] / total[valid]

        # A pre-production row can have a zero denominator. Keep it as NaN
        # rather than fabricating a 0/100% interpretation.
        for key in ddi:
            ddi[key][~valid] = np.nan

        history = pd.DataFrame(ddi)

        if self.time is not None:
            history.insert(0, "Time_days", self.time)

        if self.pressure is not None:
            insert_at = 0
            if "Time_days" in history.columns:
                insert_at = 1
            history.insert(insert_at, "Pressure_psia", self.pressure)

        final_index = np.flatnonzero(valid)[-1]

        final_ddi = {
            key: float(values[final_index])
            for key, values in ddi.items()
        }

        dominant = max(final_ddi, key=final_ddi.get)
        overall = self._classify_overall(final_ddi)

        competitive = tuple(
            getattr(
                self.result.identifiability,
                "competitive_cases",
                (),
            )
        )

        evidence = [
            f"Selected interpretation: {self._selected_case_text()}",
            f"Gas-cap status: {gas_cap_status}",
            f"Aquifer status: {aquifer_status}",
        ]

        if aquifer_status == "Present":
            evidence.append(f"Aquifer model: {aquifer_model}")

        if self.result.identifiability.status == "AMBIGUOUS":
            evidence.append(
                "Multiple competitive material-balance interpretations remain; "
                "DDI is conditional on the selected interpretation."
            )

        evidence.append(
            "DDI values are calculated from the material-balance energy terms "
            "and normalized to 100% at each history point."
        )

        return DriveMechanismResult(
            ddi_history=history,
            final_ddi=final_ddi,
            dominant_mechanism=dominant,
            overall_mechanism=overall,
            confidence=self._confidence(),
            gas_cap_status=gas_cap_status,
            aquifer_status=aquifer_status,
            aquifer_model=aquifer_model,
            identifiability_status=str(
                self.result.identifiability.status
            ),
            competitive_drive_interpretations=competitive,
            evidence=tuple(evidence),
        )


def analyze_drive_mechanism(
    model_selection_result,
    Eo_rb_per_STB,
    Eg_rb_per_STB,
    Efw_rb_per_STB,
    water_influx_rb=None,
    **kwargs,
):
    """Convenience wrapper around DriveMechanismAnalyzer."""
    return DriveMechanismAnalyzer(
        model_selection_result=model_selection_result,
        Eo_rb_per_STB=Eo_rb_per_STB,
        Eg_rb_per_STB=Eg_rb_per_STB,
        Efw_rb_per_STB=Efw_rb_per_STB,
        water_influx_rb=water_influx_rb,
        **kwargs,
    ).analyze()


__all__ = [
    "DriveMechanismResult",
    "DriveMechanismAnalyzer",
    "analyze_drive_mechanism",
]
