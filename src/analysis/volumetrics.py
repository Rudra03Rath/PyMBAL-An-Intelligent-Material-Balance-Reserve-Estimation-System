"""
PyMBAL - OOIP / GIIP Volumetric Analysis Layer

Consumes the already validated ModelSelectionResult and converts each
admissible material-balance interpretation into OOIP and, where applicable,
GIIP.

No reservoir physics or model calibration is duplicated here.

Definitions used by the coupled calibration layer:

    F - We = N [ Eo + m Eg + (1 + m) Efw_base ]

and the gas-cap ratio is:

    m = Gci * Bgi / (N * Boi)

Therefore:

    Gci = m * N * Boi / Bgi

where:
    N   = OOIP [STB]
    Gci = initial gas-cap gas in place [scf]
    Boi = initial oil FVF [rb/STB]
    Bgi = initial gas FVF [rb/scf]

For no-gas-cap candidates, GIIP is represented numerically as 0 and the
candidate status explicitly records that no gas cap is present.

The layer deliberately preserves model-selection ambiguity. It does not
collapse several competitive OOIP/GIIP estimates into a single number.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.analysis.model_selection import ModelSelectionResult


@dataclass(frozen=True)
class VolumetricCandidate:
    configuration: str
    aquifer_model: Optional[str]
    gas_cap_present: bool
    aquifer_present: bool
    success: bool
    competitive: bool

    ooip_stb: float
    gas_cap_ratio_m: float
    giip_scf: float

    rmse_rb: float
    r2: float


@dataclass
class VolumetricResult:
    """Stable volumetric result for app.py, reporting and Streamlit."""

    best_candidate: VolumetricCandidate
    candidates: pd.DataFrame

    best_ooip_stb: float
    best_giip_scf: float

    competitive_ooip_min_stb: float
    competitive_ooip_max_stb: float
    competitive_giip_min_scf: Optional[float]
    competitive_giip_max_scf: Optional[float]

    identifiability_status: str
    gas_cap_status: str
    initial_boi_rb_per_STB: float
    initial_bgi_rb_per_scf: float

    @property
    def ooip_mmstb(self):
        return self.best_ooip_stb / 1.0e6

    @property
    def giip_bscf(self):
        return self.best_giip_scf / 1.0e9

    @property
    def competitive_ooip_range_mmstb(self):
        return (
            self.competitive_ooip_min_stb / 1.0e6,
            self.competitive_ooip_max_stb / 1.0e6,
        )

    @property
    def competitive_giip_range_bscf(self):
        if (
            self.competitive_giip_min_scf is None
            or self.competitive_giip_max_scf is None
        ):
            return None
        return (
            self.competitive_giip_min_scf / 1.0e9,
            self.competitive_giip_max_scf / 1.0e9,
        )

    @property
    def is_ambiguous(self):
        return self.identifiability_status == "AMBIGUOUS"

    def summary(self) -> Dict[str, Any]:
        return {
            "best_configuration": self.best_candidate.configuration,
            "best_aquifer_model": self.best_candidate.aquifer_model,
            "best_ooip_stb": self.best_ooip_stb,
            "best_giip_scf": self.best_giip_scf,
            "best_ooip_mmstb": self.ooip_mmstb,
            "best_giip_bscf": self.giip_bscf,
            "competitive_ooip_range_stb": (
                self.competitive_ooip_min_stb,
                self.competitive_ooip_max_stb,
            ),
            "competitive_giip_range_scf": (
                None
                if self.competitive_giip_min_scf is None
                else (
                    self.competitive_giip_min_scf,
                    self.competitive_giip_max_scf,
                )
            ),
            "identifiability_status": self.identifiability_status,
            "gas_cap_status": self.gas_cap_status,
        }


class VolumetricAnalyzer:
    """
    Calculate OOIP/GIIP from a ModelSelectionResult.

    Parameters
    ----------
    model_selection_result:
        Result returned by ReservoirModelSelector.select().

    Boi_rb_per_STB:
        Initial oil formation volume factor.

    Bgi_rb_per_scf:
        Initial gas formation volume factor.

    Important:
        Boi and Bgi are required explicitly rather than being guessed from
        the material-balance history. This keeps the volumetric layer
        independent from the PVT workflow.
    """

    def __init__(
        self,
        model_selection_result: ModelSelectionResult,
        Boi_rb_per_STB: float,
        Bgi_rb_per_scf: float,
    ):
        if not isinstance(model_selection_result, ModelSelectionResult):
            raise TypeError(
                "model_selection_result must be a ModelSelectionResult."
            )

        self.model_selection_result = model_selection_result
        self.Boi = float(Boi_rb_per_STB)
        self.Bgi = float(Bgi_rb_per_scf)

        if not np.isfinite(self.Boi) or self.Boi <= 0.0:
            raise ValueError("Boi_rb_per_STB must be finite and positive.")

        if not np.isfinite(self.Bgi) or self.Bgi <= 0.0:
            raise ValueError("Bgi_rb_per_scf must be finite and positive.")

    def calculate_giip(self, ooip_stb: float, m: float, gas_cap_present: bool):
        """
        Calculate initial gas-cap gas in place.

            GIIP = m * N * Boi / Bgi

        For a no-gas-cap candidate, return 0.0.
        """
        N = float(ooip_stb)
        ratio = float(m)

        if not np.isfinite(N) or N <= 0.0:
            raise ValueError("OOIP must be finite and positive.")

        if not np.isfinite(ratio) or ratio < 0.0:
            raise ValueError("Gas-cap ratio m must be finite and non-negative.")

        if not gas_cap_present:
            return 0.0

        return float(ratio * N * self.Boi / self.Bgi)

    @staticmethod
    def _display_aquifer(model):
        if model is None:
            return None
        text = str(model)
        return None if text.lower() == "none" else text

    def _build_candidate_table(self):
        comparison = self.model_selection_result.comparison.copy()

        required = {
            "Configuration",
            "Aquifer_Model",
            "Gas_Cap",
            "Aquifer",
            "Success",
            "RMSE_rb",
            "R2",
            "OOIP_STB",
            "m",
            "Competitive",
        }
        missing = required.difference(comparison.columns)
        if missing:
            raise ValueError(
                f"Model-selection comparison is missing columns: {sorted(missing)}"
            )

        rows = []

        for _, row in comparison.iterrows():
            success = bool(row["Success"])

            if success:
                ooip = float(row["OOIP_STB"])
                m = float(row["m"])
                gas_cap = bool(row["Gas_Cap"])
                giip = self.calculate_giip(ooip, m, gas_cap)
            else:
                ooip = np.nan
                m = np.nan
                giip = np.nan

            rows.append(
                {
                    "Configuration": str(row["Configuration"]),
                    "Aquifer_Model": self._display_aquifer(row["Aquifer_Model"]),
                    "Gas_Cap": bool(row["Gas_Cap"]),
                    "Aquifer": bool(row["Aquifer"]),
                    "Success": success,
                    "Competitive": bool(row["Competitive"]) if success else False,
                    "OOIP_STB": ooip,
                    "OOIP_MMSTB": (
                        ooip / 1.0e6 if np.isfinite(ooip) else np.nan
                    ),
                    "Gas_Cap_Ratio_m": m,
                    "GIIP_scf": giip,
                    "GIIP_Bscf": (
                        giip / 1.0e9 if np.isfinite(giip) else np.nan
                    ),
                    "RMSE_rb": float(row["RMSE_rb"]),
                    "R2": float(row["R2"]),
                    "Rank": int(row["Rank"]),
                }
            )

        return pd.DataFrame(rows)

    def analyze(self) -> VolumetricResult:
        table = self._build_candidate_table()

        successful = table[
            table["Success"].astype(bool)
            & np.isfinite(table["OOIP_STB"])
        ].copy()

        if successful.empty:
            raise RuntimeError(
                "No successful model-selection candidate is available for "
                "volumetric estimation."
            )

        best_case = self.model_selection_result.best_case
        best_model = self.model_selection_result.best_aquifer_model

        mask = successful["Configuration"].eq(best_case)

        if best_model is None:
            mask &= successful["Aquifer_Model"].isna()
        else:
            mask &= successful["Aquifer_Model"].eq(str(best_model))

        best_rows = successful.loc[mask]

        if best_rows.empty:
            # Fall back to the candidate with the best mathematical rank.
            best_row = successful.sort_values(
                by=["RMSE_rb", "Rank"]
            ).iloc[0]
        else:
            best_row = best_rows.iloc[0]

        competitive = successful[
            successful["Competitive"].astype(bool)
        ].copy()

        if competitive.empty:
            competitive = best_row.to_frame().T

        ooip_min = float(competitive["OOIP_STB"].min())
        ooip_max = float(competitive["OOIP_STB"].max())

        gas_cap_competitive = competitive[
            competitive["Gas_Cap"].astype(bool)
        ]

        if gas_cap_competitive.empty:
            giip_min = None
            giip_max = None
        else:
            giip_min = float(gas_cap_competitive["GIIP_scf"].min())
            giip_max = float(gas_cap_competitive["GIIP_scf"].max())

        gas_cap_status = (
            "Present"
            if bool(best_row["Gas_Cap"])
            else "Absent"
        )

        return VolumetricResult(
            best_candidate=VolumetricCandidate(
                configuration=str(best_row["Configuration"]),
                aquifer_model=(
                    None
                    if pd.isna(best_row["Aquifer_Model"])
                    else str(best_row["Aquifer_Model"])
                ),
                gas_cap_present=bool(best_row["Gas_Cap"]),
                aquifer_present=bool(best_row["Aquifer"]),
                success=True,
                competitive=bool(best_row["Competitive"]),
                ooip_stb=float(best_row["OOIP_STB"]),
                gas_cap_ratio_m=float(best_row["Gas_Cap_Ratio_m"]),
                giip_scf=float(best_row["GIIP_scf"]),
                rmse_rb=float(best_row["RMSE_rb"]),
                r2=float(best_row["R2"]),
            ),
            candidates=table,
            best_ooip_stb=float(best_row["OOIP_STB"]),
            best_giip_scf=float(best_row["GIIP_scf"]),
            competitive_ooip_min_stb=ooip_min,
            competitive_ooip_max_stb=ooip_max,
            competitive_giip_min_scf=giip_min,
            competitive_giip_max_scf=giip_max,
            identifiability_status=(
                self.model_selection_result.identifiability.status
            ),
            gas_cap_status=gas_cap_status,
            initial_boi_rb_per_STB=self.Boi,
            initial_bgi_rb_per_scf=self.Bgi,
        )


def estimate_volumetrics(
    model_selection_result: ModelSelectionResult,
    Boi_rb_per_STB: float,
    Bgi_rb_per_scf: float,
):
    """Convenience wrapper around VolumetricAnalyzer."""
    return VolumetricAnalyzer(
        model_selection_result=model_selection_result,
        Boi_rb_per_STB=Boi_rb_per_STB,
        Bgi_rb_per_scf=Bgi_rb_per_scf,
    ).analyze()


__all__ = [
    "VolumetricCandidate",
    "VolumetricResult",
    "VolumetricAnalyzer",
    "estimate_volumetrics",
]
