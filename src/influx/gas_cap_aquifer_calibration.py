"""
PyMBAL - Coupled Gas-Cap and Aquifer Model Selection

Evaluates the four reservoir-drive cases:

1. m = 0, We = 0
2. m > 0, We = 0
3. m = 0, We > 0
4. m > 0, We > 0

For cases containing an aquifer, the three supported aquifer models are
evaluated independently:

- Schilthuis Steady-State
- Fetkovich Finite Aquifer
- Van Everdingen-Hurst

The engine receives only material-balance history.  No We history is used
for calibration.

Core material balance:

    F - We = N[Eo + m Eg + (1+m)Efw_base]

For a fixed We history, GasCapCalibrator solves N and m analytically
through the linearized system:

    F - We = N(Eo + Efw_base) + (N m)(Eg + Efw_base)

The selector evaluates all admissible physical configurations. It does
not assume that the lowest-RMSE model is uniquely identifiable. If
several configurations have practically equivalent fits, they are
reported as competitive candidates rather than pretending that the
data prove one physical mechanism.

For coupled gas-cap + aquifer calibration, the aquifer parameters are
searched numerically.  At every aquifer trial, N and m are profiled
analytically.  This keeps the gas-cap parameters out of the nonlinear
search and makes the coupling numerically better conditioned.

Gas-cap-present candidates are required to recover a strictly positive m;
a solution that collapses to m ~= 0 is treated as the explicit no-gas case.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from src.influx.schilthuis import SchilthuisAquifer
from src.influx.fetkovich import FetkovichAquifer
from src.influx.van_everdingen_hurst import VanEverdingenHurstAquifer


# ======================================================================
# RESULT CONTAINERS
# ======================================================================

@dataclass
class DriveMechanismResult:
    case_name: str
    gas_cap_present: bool
    aquifer_present: bool
    aquifer_model: Optional[str]
    success: bool

    estimated_ooip_stb: float
    gas_cap_ratio_m: float

    aquifer_parameters: dict

    rmse_rb: float
    mae_rb: float
    r2: float

    residuals_rb: np.ndarray
    predicted_we_rb: np.ndarray
    f_minus_we_rb: np.ndarray

    message: str = ""

    def summary(self):
        return {
            "case": self.case_name,
            "gas_cap_present": self.gas_cap_present,
            "aquifer_present": self.aquifer_present,
            "aquifer_model": self.aquifer_model,
            "success": self.success,
            "estimated_ooip_stb": self.estimated_ooip_stb,
            "gas_cap_ratio_m": self.gas_cap_ratio_m,
            "aquifer_parameters": self.aquifer_parameters,
            "rmse_rb": self.rmse_rb,
            "mae_rb": self.mae_rb,
            "r2": self.r2,
            "message": self.message,
        }


@dataclass
class GasCapAquiferSelectionResult:
    best_case: str
    best_result: DriveMechanismResult
    comparison: pd.DataFrame
    all_results: Dict[str, DriveMechanismResult]
    ambiguous: bool = False
    near_best_candidates: Tuple[str, ...] = ()

    @property
    def best_rmse_rb(self):
        return self.best_result.rmse_rb

    @property
    def best_r2(self):
        return self.best_result.r2

    @property
    def best_ooip_stb(self):
        return self.best_result.estimated_ooip_stb

    @property
    def best_gas_cap_ratio_m(self):
        return self.best_result.gas_cap_ratio_m

    @property
    def best_aquifer_model(self):
        return self.best_result.aquifer_model

    @property
    def is_ambiguous(self):
        return self.ambiguous

    @property
    def competitive_candidates(self):
        return self.near_best_candidates

    @property
    def unique_best(self):
        return not self.ambiguous

    @property
    def competitive_results(self):
        if self.comparison.empty:
            return self.comparison.iloc[0:0].copy()
        return self.comparison[
            self.comparison["Competitive"].astype(bool)
        ].copy()

    @property
    def possible_cases(self):
        if self.comparison.empty:
            return self.comparison.iloc[0:0].copy()
        return self.comparison[
            self.comparison["Success"].astype(bool)
        ].copy()


# ======================================================================
# COUPLED SELECTOR
# ======================================================================

class GasCapAquiferCalibrator:

    CASE_NO_GAS_NO_AQUIFER = "No Gas Cap + No Aquifer"
    CASE_GAS_NO_AQUIFER = "Gas Cap + No Aquifer"
    CASE_NO_GAS_AQUIFER = "No Gas Cap + Aquifer"
    CASE_GAS_AQUIFER = "Gas Cap + Aquifer"

    AQUIFER_MODELS = (
        "schilthuis",
        "fetkovich",
        "van_everdingen_hurst",
    )

    def __init__(
        self,
        F_rb,
        Eo_rb_per_STB,
        Eg_rb_per_STB,
        Efw_rb_per_STB,
        time_days,
        pressure_psia,
    ):
        self.F_rb = np.asarray(F_rb, dtype=float)
        self.Eo_rb_per_STB = np.asarray(Eo_rb_per_STB, dtype=float)
        self.Eg_rb_per_STB = np.asarray(Eg_rb_per_STB, dtype=float)
        self.Efw_rb_per_STB = np.asarray(Efw_rb_per_STB, dtype=float)
        self.time_days = np.asarray(time_days, dtype=float)
        self.pressure_psia = np.asarray(pressure_psia, dtype=float)

        self._validate_inputs()

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def _validate_inputs(self):
        arrays = {
            "F_rb": self.F_rb,
            "Eo_rb_per_STB": self.Eo_rb_per_STB,
            "Eg_rb_per_STB": self.Eg_rb_per_STB,
            "Efw_rb_per_STB": self.Efw_rb_per_STB,
            "time_days": self.time_days,
            "pressure_psia": self.pressure_psia,
        }

        lengths = [len(v) for v in arrays.values()]
        if len(set(lengths)) != 1:
            raise ValueError(
                "All coupled gas-cap/aquifer input histories must have "
                "identical lengths."
            )

        if len(self.F_rb) < 3:
            raise ValueError(
                "At least 3 material-balance data points are required."
            )

        for name, values in arrays.items():
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{name} contains NaN or infinite values."
                )

        if not np.isclose(self.time_days[0], 0.0):
            raise ValueError("Initial time must be zero.")

        if np.any(np.diff(self.time_days) <= 0.0):
            raise ValueError("time_days must be strictly increasing.")

        if np.any(self.pressure_psia <= 0.0):
            raise ValueError("All reservoir pressures must be positive.")

        if np.any(self.F_rb < 0.0):
            raise ValueError("F_rb cannot contain negative values.")

        if np.any(np.diff(self.F_rb) < -1e-12):
            raise ValueError("F_rb must be non-decreasing.")

        for name in ("Eo_rb_per_STB", "Eg_rb_per_STB", "Efw_rb_per_STB"):
            if np.any(getattr(self, name) < 0.0):
                raise ValueError(f"{name} cannot contain negative values.")

        if abs(self.F_rb[0]) > 1e-10:
            raise ValueError("Initial underground withdrawal must be zero.")

        for name in ("Eo_rb_per_STB", "Eg_rb_per_STB", "Efw_rb_per_STB"):
            values = getattr(self, name)
            if abs(values[0]) > 1e-10:
                raise ValueError(f"Initial {name} must be zero.")

    # ==================================================================
    # BASIC GAS-CAP PROFILE
    # ==================================================================

    def _profile_gas_cap(self, we_rb, m_bounds=(0.0, 10.0), fixed_m=None):
        """
        Profile OOIP (N) and gas-cap ratio (m) for a supplied We history.

        IMPORTANT
        ---------
        The coupled engine uses the same formation/connate-water basis as
        the synthetic generator:

            F - We = N[
                Eo + m Eg + (1 + m) Efw_base
            ]

        Therefore, for a trial m,

            D(m) = Eo + m Eg + (1 + m) Efw_base

        and the least-squares OOIP is:

            N(m) = sum(D * (F-We)) / sum(D^2)

        The remaining one-dimensional m search is bounded.  This method is
        intentionally implemented here instead of delegating to the older
        standalone GasCapCalibrator so the coupled engine cannot silently
        use a different Efw convention.
        """
        we_rb = np.asarray(we_rb, dtype=float)
        m_min, m_max = map(float, m_bounds)

        if m_min < 0.0 or m_max <= m_min:
            raise ValueError("Invalid m_bounds.")

        y = self.F_rb - we_rb

        def evaluate(m):
            denominator = (
                self.Eo_rb_per_STB
                + m * self.Eg_rb_per_STB
                + (1.0 + m) * self.Efw_rb_per_STB
            )

            mask = np.isfinite(denominator) & np.isfinite(y) & (denominator > 0.0)

            if np.count_nonzero(mask) < 2:
                return np.inf, np.nan, np.full_like(y, np.nan)

            d = denominator[mask]
            yy = y[mask]

            dd = float(np.dot(d, d))
            if dd <= 0.0:
                return np.inf, np.nan, np.full_like(y, np.nan)

            N = float(np.dot(d, yy) / dd)

            if not np.isfinite(N) or N <= 0.0:
                return np.inf, np.nan, np.full_like(y, np.nan)

            predicted = N * denominator
            residuals = y - predicted
            sse = float(np.sum(residuals[mask] ** 2))

            return sse, N, residuals

        # If user supplied m, keep it fixed and only estimate OOIP.
        if fixed_m is not None:
            m = float(fixed_m)

            if m < 0:
                raise ValueError("Fixed gas-cap ratio m cannot be negative.")

            _, N, residuals = evaluate(m)

            if not np.isfinite(N):
                raise RuntimeError("Fixed gas-cap calibration failed.")

            mask = np.isfinite(residuals)
            residual_values = residuals[mask]

            rmse = float(np.sqrt(np.mean(residual_values ** 2)))
            mae = float(np.mean(np.abs(residual_values)))

            observed = y[mask]
            ss_res = float(np.sum(residual_values ** 2))
            ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
            r2 = 1.0 if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot

            denominator = (
                self.Eo_rb_per_STB
                + m * self.Eg_rb_per_STB
                + (1.0 + m) * self.Efw_rb_per_STB
            )

            apparent_N = np.full_like(y, np.nan, dtype=float)
            valid = denominator > 1e-14
            apparent_N[valid] = y[valid] / denominator[valid]

            return {
                "N": N,
                "m": m,
                "rmse": rmse,
                "mae": mae,
                "r2": float(r2),
                "residuals": residuals,
                "we": we_rb,
                "f_minus_we": y,
                "apparent_N": apparent_N,
            }

        # A bounded scalar search is sufficient because N is profiled
        # analytically for every trial m.
        objective = lambda m: evaluate(float(m))[0]

        grid = np.linspace(m_min, m_max, 501)
        values = np.array([objective(m) for m in grid], dtype=float)

        if not np.any(np.isfinite(values)):
            raise RuntimeError("No valid gas-cap profile was found.")

        idx = int(np.nanargmin(values))
        best_m = float(grid[idx])
        best_sse = float(values[idx])

        left = grid[max(0, idx - 1)]
        right = grid[min(len(grid) - 1, idx + 1)]

        if right > left:
            refined = minimize_scalar(
                objective,
                bounds=(left, right),
                method="bounded",
                options={"xatol": 1e-12, "maxiter": 2000},
            )

            if np.isfinite(refined.fun) and refined.fun <= best_sse:
                best_m = float(refined.x)

        best_sse, best_N, residuals = evaluate(best_m)

        if not np.isfinite(best_N):
            raise RuntimeError("Gas-cap calibration returned non-finite OOIP.")

        mask = np.isfinite(residuals)
        residual_values = residuals[mask]

        rmse = float(np.sqrt(np.mean(residual_values ** 2)))
        mae = float(np.mean(np.abs(residual_values)))

        observed = y[mask]
        ss_res = float(np.sum(residual_values ** 2))
        ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
        r2 = 1.0 if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot

        denominator = (
            self.Eo_rb_per_STB
            + best_m * self.Eg_rb_per_STB
            + (1.0 + best_m) * self.Efw_rb_per_STB
        )

        apparent_N = np.full_like(y, np.nan, dtype=float)
        valid = denominator > 1e-14
        apparent_N[valid] = y[valid] / denominator[valid]

        return {
            "N": best_N,
            "m": best_m,
            "rmse": rmse,
            "mae": mae,
            "r2": float(r2),
            "residuals": residuals,
            "we": we_rb,
            "f_minus_we": y,
            "apparent_N": apparent_N,
        }

    # ==================================================================
    # NO-GAS-CAP PROFILE
    # ==================================================================

    def _profile_no_gas_cap(self, we_rb):
        """
        Solve:

            F - We = N(Eo + Efw)

        with a zero-intercept profile.
        """
        we_rb = np.asarray(we_rb, dtype=float)
        y = self.F_rb - we_rb
        x = self.Eo_rb_per_STB + self.Efw_rb_per_STB

        mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0)

        if np.count_nonzero(mask) < 2:
            raise ValueError(
                "At least two non-zero expansion points are required."
            )

        denominator = float(np.sum(x[mask] ** 2))
        if denominator <= 0.0:
            raise ValueError("Insufficient expansion for OOIP estimation.")

        N = float(np.sum(x[mask] * y[mask]) / denominator)

        if not np.isfinite(N) or N <= 0.0:
            raise RuntimeError("Estimated OOIP is non-positive.")

        predicted = N * x
        residuals_full = y - predicted

        rmse = float(np.sqrt(np.mean(residuals_full[mask] ** 2)))
        mae = float(np.mean(np.abs(residuals_full[mask])))

        observed = y[mask]
        residuals = residuals_full[mask]
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
        r2 = 1.0 if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot

        return {
            "N": N,
            "m": 0.0,
            "rmse": rmse,
            "mae": mae,
            "r2": float(r2),
            "residuals": residuals_full,
            "we": we_rb,
            "f_minus_we": y,
        }

    # ==================================================================
    # RESULT BUILDER
    # ==================================================================

    def _build_result(
        self,
        case_name,
        gas_cap_present,
        aquifer_present,
        aquifer_model,
        profile,
        aquifer_parameters=None,
        message="",
    ):
        return DriveMechanismResult(
            case_name=case_name,
            gas_cap_present=gas_cap_present,
            aquifer_present=aquifer_present,
            aquifer_model=aquifer_model,
            success=True,
            estimated_ooip_stb=float(profile["N"]),
            gas_cap_ratio_m=float(profile["m"]),
            aquifer_parameters=dict(aquifer_parameters or {}),
            rmse_rb=float(profile["rmse"]),
            mae_rb=float(profile["mae"]),
            r2=float(profile["r2"]),
            residuals_rb=np.asarray(profile["residuals"], dtype=float),
            predicted_we_rb=np.asarray(profile["we"], dtype=float),
            f_minus_we_rb=np.asarray(profile["f_minus_we"], dtype=float),
            message=message,
        )

    # ==================================================================
    # CASE 1: NO GAS CAP + NO AQUIFER
    # ==================================================================

    def _run_no_gas_no_aquifer(self):
        profile = self._profile_no_gas_cap(
            np.zeros_like(self.F_rb)
        )

        return self._build_result(
            case_name=self.CASE_NO_GAS_NO_AQUIFER,
            gas_cap_present=False,
            aquifer_present=False,
            aquifer_model=None,
            profile=profile,
            message="Material balance calibrated without gas-cap or aquifer support.",
        )

    # ==================================================================
    # CASE 2: GAS CAP + NO AQUIFER
    # ==================================================================

    def _run_gas_no_aquifer(self, m_bounds, fixed_m=None):
        profile_result = self._profile_gas_cap(
            np.zeros_like(self.F_rb),
            m_bounds=m_bounds,
            fixed_m=fixed_m,
        )

        # _profile_gas_cap returns a plain dictionary containing the
        # profiled N, m, residuals and fit statistics.
        profile = self._profile_gas_cap(
            np.zeros_like(self.F_rb),
            m_bounds=m_bounds,
            fixed_m=fixed_m,
        )

        return self._build_result(
            case_name=self.CASE_GAS_NO_AQUIFER,
            gas_cap_present=True,
            aquifer_present=False,
            aquifer_model=None,
            profile=profile,
            message="Gas-cap ratio and OOIP calibrated with no aquifer.",
        )

    # ==================================================================
    # AQUIFER HELPERS
    # ==================================================================

    def _create_mb_calibrator(self):
        from src.influx.mb_calibration import MaterialBalanceInfluxCalibrator

        return MaterialBalanceInfluxCalibrator(
            time_days=self.time_days,
            pressure_psia=self.pressure_psia,
            F_rb=self.F_rb,
            E_rb_per_STB=self.Eo_rb_per_STB
            + self.Efw_rb_per_STB,
        )

    def _calculate_schilthuis_we(self, C, Pa):
        model = SchilthuisAquifer(
            aquifer_pressure_psia=float(Pa),
            influx_constant_rb_per_day_psi=float(C),
        )
        result = model.calculate_influx(
            pressure_psia=self.pressure_psia,
            time_days=self.time_days,
        )
        return self._extract_we(result)

    def _calculate_fetkovich_we(self, J, Vi, Pa, ct):
        model = FetkovichAquifer(
            initial_aquifer_pressure_psia=float(Pa),
            initial_water_volume_rb=float(Vi),
            total_compressibility_psi_inv=float(ct),
            productivity_index_rb_per_day_psi=float(J),
        )
        result = model.calculate_influx(
            pressure_psia=self.pressure_psia,
            time_days=self.time_days,
        )
        return self._extract_we(result)

    def _calculate_veh_we(
        self,
        k,
        phi,
        h,
        ct,
        mu,
        re,
        angle,
    ):
        model = VanEverdingenHurstAquifer(
            aquifer_permeability_md=float(k),
            aquifer_porosity=float(phi),
            aquifer_thickness_ft=float(h),
            total_compressibility_psi_inv=float(ct),
            water_viscosity_cp=float(mu),
            reservoir_radius_ft=float(re),
            encroachment_angle_deg=float(angle),
        )
        result = model.calculate_influx(
            pressure_psia=self.pressure_psia,
            time_days=self.time_days,
        )
        return self._extract_we(result)

    def _extract_we(self, result):
        if isinstance(result, pd.DataFrame):
            for column in (
                "We_rb",
                "we_rb",
                "cumulative_We_rb",
                "cumulative_we_rb",
            ):
                if column in result.columns:
                    we = result[column].to_numpy(dtype=float)
                    break
            else:
                raise ValueError(
                    "Aquifer result does not contain a recognized cumulative We column."
                )
        elif isinstance(result, dict):
            for key in (
                "We_rb",
                "we_rb",
                "cumulative_We_rb",
                "cumulative_we_rb",
            ):
                if key in result:
                    we = np.asarray(result[key], dtype=float)
                    break
            else:
                raise ValueError(
                    "Aquifer result does not contain a recognized cumulative We key."
                )
        else:
            raise TypeError(
                "Aquifer model result must be a DataFrame or dict."
            )

        # Validate against the reservoir history itself. Aquifer model
        # result structures are not required to expose a "Days" field.
        if len(we) != len(self.time_days):
            raise ValueError(
                "Aquifer We history length does not match the reservoir history."
            )

        if not np.all(np.isfinite(we)):
            raise ValueError("Aquifer model produced non-finite We values.")

        if len(we) == 0:
            raise ValueError("Aquifer model returned an empty We history.")

        return np.maximum(we, 0.0)

    # ==================================================================
    # AQUifer-ONLY OBJECTIVES
    # ==================================================================

    def _aquifer_only_profile(self, we):
        return self._profile_no_gas_cap(we)

    def _gas_aquifer_profile(self, we, m_bounds, fixed_m=None):
        # _profile_gas_cap returns a plain dictionary.  Keep the We
        # history explicit because it is the quantity subtracted from F.
        return self._profile_gas_cap(
            np.asarray(we, dtype=float),
            m_bounds=m_bounds,
            fixed_m=fixed_m,
        )

    # ==================================================================
    # SCHILTHUIS COUPLED SEARCH
    # ==================================================================

    def _run_schilthuis(
        self,
        config,
        gas_cap,
        m_bounds,
        fixed_m=None,
    ):
        Pa = float(config.get("aquifer_pressure_psia", self.pressure_psia[0]))
        c_bounds = tuple(config.get("c_bounds", (1e-6, 100.0)))

        if gas_cap:
            objective = lambda C: self._coupled_schilthuis_objective(
                C, Pa, m_bounds, fixed_m
            )
        else:
            objective = lambda C: self._aquifer_only_schilthuis_objective(
                C, Pa
            )

        grid = np.geomspace(c_bounds[0], c_bounds[1], 301)
        values = np.array([objective(C) for C in grid])
        finite = np.isfinite(values)

        if not np.any(finite):
            raise RuntimeError("No valid Schilthuis coupled solution was found.")

        idx = int(np.nanargmin(values))
        best = float(grid[idx])
        best_value = float(values[idx])

        left = grid[max(0, idx - 1)]
        right = grid[min(len(grid) - 1, idx + 1)]

        if right > left:
            refined = minimize_scalar(
                objective,
                bounds=(left, right),
                method="bounded",
                options={"xatol": 1e-12, "maxiter": 1000},
            )
            if np.isfinite(refined.fun) and refined.fun <= best_value:
                best = float(refined.x)

        we = self._calculate_schilthuis_we(best, Pa)
        profile = (
            self._gas_aquifer_profile(
                we,
                m_bounds,
                fixed_m=fixed_m,
            )
            if gas_cap
            else self._aquifer_only_profile(we)
        )

        return self._build_result(
            case_name=(
                self.CASE_GAS_AQUIFER
                if gas_cap
                else self.CASE_NO_GAS_AQUIFER
            ),
            gas_cap_present=gas_cap,
            aquifer_present=True,
            aquifer_model="Schilthuis Steady-State",
            profile=profile,
            aquifer_parameters={
                "aquifer_pressure_psia": Pa,
                "C_rb_per_day_psi": best,
            },
        )

    def _aquifer_only_schilthuis_objective(self, C, Pa):
        try:
            we = self._calculate_schilthuis_we(C, Pa)
            profile = self._aquifer_only_profile(we)
            return profile["rmse"] ** 2
        except Exception:
            return 1.0e30

    def _coupled_schilthuis_objective(self, C, Pa, m_bounds, fixed_m=None):
        try:
            we = self._calculate_schilthuis_we(C, Pa)
            profile = self._gas_aquifer_profile(
                we,
                m_bounds,
                fixed_m=fixed_m,
            )
            scale = max(float(np.max(np.abs(self.F_rb))), 1.0)
            return float(np.sum(profile["residuals"] ** 2) / scale ** 2)
        except Exception:
            return 1.0e30

    # ==================================================================
    # FETKOVICH COUPLED SEARCH
    # ==================================================================

    def _run_fetkovich(self, config, gas_cap, m_bounds, fixed_m=None):
        Pa = float(
            config.get(
                "initial_aquifer_pressure_psia",
                config.get("aquifer_pressure_psia", self.pressure_psia[0]),
            )
        )
        ct = float(config["total_compressibility_psi_inv"])

        j_bounds = tuple(config.get("productivity_index_bounds", (1.0, 1000.0)))
        vi_bounds = tuple(
            config.get(
                "aquifer_volume_bounds",
                (1.0e5, 1.0e8),
            )
        )

        bounds = [
            (np.log(j_bounds[0]), np.log(j_bounds[1])),
            (np.log(vi_bounds[0]), np.log(vi_bounds[1])),
        ]

        def objective(x):
            try:
                J = float(np.exp(x[0]))
                Vi = float(np.exp(x[1]))
                we = self._calculate_fetkovich_we(J, Vi, Pa, ct)

                profile = (
                    self._gas_aquifer_profile(we, m_bounds)
                    if gas_cap
                    else self._aquifer_only_profile(we)
                )

                scale = max(float(np.max(np.abs(self.F_rb))), 1.0)
                return float(np.sum(profile["residuals"] ** 2) / scale ** 2)
            except Exception:
               return 1.0e30 

        starts = [
            np.log([np.sqrt(j_bounds[0] * j_bounds[1]),
                    np.sqrt(vi_bounds[0] * vi_bounds[1])]),
            np.log([j_bounds[0], vi_bounds[0]]),
            np.log([j_bounds[0], vi_bounds[1]]),
            np.log([j_bounds[1], vi_bounds[0]]),
            np.log([j_bounds[1], vi_bounds[1]]),
        ]

        best = None
        for start in starts:
            result = minimize(
                objective,
                start,
                method="Powell",
                bounds=bounds,
                options={
                    "xtol": 1e-9,
                    "ftol": 1e-13,
                    "maxiter": 3000,
                },
            )
            if np.isfinite(result.fun) and (
                best is None or result.fun < best.fun
            ):
                best = result

        if best is None:
            raise RuntimeError("No valid Fetkovich coupled solution was found.")

        J = float(np.exp(best.x[0]))
        Vi = float(np.exp(best.x[1]))
        we = self._calculate_fetkovich_we(J, Vi, Pa, ct)

        profile = (
            self._gas_aquifer_profile(
                we,
                m_bounds,
                fixed_m=fixed_m,
            )
            if gas_cap
            else self._aquifer_only_profile(we)
        )

        return self._build_result(
            case_name=(
                self.CASE_GAS_AQUIFER
                if gas_cap
                else self.CASE_NO_GAS_AQUIFER
            ),
            gas_cap_present=gas_cap,
            aquifer_present=True,
            aquifer_model="Fetkovich Finite Aquifer",
            profile=profile,
            aquifer_parameters={
                "initial_aquifer_pressure_psia": Pa,
                "initial_water_volume_rb": Vi,
                "total_compressibility_psi_inv": ct,
                "productivity_index_rb_per_day_psi": J,
            },
            message=str(best.message),
        )

    # ==================================================================
    # V-E-H COUPLED SEARCH
    # ==================================================================

    def _run_veh(self, config, gas_cap, m_bounds, fixed_m=None):
        phi = float(config["aquifer_porosity"])
        h = float(config["aquifer_thickness_ft"])
        ct = float(config["total_compressibility_psi_inv"])
        mu = float(config["water_viscosity_cp"])
        re = float(config["reservoir_radius_ft"])
        angle = float(config.get("encroachment_angle_deg", 360.0))
        k_bounds = tuple(config.get("permeability_bounds", (1.0, 1000.0)))

        def objective(log_k):
            try:
                k = float(np.exp(log_k))
                we = self._calculate_veh_we(
                    k, phi, h, ct, mu, re, angle
                )
                profile = (
                    self._gas_aquifer_profile(we, m_bounds)
                    if gas_cap
                    else self._aquifer_only_profile(we)
                )
                scale = max(float(np.max(np.abs(self.F_rb))), 1.0)
                return float(np.sum(profile["residuals"] ** 2) / scale ** 2)
            except Exception:
                return 1.0e30

        grid = np.linspace(
            np.log(k_bounds[0]),
            np.log(k_bounds[1]),
            301,
        )
        values = np.array([objective(x) for x in grid])
        finite = np.isfinite(values)

        if not np.any(finite):
            raise RuntimeError("No valid V-E-H coupled solution was found.")

        idx = int(np.nanargmin(values))
        best_log_k = float(grid[idx])
        best_value = float(values[idx])

        left = grid[max(0, idx - 1)]
        right = grid[min(len(grid) - 1, idx + 1)]

        if right > left:
            refined = minimize_scalar(
                objective,
                bounds=(left, right),
                method="bounded",
                options={"xatol": 1e-12, "maxiter": 2000},
            )
            if np.isfinite(refined.fun) and refined.fun <= best_value:
                best_log_k = float(refined.x)

        k = float(np.exp(best_log_k))
        we = self._calculate_veh_we(
            k, phi, h, ct, mu, re, angle
        )

        profile = (
            self._gas_aquifer_profile(
                we,
                m_bounds,
                fixed_m=fixed_m,
            )
            if gas_cap
            else self._aquifer_only_profile(we)
        )

        return self._build_result(
            case_name=(
                self.CASE_GAS_AQUIFER
                if gas_cap
                else self.CASE_NO_GAS_AQUIFER
            ),
            gas_cap_present=gas_cap,
            aquifer_present=True,
            aquifer_model="Van Everdingen-Hurst",
            profile=profile,
            aquifer_parameters={
                "aquifer_permeability_md": k,
                "aquifer_porosity": phi,
                "aquifer_thickness_ft": h,
                "total_compressibility_psi_inv": ct,
                "water_viscosity_cp": mu,
                "reservoir_radius_ft": re,
                "encroachment_angle_deg": angle,
            },
        )

    # ==================================================================
    # AQUIFER CONFIG DISPATCH
    # ==================================================================

    def _run_aquifer_model(self, model, config, gas_cap, m_bounds, fixed_m=None):
        normalized = model.lower().strip()

        if normalized == "schilthuis":
            return self._run_schilthuis(
                config,
                gas_cap,
                m_bounds,
                fixed_m=fixed_m,
            )

        if normalized == "fetkovich":
            return self._run_fetkovich(
                config,
                gas_cap,
                m_bounds,
                fixed_m=fixed_m,
            )

        if normalized in (
            "van_everdingen_hurst",
            "van-everdingen-hurst",
            "veh",
        ):
            return self._run_veh(
                config,
                gas_cap,
                m_bounds,
                fixed_m=fixed_m,
            )

        raise ValueError(f"Unsupported aquifer model: {model}")

    # ==================================================================
    # FULL SELECTION
    # ==================================================================

    def select_best_model(
        self,
        model_configs=None,
        m_bounds=(0.0, 10.0),
        include_all_aquifer_models=True,
        gas_cap_presence_threshold=1.0e-6,
        gas_cap_present=None,
        aquifer_present=None,
        aquifer_model=None,
        competitive_relative_tolerance=0.05,
        competitive_absolute_tolerance=1.0e-8,
        fixed_gas_cap_ratio_m=None,
    ):
        """
        Evaluate all admissible coupled material-balance configurations.

        Important design principle
        ---------------------------
        The selector separates:

        1. mathematical best fit,
        2. competitive / practically indistinguishable fits, and
        3. user-supplied physical constraints.

        Material balance alone can be non-unique. Therefore the selector
        never treats a tiny RMSE difference as proof that one physical
        mechanism is uniquely correct.

        Optional physical constraints
        -------------------------------
        gas_cap_present:
            True  -> evaluate gas-cap candidates only
            False -> evaluate no-gas-cap candidates only
            None  -> evaluate both

        aquifer_present:
            True  -> evaluate aquifer candidates only
            False -> evaluate no-aquifer candidates only
            None  -> evaluate both

        aquifer_model:
            Optional one of:
                "schilthuis", "fetkovich", "van_everdingen_hurst"

        competitive_relative_tolerance:
            A candidate is considered competitive when:

                RMSE <= best_RMSE * (1 + tolerance)

        competitive_absolute_tolerance:
            Absolute RMSE allowance for very small residuals.

        Returns
        -------
        GasCapAquiferSelectionResult
            Contains every valid candidate, a ranked comparison table,
            the mathematical best candidate, and all competitive
            candidates.
        """
        if len(m_bounds) != 2:
            raise ValueError("m_bounds must contain exactly two values.")

        m_min, m_max = map(float, m_bounds)
        if m_min < 0.0 or m_max <= m_min:
            raise ValueError("Invalid m_bounds.")

        gas_cap_presence_threshold = float(gas_cap_presence_threshold)
        if gas_cap_presence_threshold < 0.0:
            raise ValueError(
                "gas_cap_presence_threshold cannot be negative."
            )

        if m_max <= gas_cap_presence_threshold:
            raise ValueError(
                "m_bounds upper limit must exceed gas_cap_presence_threshold."
            )

        if competitive_relative_tolerance < 0.0:
            raise ValueError(
                "competitive_relative_tolerance cannot be negative."
            )

        if competitive_absolute_tolerance < 0.0:
            raise ValueError(
                "competitive_absolute_tolerance cannot be negative."
            )

        if aquifer_model is not None:
            normalized_aquifer_model = aquifer_model.lower().strip()
            aliases = {
                "schilthuis": "schilthuis",
                "fetkovich": "fetkovich",
                "van_everdingen_hurst": "van_everdingen_hurst",
                "van-everdingen-hurst": "van_everdingen_hurst",
                "veh": "van_everdingen_hurst",
            }
            if normalized_aquifer_model not in aliases:
                raise ValueError(
                    f"Unsupported aquifer_model: {aquifer_model}"
                )
            aquifer_model = aliases[normalized_aquifer_model]

        if gas_cap_present not in (None, True, False):
            raise ValueError("gas_cap_present must be True, False, or None.")

        if aquifer_present not in (None, True, False):
            raise ValueError("aquifer_present must be True, False, or None.")

        model_configs = model_configs or {}

        all_results = {}
        rows = []

        def add_result(key, result):
            all_results[key] = result
            rows.append({
                "Configuration": result.case_name,
                "Aquifer_Model": result.aquifer_model or "None",
                "Gas_Cap": bool(result.gas_cap_present),
                "Aquifer": bool(result.aquifer_present),
                "Success": bool(result.success),
                "RMSE_rb": float(result.rmse_rb),
                "MAE_rb": float(result.mae_rb),
                "R2": float(result.r2),
                "OOIP_STB": float(result.estimated_ooip_stb),
                "m": float(result.gas_cap_ratio_m),
                "Error": "",
                "Candidate_Key": key,
            })

        def add_failure(key, case_name, model, gas_cap, aquifer, exc):
            rows.append({
                "Configuration": case_name,
                "Aquifer_Model": model or "None",
                "Gas_Cap": bool(gas_cap),
                "Aquifer": bool(aquifer),
                "Success": False,
                "RMSE_rb": np.inf,
                "MAE_rb": np.inf,
                "R2": -np.inf,
                "OOIP_STB": np.nan,
                "m": np.nan if gas_cap else 0.0,
                "Error": str(exc),
                "Candidate_Key": key,
            })

        # --------------------------------------------------------------
        # Physical-case filters
        # --------------------------------------------------------------
        allow_no_gas = gas_cap_present in (None, False)
        allow_gas = gas_cap_present in (None, True)
        allow_no_aquifer = aquifer_present in (None, False)
        allow_aquifer = aquifer_present in (None, True)

        # --------------------------------------------------------------
        # 1. No gas + no aquifer
        # --------------------------------------------------------------
        if allow_no_gas and allow_no_aquifer:
            try:
                add_result(
                    "no_gas_no_aquifer",
                    self._run_no_gas_no_aquifer(),
                )
            except Exception as exc:
                add_failure(
                    "no_gas_no_aquifer",
                    self.CASE_NO_GAS_NO_AQUIFER,
                    None,
                    False,
                    False,
                    exc,
                )

        # --------------------------------------------------------------
        # 2. Gas cap + no aquifer
        # --------------------------------------------------------------
        if allow_gas and allow_no_aquifer:
            try:
                add_result(
                    "gas_no_aquifer",
                    self._run_gas_no_aquifer(m_bounds, fixed_m=fixed_gas_cap_ratio_m),
                )
            except Exception as exc:
                add_failure(
                    "gas_no_aquifer",
                    self.CASE_GAS_NO_AQUIFER,
                    None,
                    True,
                    False,
                    exc,
                )

        # --------------------------------------------------------------
        # Aquifer model candidates
        # --------------------------------------------------------------
        if include_all_aquifer_models:
            models = list(self.AQUIFER_MODELS)
        else:
            models = list(model_configs.keys())

        if aquifer_model is not None:
            models = [aquifer_model]

        for model in models:
            config = model_configs.get(model, {})

            # No gas + aquifer
            if allow_no_gas and allow_aquifer:
                key = f"no_gas_{model}"
                try:
                    result = self._run_aquifer_model(
                        model,
                        config,
                        gas_cap=False,
                        m_bounds=m_bounds,
                    )
                    add_result(key, result)
                except Exception as exc:
                    add_failure(
                        key,
                        self.CASE_NO_GAS_AQUIFER,
                        self._display_aquifer_name(model),
                        False,
                        True,
                        exc,
                    )

            # Gas + aquifer
            if allow_gas and allow_aquifer:
                key = f"gas_{model}"
                try:
                    result = self._run_aquifer_model(
                        model,
                        config,
                        gas_cap=True,
                        m_bounds=m_bounds,
                        fixed_m=fixed_gas_cap_ratio_m,
                    )
                    add_result(key, result)
                except Exception as exc:
                    add_failure(
                        key,
                        self.CASE_GAS_AQUIFER,
                        self._display_aquifer_name(model),
                        True,
                        True,
                        exc,
                    )

        comparison = pd.DataFrame(rows)

        if comparison.empty:
            raise RuntimeError(
                "No candidate configurations were requested."
            )

        valid = comparison[
            comparison["Success"]
            & np.isfinite(comparison["RMSE_rb"])
        ].copy()

        if valid.empty:
            raise RuntimeError(
                "No physically valid gas-cap/aquifer configuration was found."
            )

        # A gas-cap-present candidate must actually contain a gas cap.
        gas_rows = valid["Gas_Cap"].astype(bool)
        invalid_gas_rows = (
            gas_rows
            & (
                ~np.isfinite(valid["m"])
                | (valid["m"] <= gas_cap_presence_threshold)
            )
        )

        valid = valid.loc[~invalid_gas_rows].copy()

        if valid.empty:
            raise RuntimeError(
                "No physically valid gas-cap configuration remained after "
                "enforcing the gas-cap presence criterion."
            )

        # --------------------------------------------------------------
        # Rank by mathematical fit.
        # --------------------------------------------------------------
        valid = valid.sort_values(
            by=["RMSE_rb", "MAE_rb"],
            ascending=[True, True],
        ).copy()

        best_index = valid.index[0]
        best_row = valid.loc[best_index]
        best_rmse = float(best_row["RMSE_rb"])

        comparison["Delta_RMSE_rb"] = (
            comparison["RMSE_rb"] - best_rmse
        )
        comparison["RMSE_Ratio_to_Best"] = (
            comparison["RMSE_rb"] / best_rmse
            if best_rmse > 0.0
            else np.where(
                comparison["RMSE_rb"] <= competitive_absolute_tolerance,
                1.0,
                np.inf,
            )
        )

        competitive_limit = max(
            competitive_absolute_tolerance,
            best_rmse * (1.0 + competitive_relative_tolerance),
        )

        comparison["Competitive"] = (
            comparison["Success"]
            & np.isfinite(comparison["RMSE_rb"])
            & (
                comparison["RMSE_rb"]
                <= competitive_limit
            )
        )

        # Only successful, physically admissible candidates can be
        # competitive. Invalid gas-cap candidates are explicitly removed.
        comparison.loc[
            comparison["Gas_Cap"]
            & (
                ~np.isfinite(comparison["m"])
                | (comparison["m"] <= gas_cap_presence_threshold)
            ),
            "Competitive",
        ] = False

        comparison = comparison.sort_values(
            by=["RMSE_rb", "MAE_rb"],
            ascending=[True, True],
            na_position="last",
        ).reset_index(drop=True)

        comparison["Rank"] = np.arange(1, len(comparison) + 1)

        near_best = comparison[
            comparison["Competitive"]
        ].copy()

        # --------------------------------------------------------------
        # Recover the best result.
        # --------------------------------------------------------------
        best_key = str(best_row["Candidate_Key"])
        best_result = all_results.get(best_key)

        if best_result is None:
            raise RuntimeError(
                "Best gas-cap/aquifer result could not be recovered."
            )

        # --------------------------------------------------------------
        # Engineering interpretation:
        #
        # If several candidates are within the practical tolerance,
        # report ambiguity rather than pretending that RMSE proves
        # a unique reservoir mechanism.
        # --------------------------------------------------------------
        ambiguous = len(near_best) > 1

        return GasCapAquiferSelectionResult(
            best_case=best_result.case_name,
            best_result=best_result,
            comparison=comparison.drop(
                columns=["Candidate_Key"]
            ),
            all_results=all_results,
            ambiguous=ambiguous,
            near_best_candidates=tuple(
                (
                    near_best["Configuration"].astype(str)
                    + np.where(
                        near_best["Aquifer_Model"].astype(str).str.lower() == "none",
                        "",
                        " [" + near_best["Aquifer_Model"].astype(str) + "]",
                    )
                ).tolist()
            ),
        )

    @staticmethod
    def _display_aquifer_name(model):
        names = {
            "schilthuis": "Schilthuis Steady-State",
            "fetkovich": "Fetkovich Finite Aquifer",
            "van_everdingen_hurst": "Van Everdingen-Hurst",
        }
        return names.get(model, model)


# ======================================================================
# ALIAS
# ======================================================================

GasCapAquiferModelSelector = GasCapAquiferCalibrator


if __name__ == "__main__":
    print("=" * 70)
    print("PyMBAL - COUPLED GAS-CAP + AQUIFER MODEL SELECTION")
    print("=" * 70)
    print()
    print("Physical cases:")
    print("  1. m = 0, We = 0")
    print("  2. m > 0, We = 0")
    print("  3. m = 0, We > 0")
    print("  4. m > 0, We > 0")
    print()
    print("Aquifer candidates:")
    print("  - Schilthuis Steady-State")
    print("  - Fetkovich Finite Aquifer")
    print("  - Van Everdingen-Hurst")
