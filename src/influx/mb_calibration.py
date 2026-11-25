"""
PyMBAL - Material-Balance-Based Water Influx Calibration

Coupled calibration of aquifer parameters and OOIP using:

    F - We(theta) = N * E

We is NOT supplied to this module. For every trial aquifer parameter set,
the corresponding aquifer model generates We(t), after which OOIP is
profiled from the zero-intercept Havlena-Ode relationship.

Supported coupled models
------------------------
1. Schilthuis Steady-State
2. Fetkovich Finite Aquifer
3. Van Everdingen-Hurst
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from src.influx.schilthuis import SchilthuisAquifer
from src.influx.fetkovich import FetkovichAquifer
from src.influx.van_everdingen_hurst import VanEverdingenHurstAquifer


# ============================================================================
# RESULT CONTAINER
# ============================================================================

@dataclass
class MBCalibrationResult:
    model_name: str
    success: bool
    estimated_ooip_stb: float
    parameters: dict
    rmse_rb: float
    mae_rb: float
    r2: float
    residuals_rb: np.ndarray
    predicted_we_rb: np.ndarray
    f_minus_we_rb: np.ndarray
    expansion_rb_per_stb: np.ndarray
    message: str = ""

    def summary(self):
        return {
            "model": self.model_name,
            "success": self.success,
            "estimated_ooip_stb": self.estimated_ooip_stb,
            "parameters": self.parameters,
            "rmse_rb": self.rmse_rb,
            "mae_rb": self.mae_rb,
            "r2": self.r2,
            "message": self.message,
        }


# ============================================================================
# MAIN CALIBRATOR
# ============================================================================

class MaterialBalanceInfluxCalibrator:

    def __init__(
        self,
        time_days,
        pressure_psia,
        F_rb,
        E_rb_per_STB,
    ):
        self.time_days = np.asarray(time_days, dtype=float)
        self.pressure_psia = np.asarray(pressure_psia, dtype=float)
        self.F_rb = np.asarray(F_rb, dtype=float)
        self.E_rb_per_STB = np.asarray(E_rb_per_STB, dtype=float)

        self._validate_inputs()

    # ========================================================================
    # INPUT VALIDATION
    # ========================================================================

    def _validate_inputs(self):
        n = len(self.time_days)

        if n == 0:
            raise ValueError("Calibration input cannot be empty.")

        if not (
            len(self.pressure_psia)
            == len(self.F_rb)
            == len(self.E_rb_per_STB)
            == n
        ):
            raise ValueError(
                "time_days, pressure_psia, F_rb and "
                "E_rb_per_STB must have equal lengths."
            )

        arrays = {
            "time_days": self.time_days,
            "pressure_psia": self.pressure_psia,
            "F_rb": self.F_rb,
            "E_rb_per_STB": self.E_rb_per_STB,
        }

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

        if np.any(self.E_rb_per_STB < 0.0):
            raise ValueError("Expansion values cannot be negative.")

        if abs(self.F_rb[0]) > 1e-12:
            raise ValueError(
                "Initial underground withdrawal must be zero."
            )

        if abs(self.E_rb_per_STB[0]) > 1e-12:
            raise ValueError("Initial expansion must be zero.")

    # ========================================================================
    # OOIP PROFILE
    # ========================================================================

    @staticmethod
    def _calculate_ooip_and_residuals(f_minus_we, expansion):
        f_minus_we = np.asarray(f_minus_we, dtype=float)
        expansion = np.asarray(expansion, dtype=float)

        mask = (
            np.isfinite(f_minus_we)
            & np.isfinite(expansion)
            & (expansion > 0.0)
        )

        if np.count_nonzero(mask) < 2:
            raise ValueError(
                "At least two non-zero expansion points are required."
            )

        x = expansion[mask]
        y = f_minus_we[mask]

        denominator = float(np.sum(x * x))
        if denominator <= 0.0:
            raise ValueError("Insufficient expansion for OOIP estimation.")

        N = float(np.sum(x * y) / denominator)
        residuals = y - N * x

        return N, residuals, mask

    @staticmethod
    def _calculate_statistics(residuals, observed, predicted):
        residuals = np.asarray(residuals, dtype=float)
        observed = np.asarray(observed, dtype=float)
        predicted = np.asarray(predicted, dtype=float)

        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        mae = float(np.mean(np.abs(residuals)))

        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(
            np.sum((observed - np.mean(observed)) ** 2)
        )

        if ss_tot <= 0.0:
            r2 = 1.0 if ss_res <= 1e-20 else 0.0
        else:
            r2 = 1.0 - ss_res / ss_tot

        return rmse, mae, r2

    # ========================================================================
    # WATER-INFLUX EXTRACTION
    # ========================================================================

    @staticmethod
    def _extract_cumulative_we(result):
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
                    "Aquifer model result does not contain a recognized "
                    "cumulative water-influx column."
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
                    "Aquifer model result does not contain a recognized "
                    "cumulative water-influx key."
                )
        else:
            raise TypeError(
                "Aquifer model result must be a pandas DataFrame or dict."
            )

        if not np.all(np.isfinite(we)):
            raise ValueError(
                "Aquifer model produced invalid cumulative We values."
            )

        if len(we) == 0:
            raise ValueError(
            "Aquifer model returned an empty water-influx history."
            )

        return np.maximum(we, 0.0)

    # ========================================================================
    # COMMON FINALIZATION
    # ========================================================================

    def _build_result(
        self,
        model_name,
        parameters,
        we,
        optimization_success,
        message,
    ):
        we = np.asarray(we, dtype=float)
        f_minus_we = self.F_rb - we

        N, _, mask = self._calculate_ooip_and_residuals(
            f_minus_we,
            self.E_rb_per_STB,
        )

        if not np.isfinite(N) or N <= 0.0:
            raise RuntimeError(
                "Final calibrated OOIP is non-positive."
            )

        predicted = N * self.E_rb_per_STB
        residuals_full = f_minus_we - predicted

        rmse, mae, r2 = self._calculate_statistics(
            residuals_full[mask],
            f_minus_we[mask],
            predicted[mask],
        )

        success = (
            bool(optimization_success)
            and np.isfinite(N)
            and N > 0.0
            and np.isfinite(rmse)
        )

        return MBCalibrationResult(
            model_name=model_name,
            success=success,
            estimated_ooip_stb=float(N),
            parameters=parameters,
            rmse_rb=float(rmse),
            mae_rb=float(mae),
            r2=float(r2),
            residuals_rb=residuals_full,
            predicted_we_rb=we,
            f_minus_we_rb=f_minus_we,
            expansion_rb_per_stb=self.E_rb_per_STB.copy(),
            message=str(message),
        )

    # ========================================================================
    # SCHILTHUIS
    # ========================================================================

    def _calculate_schilthuis_we(self, C, aquifer_pressure_psia):
        model = SchilthuisAquifer(
            aquifer_pressure_psia=aquifer_pressure_psia,
            influx_constant_rb_per_day_psi=float(C),
        )
        return self._extract_cumulative_we(
            model.calculate_influx(
                pressure_psia=self.pressure_psia,
                time_days=self.time_days,
            )
        )

    def _schilthuis_objective(self, C, aquifer_pressure_psia):
        try:
            C = float(C)
            if C <= 0.0 or not np.isfinite(C):
                return np.inf

            we = self._calculate_schilthuis_we(
                C,
                aquifer_pressure_psia,
            )
            N, residuals, _ = self._calculate_ooip_and_residuals(
                self.F_rb - we,
                self.E_rb_per_STB,
            )

            if not np.isfinite(N) or N <= 0.0:
                return np.inf

            sse = float(np.sum(residuals ** 2))
            return sse if np.isfinite(sse) else np.inf

        except (ValueError, TypeError, FloatingPointError, OverflowError):
            return np.inf

    def calibrate_schilthuis(
        self,
        aquifer_pressure_psia,
        c_bounds=(1e-6, 100.0),
    ):
        Pa = float(aquifer_pressure_psia)
        c_low, c_high = map(float, c_bounds)

        if Pa <= 0.0 or not np.isfinite(Pa):
            raise ValueError("Aquifer pressure must be positive.")
        if c_low <= 0.0 or c_high <= c_low:
            raise ValueError("Invalid Schilthuis C bounds.")

        grid = np.geomspace(c_low, c_high, 400)
        values = np.array([
            self._schilthuis_objective(C, Pa)
            for C in grid
        ])

        finite = np.isfinite(values)
        if not np.any(finite):
            raise RuntimeError(
                "No physically valid Schilthuis solution was found."
            )

        best_index = int(np.nanargmin(values))
        best_C = float(grid[best_index])
        best_sse = float(values[best_index])

        left = grid[max(0, best_index - 1)]
        right = grid[min(len(grid) - 1, best_index + 1)]

        if right > left:
            refined = minimize_scalar(
                lambda x: self._schilthuis_objective(x, Pa),
                bounds=(left, right),
                method="bounded",
                options={"xatol": 1e-12, "maxiter": 1000},
            )

            if np.isfinite(refined.fun) and refined.fun <= best_sse:
                best_C = float(refined.x)
                best_sse = float(refined.fun)
                opt_success = bool(refined.success)
                message = str(refined.message)
            else:
                opt_success = True
                message = "Best solution obtained from coarse parameter scan."
        else:
            opt_success = True
            message = "Best solution obtained from coarse parameter scan."

        we = self._calculate_schilthuis_we(best_C, Pa)

        return self._build_result(
            model_name="Schilthuis Steady-State",
            parameters={
                "aquifer_pressure_psia": Pa,
                "C_rb_per_day_psi": best_C,
            },
            we=we,
            optimization_success=opt_success,
            message=message,
        )

    # ========================================================================
    # FETKOVICH
    # ========================================================================

    def _calculate_fetkovich_we(
        self,
        J,
        Vi,
        Pa,
        ct,
    ):
        model = FetkovichAquifer(
            initial_aquifer_pressure_psia=Pa,
            initial_water_volume_rb=Vi,
            total_compressibility_psi_inv=ct,
            productivity_index_rb_per_day_psi=J,
        )

        return self._extract_cumulative_we(
            model.calculate_influx(
                pressure_psia=self.pressure_psia,
                time_days=self.time_days,
            )
        )

    def _fetkovich_objective(
        self,
        log_parameters,
        Pa,
        ct,
    ):
        try:
            J = float(np.exp(log_parameters[0]))
            Vi = float(np.exp(log_parameters[1]))

            we = self._calculate_fetkovich_we(J, Vi, Pa, ct)
            N, residuals, _ = self._calculate_ooip_and_residuals(
                self.F_rb - we,
                self.E_rb_per_STB,
            )

            if not np.isfinite(N) or N <= 0.0:
                return np.inf

            scale = max(float(np.max(np.abs(self.F_rb))), 1.0)
            value = float(np.sum(residuals ** 2) / scale ** 2)

            return value if np.isfinite(value) else np.inf

        except (
            ValueError,
            TypeError,
            FloatingPointError,
            OverflowError,
        ):
            return np.inf

    def _optimize_fetkovich(
        self,
        Pa,
        ct,
        productivity_index_bounds,
        aquifer_volume_bounds,
    ):
        j_low, j_high = map(float, productivity_index_bounds)
        vi_low, vi_high = map(float, aquifer_volume_bounds)

        if j_low <= 0.0 or j_high <= j_low:
            raise ValueError("Invalid productivity_index_bounds.")
        if vi_low <= 0.0 or vi_high <= vi_low:
            raise ValueError("Invalid aquifer_volume_bounds.")

        bounds = [
            (np.log(j_low), np.log(j_high)),
            (np.log(vi_low), np.log(vi_high)),
        ]

        # Start at the geometric center first, then use a small deterministic
        # set of log-space starts. This avoids the warning-heavy behavior of
        # blindly differentiating through invalid regions.
        starts = [
            np.log([np.sqrt(j_low * j_high),
                    np.sqrt(vi_low * vi_high)]),
            np.log([j_low, vi_low]),
            np.log([j_low, vi_high]),
            np.log([j_high, vi_low]),
            np.log([j_high, vi_high]),
            np.log([100.0, 5.0e6]),
        ]

        starts = [
            x for x in starts
            if bounds[0][0] <= x[0] <= bounds[0][1]
            and bounds[1][0] <= x[1] <= bounds[1][1]
        ]

        best = None

        # Powell is deliberately used here instead of numerical-gradient
        # optimization. The objective contains physically invalid regions
        # represented by inf, for which finite-difference gradients are noisy.
        for start in starts:
            result = minimize(
                self._fetkovich_objective,
                start,
                args=(Pa, ct),
                method="Powell",
                bounds=bounds,
                options={
                    "xtol": 1e-10,
                    "ftol": 1e-14,
                    "maxiter": 3000,
                },
            )

            if np.isfinite(result.fun) and (
                best is None or result.fun < best.fun
            ):
                best = result

        if best is None:
            raise RuntimeError(
                "No valid Fetkovich solution was found within the bounds."
            )

        return (
            float(np.exp(best.x[0])),
            float(np.exp(best.x[1])),
            float(best.fun),
            True,
            str(best.message),
        )

    def calibrate_fetkovich(
        self,
        initial_aquifer_pressure_psia,
        total_compressibility_psi_inv,
        productivity_index_bounds=(1.0, 1000.0),
        aquifer_volume_bounds=(1.0e5, 1.0e8),
    ):
        Pa = float(initial_aquifer_pressure_psia)
        ct = float(total_compressibility_psi_inv)

        if Pa <= 0.0 or not np.isfinite(Pa):
            raise ValueError(
                "Initial aquifer pressure must be positive."
            )
        if ct <= 0.0 or not np.isfinite(ct):
            raise ValueError(
                "Aquifer total compressibility must be positive."
            )

        J, Vi, _, success, message = self._optimize_fetkovich(
            Pa,
            ct,
            productivity_index_bounds,
            aquifer_volume_bounds,
        )

        we = self._calculate_fetkovich_we(J, Vi, Pa, ct)

        return self._build_result(
            model_name="Fetkovich Finite Aquifer",
            parameters={
                "initial_aquifer_pressure_psia": Pa,
                "initial_water_volume_rb": Vi,
                "total_compressibility_psi_inv": ct,
                "productivity_index_rb_per_day_psi": J,
            },
            we=we,
            optimization_success=success,
            message=message,
        )

    # ========================================================================
    # VAN EVERDINGEN-HURST
    # ========================================================================

    def _calculate_veh_we(
        self,
        permeability_md,
        aquifer_porosity,
        aquifer_thickness_ft,
        total_compressibility_psi_inv,
        water_viscosity_cp,
        reservoir_radius_ft,
        encroachment_angle_deg,
    ):
        model = VanEverdingenHurstAquifer(
            aquifer_permeability_md=float(permeability_md),
            aquifer_porosity=float(aquifer_porosity),
            aquifer_thickness_ft=float(aquifer_thickness_ft),
            total_compressibility_psi_inv=float(
                total_compressibility_psi_inv
            ),
            water_viscosity_cp=float(water_viscosity_cp),
            reservoir_radius_ft=float(reservoir_radius_ft),
            encroachment_angle_deg=float(encroachment_angle_deg),
        )

        return self._extract_cumulative_we(
            model.calculate_influx(
                pressure_psia=self.pressure_psia,
                time_days=self.time_days,
            )
        )

    def _veh_objective(
        self,
        log_k,
        aquifer_porosity,
        aquifer_thickness_ft,
        total_compressibility_psi_inv,
        water_viscosity_cp,
        reservoir_radius_ft,
        encroachment_angle_deg,
    ):
        try:
            k = float(np.exp(log_k))

            we = self._calculate_veh_we(
                k,
                aquifer_porosity,
                aquifer_thickness_ft,
                total_compressibility_psi_inv,
                water_viscosity_cp,
                reservoir_radius_ft,
                encroachment_angle_deg,
            )

            N, residuals, _ = self._calculate_ooip_and_residuals(
                self.F_rb - we,
                self.E_rb_per_STB,
            )

            if not np.isfinite(N) or N <= 0.0:
                return np.inf

            scale = max(float(np.max(np.abs(self.F_rb))), 1.0)
            value = float(np.sum(residuals ** 2) / scale ** 2)

            return value if np.isfinite(value) else np.inf

        except (
            ValueError,
            TypeError,
            FloatingPointError,
            OverflowError,
        ):
            return np.inf

    def calibrate_van_everdingen_hurst(
        self,
        aquifer_porosity,
        aquifer_thickness_ft,
        total_compressibility_psi_inv,
        water_viscosity_cp,
        reservoir_radius_ft,
        encroachment_angle_deg=360.0,
        permeability_bounds=(1.0, 1000.0),
    ):
        phi = float(aquifer_porosity)
        h = float(aquifer_thickness_ft)
        ct = float(total_compressibility_psi_inv)
        mu = float(water_viscosity_cp)
        re = float(reservoir_radius_ft)
        angle = float(encroachment_angle_deg)

        k_low, k_high = map(float, permeability_bounds)

        if not 0.0 < phi <= 1.0:
            raise ValueError(
                "Aquifer porosity must be > 0 and <= 1."
            )
        if h <= 0.0:
            raise ValueError("Aquifer thickness must be positive.")
        if ct <= 0.0:
            raise ValueError(
                "Total compressibility must be positive."
            )
        if mu <= 0.0:
            raise ValueError("Water viscosity must be positive.")
        if re <= 0.0:
            raise ValueError("Reservoir radius must be positive.")
        if not 0.0 < angle <= 360.0:
            raise ValueError(
                "Encroachment angle must be > 0 and <= 360 degrees."
            )
        if k_low <= 0.0 or k_high <= k_low:
            raise ValueError("Invalid permeability_bounds.")

        # V-E-H is one-dimensional in the unknown permeability. Use a
        # logarithmic scan followed by bounded refinement. This mirrors the
        # standalone calibration engine's successful strategy.
        log_grid = np.linspace(
            np.log(k_low),
            np.log(k_high),
            301,
        )

        objective_values = np.array([
            self._veh_objective(
                x,
                phi,
                h,
                ct,
                mu,
                re,
                angle,
            )
            for x in log_grid
        ])

        finite = np.isfinite(objective_values)
        if not np.any(finite):
            raise RuntimeError(
                "No physically valid V-E-H solution was found."
            )

        best_index = int(np.nanargmin(objective_values))
        best_log_k = float(log_grid[best_index])
        best_value = float(objective_values[best_index])

        left_index = max(0, best_index - 1)
        right_index = min(len(log_grid) - 1, best_index + 1)
        left = float(log_grid[left_index])
        right = float(log_grid[right_index])

        if right > left:
            refined = minimize_scalar(
                lambda x: self._veh_objective(
                    x,
                    phi,
                    h,
                    ct,
                    mu,
                    re,
                    angle,
                ),
                bounds=(left, right),
                method="bounded",
                options={
                    "xatol": 1e-12,
                    "maxiter": 2000,
                },
            )

            if np.isfinite(refined.fun) and refined.fun <= best_value:
                best_log_k = float(refined.x)
                best_value = float(refined.fun)
                success = bool(refined.success)
                message = str(refined.message)
            else:
                success = True
                message = (
                    "Best solution obtained from coarse log-space scan."
                )
        else:
            success = True
            message = "Best solution obtained from coarse log-space scan."

        k = float(np.exp(best_log_k))

        we = self._calculate_veh_we(
            k,
            phi,
            h,
            ct,
            mu,
            re,
            angle,
        )

        return self._build_result(
            model_name="Van Everdingen-Hurst",
            parameters={
                "aquifer_permeability_md": k,
                "aquifer_porosity": phi,
                "aquifer_thickness_ft": h,
                "total_compressibility_psi_inv": ct,
                "water_viscosity_cp": mu,
                "reservoir_radius_ft": re,
                "encroachment_angle_deg": angle,
            },
            we=we,
            optimization_success=success,
            message=message,
        )


# ============================================================================
# ALIAS
# ============================================================================

MBCalibrator = MaterialBalanceInfluxCalibrator


# ============================================================================
# MODULE SELF-CHECK
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PyMBAL - MATERIAL-BALANCE INFLUX CALIBRATION")
    print("=" * 70)
    print()
    print("✓ Module loaded successfully.")
    print()
    print("Supported models:")
    print("  ✓ Schilthuis Steady-State")
    print("  ✓ Fetkovich Finite Aquifer")
    print("  ✓ Van Everdingen-Hurst")
    print()
    print("Calibration equation:")
    print("  F - We(theta) = N × E")
