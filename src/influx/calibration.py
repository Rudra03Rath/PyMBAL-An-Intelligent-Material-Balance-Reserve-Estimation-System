"""
PyMBAL - Water Influx Parameter Calibration

Calibration utilities for:
    1. Schilthuis steady-state aquifer
    2. Fetkovich finite aquifer
    3. Van Everdingen-Hurst aquifer

The calibration engine estimates aquifer parameters by minimizing the
difference between target cumulative water influx (We) and model-predicted
cumulative water influx.

IMPORTANT
---------
At the current development stage, target We is supplied explicitly only
to validate parameter-recovery capability.

In the final PyMBAL workflow, the required We history will be inferred
from the material-balance equation.
"""

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Iterable, Optional

from scipy.optimize import minimize, minimize_scalar

from .schilthuis import SchilthuisAquifer
from .fetkovich import FetkovichAquifer
from .van_everdingen_hurst import VanEverdingenHurstAquifer


# ======================================================================
# RESULT CONTAINER
# ======================================================================

@dataclass
class CalibrationResult:
    """Container for aquifer calibration results."""

    model_name: str
    parameters: dict
    predicted_we_rb: np.ndarray
    observed_we_rb: np.ndarray
    residuals_rb: np.ndarray
    rmse_rb: float
    mae_rb: float
    r_squared: float
    objective_value: float
    success: bool
    message: str

    @property
    def r2(self) -> float:
        """Public alias for R-squared."""
        return self.r_squared

    @property
    def max_absolute_error_rb(self) -> float:
        """Maximum absolute cumulative-influx error."""
        if len(self.residuals_rb) == 0:
            return 0.0
        return float(np.max(np.abs(self.residuals_rb)))

    def summary(self) -> dict:
        """Return a compact calibration summary."""
        return {
            "model": self.model_name,
            "parameters": self.parameters,
            "RMSE_rb": self.rmse_rb,
            "MAE_rb": self.mae_rb,
            "R2": self.r_squared,
            "max_absolute_error_rb": self.max_absolute_error_rb,
            "success": self.success,
            "message": self.message,
        }


# ======================================================================
# CALIBRATION ENGINE
# ======================================================================

class InfluxCalibrator:
    """
    Calibrate water-influx models against cumulative water-influx data.

    Parameters
    ----------
    time_days :
        Reservoir time history in days.

    pressure_psia :
        Reservoir pressure history in psia.

    target_we_rb :
        Target/inferred cumulative water influx in rb.

    observed_we_rb :
        Alias for target_we_rb retained for compatibility.
    """

    def __init__(
        self,
        time_days: Iterable[float],
        pressure_psia: Iterable[float],
        target_we_rb: Optional[Iterable[float]] = None,
        observed_we_rb: Optional[Iterable[float]] = None,
    ):
        if target_we_rb is None and observed_we_rb is None:
            raise ValueError(
                "Either target_we_rb or observed_we_rb must be provided."
            )

        if target_we_rb is not None and observed_we_rb is not None:
            target_array = np.asarray(target_we_rb, dtype=float)
            observed_array = np.asarray(observed_we_rb, dtype=float)

            if not np.array_equal(target_array, observed_array):
                raise ValueError(
                    "target_we_rb and observed_we_rb were both provided "
                    "but contain different values."
                )

        we_input = (
            target_we_rb
            if target_we_rb is not None
            else observed_we_rb
        )

        self.time_days = np.asarray(time_days, dtype=float)
        self.pressure_psia = np.asarray(pressure_psia, dtype=float)
        self.observed_we_rb = np.asarray(we_input, dtype=float)

        # Public compatibility name expected by the test/application API.
        self.target_we_rb = self.observed_we_rb.copy()

        self._validate_inputs()

    # ==================================================================
    # INPUT VALIDATION
    # ==================================================================

    def _validate_inputs(self):
        """Validate calibration histories."""

        if (
            len(self.time_days) != len(self.pressure_psia)
            or len(self.time_days) != len(self.observed_we_rb)
        ):
            raise ValueError(
                "time_days, pressure_psia, and observed_we_rb "
                "must have identical lengths."
            )

        if len(self.time_days) < 2:
            raise ValueError(
                "At least two pressure/time observations are required."
            )

        if not np.all(np.isfinite(self.time_days)):
            raise ValueError(
                "time_days contains NaN or infinite values."
            )

        if not np.all(np.isfinite(self.pressure_psia)):
            raise ValueError(
                "pressure_psia contains NaN or infinite values."
            )

        if not np.all(np.isfinite(self.observed_we_rb)):
            raise ValueError(
                "observed_we_rb contains NaN or infinite values."
            )

        if not np.isclose(self.time_days[0], 0.0):
            raise ValueError("Initial time must be zero.")

        if np.any(np.diff(self.time_days) <= 0.0):
            raise ValueError(
                "time_days must be strictly increasing."
            )

        if np.any(self.pressure_psia <= 0.0):
            raise ValueError(
                "Reservoir pressure must be positive."
            )

        if np.any(self.observed_we_rb < 0.0):
            raise ValueError(
                "Water influx cannot be negative."
            )

        if np.any(np.diff(self.observed_we_rb) < -1e-10):
            raise ValueError(
                "Cumulative water influx must be non-decreasing."
            )

    # ==================================================================
    # COMMON HELPERS
    # ==================================================================

    @staticmethod
    def _calculate_metrics(observed, predicted):
        """Calculate RMSE, MAE and R-squared."""

        observed = np.asarray(observed, dtype=float)
        predicted = np.asarray(predicted, dtype=float)

        residuals = predicted - observed

        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        mae = float(np.mean(np.abs(residuals)))

        ss_res = float(np.sum((observed - predicted) ** 2))
        ss_tot = float(
            np.sum((observed - np.mean(observed)) ** 2)
        )

        if ss_tot <= 1e-30:
            r_squared = 1.0 if ss_res <= 1e-30 else 0.0
        else:
            r_squared = 1.0 - ss_res / ss_tot

        return rmse, mae, r_squared

    @staticmethod
    def _extract_we(result):
        """Extract cumulative We from an aquifer-model result."""

        if "We_rb" not in result.columns:
            raise ValueError(
                "Aquifer model result does not contain 'We_rb'."
            )

        we = result["We_rb"].to_numpy(dtype=float)

        if not np.all(np.isfinite(we)):
            raise ValueError(
                "Aquifer model produced invalid cumulative We values."
            )

        return we

    def _make_result(
        self,
        model_name,
        parameters,
        predicted,
        objective_value,
        success,
        message,
    ):
        """Build a consistent CalibrationResult for every model."""

        predicted = np.asarray(predicted, dtype=float)
        residuals = predicted - self.observed_we_rb

        rmse, mae, r_squared = self._calculate_metrics(
            self.observed_we_rb,
            predicted,
        )

        return CalibrationResult(
            model_name=model_name,
            parameters=parameters,
            predicted_we_rb=predicted,
            observed_we_rb=self.observed_we_rb.copy(),
            residuals_rb=residuals,
            rmse_rb=rmse,
            mae_rb=mae,
            r_squared=r_squared,
            objective_value=float(objective_value),
            success=bool(success),
            message=str(message),
        )

    # ==================================================================
    # SCHILTHUIS
    # ==================================================================

    def calibrate_schilthuis(
        self,
        aquifer_pressure_psia: float,
        c_bounds=(1e-6, 100.0),
    ):
        """Calibrate the Schilthuis influx constant C."""

        if aquifer_pressure_psia <= 0.0:
            raise ValueError(
                "aquifer_pressure_psia must be positive."
            )

        if len(c_bounds) != 2:
            raise ValueError(
                "c_bounds must contain exactly two values."
            )

        c_lower, c_upper = map(float, c_bounds)

        if c_lower <= 0.0:
            raise ValueError(
                "Lower C bound must be positive."
            )

        if c_upper <= c_lower:
            raise ValueError(
                "Upper C bound must be greater than lower C bound."
            )

        def evaluate(c):
            model = SchilthuisAquifer(
                aquifer_pressure_psia,
                float(c),
            )

            result = model.calculate_influx(
                self.pressure_psia,
                self.time_days,
            )

            return self._extract_we(result)

        def objective(c):
            predicted = evaluate(c)
            residuals = predicted - self.observed_we_rb
            return float(np.sum(residuals ** 2))

        optimization = minimize_scalar(
            objective,
            bounds=(c_lower, c_upper),
            method="bounded",
            options={"xatol": 1e-12, "maxiter": 1000},
        )

        best_c = float(optimization.x)
        predicted = evaluate(best_c)

        return self._make_result(
            model_name="Schilthuis Steady-State",
            parameters={
                "aquifer_pressure_psia": float(
                    aquifer_pressure_psia
                ),
                "C_rb_per_day_psi": best_c,
            },
            predicted=predicted,
            objective_value=optimization.fun,
            success=optimization.success,
            message=optimization.message,
        )

    # ==================================================================
    # FETKOVICH
    # ==================================================================

    def calibrate_fetkovich(
        self,
        initial_aquifer_pressure_psia: float,
        total_compressibility_psi_inv: float,
        productivity_index_bounds=(1e-6, 1e4),
        aquifer_volume_bounds=(1e5, 1e8),
    ):
        """
        Calibrate Fetkovich productivity index J and initial aquifer
        water volume Vi.
        """

        if initial_aquifer_pressure_psia <= 0.0:
            raise ValueError(
                "initial_aquifer_pressure_psia must be positive."
            )

        if total_compressibility_psi_inv <= 0.0:
            raise ValueError(
                "total_compressibility_psi_inv must be positive."
            )

        j_lower, j_upper = map(
            float,
            productivity_index_bounds,
        )
        vi_lower, vi_upper = map(
            float,
            aquifer_volume_bounds,
        )

        if j_lower <= 0.0 or j_upper <= j_lower:
            raise ValueError(
                "productivity_index_bounds must be positive and "
                "upper bound must exceed lower bound."
            )

        if vi_lower <= 0.0 or vi_upper <= vi_lower:
            raise ValueError(
                "aquifer_volume_bounds must be positive and "
                "upper bound must exceed lower bound."
            )

        x0 = np.log([
            np.sqrt(j_lower * j_upper),
            np.sqrt(vi_lower * vi_upper),
        ])

        bounds = [
            (np.log(j_lower), np.log(j_upper)),
            (np.log(vi_lower), np.log(vi_upper)),
        ]

        observed_scale = max(
            float(np.max(np.abs(self.observed_we_rb))),
            1.0,
        )

        def evaluate(log_parameters):
            j = float(np.exp(log_parameters[0]))
            vi = float(np.exp(log_parameters[1]))

            model = FetkovichAquifer(
                initial_aquifer_pressure_psia,
                vi,
                total_compressibility_psi_inv,
                j,
            )

            result = model.calculate_influx(
                self.pressure_psia,
                self.time_days,
            )

            predicted = self._extract_we(result)
            residuals = (
                predicted - self.observed_we_rb
            ) / observed_scale

            return float(np.sum(residuals ** 2))

        optimization = minimize(
            evaluate,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "ftol": 1e-15,
                "gtol": 1e-10,
                "maxiter": 2000,
            },
        )

        if not optimization.success:
            fallback = minimize(
                evaluate,
                x0=optimization.x,
                method="Nelder-Mead",
                options={
                    "xatol": 1e-10,
                    "fatol": 1e-15,
                    "maxiter": 5000,
                },
            )

            if fallback.fun < optimization.fun:
                optimization = fallback

        best_j = float(np.exp(optimization.x[0]))
        best_vi = float(np.exp(optimization.x[1]))

        model = FetkovichAquifer(
            initial_aquifer_pressure_psia,
            best_vi,
            total_compressibility_psi_inv,
            best_j,
        )

        result = model.calculate_influx(
            self.pressure_psia,
            self.time_days,
        )

        predicted = self._extract_we(result)

        return self._make_result(
            model_name="Fetkovich Finite Aquifer",
            parameters={
                "initial_aquifer_pressure_psia":
                    float(initial_aquifer_pressure_psia),
                "initial_water_volume_rb": best_vi,
                "total_compressibility_psi_inv":
                    float(total_compressibility_psi_inv),
                "productivity_index_rb_per_day_psi": best_j,
            },
            predicted=predicted,
            objective_value=float(optimization.fun),
            success=optimization.success,
            message=optimization.message,
        )

    # ==================================================================
    # VAN EVERDINGEN-HURST
    # ==================================================================

    def calibrate_van_everdingen_hurst(
        self,
        aquifer_porosity: float,
        aquifer_thickness_ft: float,
        total_compressibility_psi_inv: float,
        water_viscosity_cp: float,
        reservoir_radius_ft: float,
        encroachment_angle_deg: float = 360.0,
        permeability_bounds=(1.0, 1000.0),
    ):
        """
        Calibrate Van Everdingen-Hurst aquifer permeability.

        A bounded scalar optimization is used in log10(k) space.
        A least-squares solve is also evaluated as an independent
        candidate. The final parameter is selected from the candidate
        producing the smallest *actual unscaled* cumulative-We error.

        This final selection is important: the calibration API reports
        RMSE in rb, so optimizer tolerances must not be confused with
        the scaled objective used internally.
        """

        if len(permeability_bounds) != 2:
            raise ValueError(
                "permeability_bounds must contain exactly two values."
            )

        k_min, k_max = map(float, permeability_bounds)

        if k_min <= 0.0 or k_max <= 0.0:
            raise ValueError(
                "V-E-H permeability bounds must be positive."
            )

        if k_min >= k_max:
            raise ValueError(
                "V-E-H permeability lower bound must be smaller "
                "than upper bound."
            )

        if aquifer_porosity <= 0.0 or aquifer_porosity > 1.0:
            raise ValueError(
                "Aquifer porosity must be > 0 and <= 1."
            )

        if aquifer_thickness_ft <= 0.0:
            raise ValueError(
                "Aquifer thickness must be positive."
            )

        if total_compressibility_psi_inv <= 0.0:
            raise ValueError(
                "Total compressibility must be positive."
            )

        if water_viscosity_cp <= 0.0:
            raise ValueError(
                "Water viscosity must be positive."
            )

        if reservoir_radius_ft <= 0.0:
            raise ValueError(
                "Reservoir radius must be positive."
            )

        if not 0.0 < encroachment_angle_deg <= 360.0:
            raise ValueError(
                "Encroachment angle must be > 0 and <= 360 degrees."
            )

        def evaluate(k_md):
            model = VanEverdingenHurstAquifer(
                aquifer_permeability_md=float(k_md),
                aquifer_porosity=float(aquifer_porosity),
                aquifer_thickness_ft=float(
                    aquifer_thickness_ft
                ),
                total_compressibility_psi_inv=float(
                    total_compressibility_psi_inv
                ),
                water_viscosity_cp=float(water_viscosity_cp),
                reservoir_radius_ft=float(
                    reservoir_radius_ft
                ),
                encroachment_angle_deg=float(
                    encroachment_angle_deg
                ),
            )

            result = model.calculate_influx(
                self.pressure_psia,
                self.time_days,
            )

            return self._extract_we(result)

        observed_scale = max(
            float(np.max(np.abs(self.observed_we_rb))),
            1.0,
        )

        # --------------------------------------------------------------
        # Primary optimization: bounded scalar minimization in log10(k)
        # --------------------------------------------------------------

        log_k_min = np.log10(k_min)
        log_k_max = np.log10(k_max)

        def objective_log_k(log_k):
            k_md = 10.0 ** float(log_k)
            predicted = evaluate(k_md)
            residuals = (
                predicted - self.observed_we_rb
            ) / observed_scale
            return float(np.sum(residuals ** 2))

        scalar_optimization = minimize_scalar(
            objective_log_k,
            bounds=(log_k_min, log_k_max),
            method="bounded",
            options={
                "xatol": 1e-14,
                "maxiter": 2000,
            },
        )

        scalar_log_k = float(scalar_optimization.x)
        scalar_k = float(10.0 ** scalar_log_k)

        # --------------------------------------------------------------
        # Independent local least-squares candidate
        # --------------------------------------------------------------

        from scipy.optimize import least_squares

        def residual_log_k(x):
            k_md = 10.0 ** float(np.asarray(x).reshape(-1)[0])
            predicted = evaluate(k_md)

            return (
                predicted - self.observed_we_rb
            ) / observed_scale

        lsq_optimization = least_squares(
            residual_log_k,
            x0=np.array([scalar_log_k]),
            bounds=(
                np.array([log_k_min]),
                np.array([log_k_max]),
            ),
            xtol=1e-15,
            ftol=1e-15,
            gtol=1e-15,
            max_nfev=500,
        )

        lsq_log_k = float(lsq_optimization.x[0])
        lsq_k = float(10.0 ** lsq_log_k)

        # --------------------------------------------------------------
        # Select candidate using the actual physical RMSE in rb
        # --------------------------------------------------------------

        candidates = [
            (
                scalar_k,
                bool(scalar_optimization.success),
                str(scalar_optimization.message),
            ),
            (
                lsq_k,
                bool(lsq_optimization.success),
                str(lsq_optimization.message),
            ),
            # Evaluate exact-ish bound candidates too.
            (
                k_min,
                True,
                "Lower permeability bound evaluated.",
            ),
            (
                k_max,
                True,
                "Upper permeability bound evaluated.",
            ),
        ]

        best_k = None
        best_rmse = np.inf
        best_predicted = None
        best_success = False
        best_message = ""

        for k_md, success, message in candidates:
            predicted = evaluate(k_md)
            residuals = predicted - self.observed_we_rb
            rmse = float(np.sqrt(np.mean(residuals ** 2)))

            if rmse < best_rmse:
                best_rmse = rmse
                best_k = float(k_md)
                best_predicted = predicted
                best_success = bool(success)
                best_message = message

        # The calibration is only considered successful when the optimizer
        # itself converged. Bound candidates are diagnostic fallbacks, not
        # proof of optimizer convergence.
        optimizer_success = (
            bool(scalar_optimization.success)
            or bool(lsq_optimization.success)
        )

        if optimizer_success:
            best_success = True

        # Re-evaluate once at the selected permeability so the returned
        # arrays and metrics are guaranteed to correspond to best_k.
        best_predicted = evaluate(best_k)

        residuals = (
            best_predicted - self.observed_we_rb
        )

        rmse, mae, r_squared = self._calculate_metrics(
            self.observed_we_rb,
            best_predicted,
        )

        scaled_objective = float(
            np.sum((residuals / observed_scale) ** 2)
        )

        return CalibrationResult(
            model_name="Van Everdingen-Hurst",
            parameters={
                "aquifer_permeability_md": best_k,
                "aquifer_porosity": float(aquifer_porosity),
                "aquifer_thickness_ft": float(
                    aquifer_thickness_ft
                ),
                "total_compressibility_psi_inv": float(
                    total_compressibility_psi_inv
                ),
                "water_viscosity_cp": float(
                    water_viscosity_cp
                ),
                "reservoir_radius_ft": float(
                    reservoir_radius_ft
                ),
                "encroachment_angle_deg": float(
                    encroachment_angle_deg
                ),
            },
            predicted_we_rb=best_predicted,
            observed_we_rb=self.observed_we_rb.copy(),
            residuals_rb=residuals,
            rmse_rb=rmse,
            mae_rb=mae,
            r_squared=r_squared,
            objective_value=scaled_objective,
            success=best_success,
            message=best_message,
        )

    # ==================================================================
    # MODEL COMPARISON
    # ==================================================================

    @staticmethod
    def compare_results(results: Iterable[CalibrationResult]):
        """Compare and rank calibration results."""

        rows = []

        for result in results:
            rows.append(
                {
                    "Model": result.model_name,
                    "RMSE_rb": result.rmse_rb,
                    "MAE_rb": result.mae_rb,
                    "R2": result.r_squared,
                    "Max_Error_rb":
                        result.max_absolute_error_rb,
                    "Success": result.success,
                }
            )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "Rank",
                    "Model",
                    "RMSE_rb",
                    "MAE_rb",
                    "R2",
                    "Max_Error_rb",
                    "Success",
                ]
            )

        comparison = pd.DataFrame(rows)

        comparison = comparison.sort_values(
            by=["RMSE_rb", "MAE_rb"],
            ascending=[True, True],
        ).reset_index(drop=True)

        comparison.insert(
            0,
            "Rank",
            np.arange(1, len(comparison) + 1),
        )

        return comparison

    # ==================================================================
    # BEST MODEL
    # ==================================================================

    @staticmethod
    def select_best_model(
        results: Iterable[CalibrationResult],
    ):
        """Select the successful model with minimum RMSE."""

        results = list(results)

        if not results:
            raise ValueError(
                "At least one calibration result is required."
            )

        successful = [
            result for result in results
            if result.success
        ]

        if not successful:
            raise ValueError(
                "No successful calibration results available."
            )

        return min(
            successful,
            key=lambda result: (
                result.rmse_rb,
                result.mae_rb,
            ),
        )


# ======================================================================
# CONVENIENCE FUNCTION
# ======================================================================

def calibrate_all_models(
    time_days,
    pressure_psia,
    observed_we_rb,
    *,
    schilthuis_aquifer_pressure_psia,
    schilthuis_c_bounds=(1e-6, 100.0),
    fetkovich_initial_aquifer_pressure_psia=4000.0,
    fetkovich_total_compressibility_psi_inv=1e-5,
    fetkovich_productivity_index_bounds=(1e-6, 1e4),
    fetkovich_aquifer_volume_bounds=(1e5, 1e8),
    veh_aquifer_porosity=0.2,
    veh_aquifer_thickness_ft=100.0,
    veh_total_compressibility_psi_inv=1e-5,
    veh_water_viscosity_cp=0.5,
    veh_reservoir_radius_ft=1000.0,
    veh_encroachment_angle_deg=360.0,
    veh_permeability_bounds=(1.0, 1000.0),
):
    """
    Calibrate Schilthuis, Fetkovich and V-E-H and return the best fit.
    """

    calibrator = InfluxCalibrator(
        time_days=time_days,
        pressure_psia=pressure_psia,
        observed_we_rb=observed_we_rb,
    )

    schilthuis_result = calibrator.calibrate_schilthuis(
        aquifer_pressure_psia=schilthuis_aquifer_pressure_psia,
        c_bounds=schilthuis_c_bounds,
    )

    fetkovich_result = calibrator.calibrate_fetkovich(
        initial_aquifer_pressure_psia=
            fetkovich_initial_aquifer_pressure_psia,
        total_compressibility_psi_inv=
            fetkovich_total_compressibility_psi_inv,
        productivity_index_bounds=
            fetkovich_productivity_index_bounds,
        aquifer_volume_bounds=
            fetkovich_aquifer_volume_bounds,
    )

    veh_result = calibrator.calibrate_van_everdingen_hurst(
        aquifer_porosity=veh_aquifer_porosity,
        aquifer_thickness_ft=veh_aquifer_thickness_ft,
        total_compressibility_psi_inv=
            veh_total_compressibility_psi_inv,
        water_viscosity_cp=veh_water_viscosity_cp,
        reservoir_radius_ft=veh_reservoir_radius_ft,
        encroachment_angle_deg=veh_encroachment_angle_deg,
        permeability_bounds=veh_permeability_bounds,
    )

    results = [
        schilthuis_result,
        fetkovich_result,
        veh_result,
    ]

    comparison = InfluxCalibrator.compare_results(results)
    best_result = InfluxCalibrator.select_best_model(results)

    return (
        schilthuis_result,
        fetkovich_result,
        veh_result,
        comparison,
        best_result,
    )


# ======================================================================
# BACKWARD-COMPATIBILITY / PUBLIC API ALIAS
# ======================================================================

WaterInfluxCalibrator = InfluxCalibrator
