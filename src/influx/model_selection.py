"""
PyMBAL - Aquifer Model Selection Engine

Compares material-balance-based calibration of:
    1. Schilthuis
    2. Fetkovich
    3. Van Everdingen-Hurst

This module does not modify calibration.py or mb_calibration.py.
It uses the validated MaterialBalanceInfluxCalibrator API.

The selector can derive a physically informed Fetkovich aquifer-volume
search interval from V-E-H geometry when an explicit Vi interval is not
supplied. This prevents the application from unnecessarily searching an
extremely broad 1e5 -> 1e8 rb interval.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .mb_calibration import MaterialBalanceInfluxCalibrator


# ======================================================================
# RESULT CONTAINER
# ======================================================================

@dataclass
class ModelSelectionResult:

    best_model: str
    best_result: object
    comparison: pd.DataFrame
    all_results: dict

    @property
    def best_rmse_rb(self):
        return self.best_result.rmse_rb

    @property
    def best_r2(self):
        return self.best_result.r2

    @property
    def best_ooip_stb(self):
        return self.best_result.estimated_ooip_stb


# ======================================================================
# MODEL SELECTOR
# ======================================================================

class AquiferModelSelector:

    SUPPORTED_MODELS = (
        "schilthuis",
        "fetkovich",
        "van_everdingen_hurst",
    )

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
        self.E_rb_per_STB = np.asarray(
            E_rb_per_STB,
            dtype=float,
        )

        self._validate_inputs()

    # ==================================================================
    # INPUT VALIDATION
    # ==================================================================

    def _validate_inputs(self):

        lengths = [
            len(self.time_days),
            len(self.pressure_psia),
            len(self.F_rb),
            len(self.E_rb_per_STB),
        ]

        if len(set(lengths)) != 1:
            raise ValueError(
                "All model-selection input histories must have the same length."
            )

        if len(self.time_days) < 2:
            raise ValueError(
                "At least two reservoir-history points are required."
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
                    f"{name} contains non-finite values."
                )

        if self.time_days[0] != 0.0:
            raise ValueError(
                "The first time value must be zero."
            )

        if np.any(np.diff(self.time_days) <= 0.0):
            raise ValueError(
                "time_days must be strictly increasing."
            )

        if np.any(self.F_rb < 0.0):
            raise ValueError(
                "F_rb cannot contain negative values."
            )

        if np.any(self.E_rb_per_STB < 0.0):
            raise ValueError(
                "E_rb_per_STB cannot contain negative values."
            )

    # ==================================================================
    # CALIBRATOR
    # ==================================================================

    def _create_calibrator(self):

        return MaterialBalanceInfluxCalibrator(
            time_days=self.time_days,
            pressure_psia=self.pressure_psia,
            F_rb=self.F_rb,
            E_rb_per_STB=self.E_rb_per_STB,
        )

    # ==================================================================
    # PHYSICAL FETKOVICH VOLUME BOUNDS
    # ==================================================================

    @staticmethod
    def estimate_fetkovich_volume_bounds(
        aquifer_porosity,
        aquifer_thickness_ft,
        reservoir_radius_ft,
        encroachment_angle_deg=360.0,
        lower_fraction=0.25,
        upper_fraction=4.0,
        hard_lower_bound=1.0e5,
        hard_upper_bound=1.0e8,
    ):
        """
        Estimate a physically informed Fetkovich Vi interval.

        The geometric pore-volume estimate is:

            Vi ~= phi * A_aquifer * h

        with:

            A_aquifer = pi * re^2 * theta / 360

        The returned interval is deliberately broad around that estimate.
        Vi is still calibrated; this only constrains the numerical search.
        """

        values = {
            "aquifer_porosity": aquifer_porosity,
            "aquifer_thickness_ft": aquifer_thickness_ft,
            "reservoir_radius_ft": reservoir_radius_ft,
            "encroachment_angle_deg": encroachment_angle_deg,
        }

        for name, value in values.items():

            if not np.isfinite(float(value)):
                raise ValueError(
                    f"{name} must be finite."
                )

        phi = float(aquifer_porosity)
        h = float(aquifer_thickness_ft)
        radius = float(reservoir_radius_ft)
        angle = float(encroachment_angle_deg)

        if not (0.0 < phi <= 1.0):
            raise ValueError(
                "aquifer_porosity must be between 0 and 1."
            )

        if h <= 0.0:
            raise ValueError(
                "aquifer_thickness_ft must be positive."
            )

        if radius <= 0.0:
            raise ValueError(
                "reservoir_radius_ft must be positive."
            )

        if not (0.0 < angle <= 360.0):
            raise ValueError(
                "encroachment_angle_deg must be > 0 and <= 360."
            )

        geometric_pore_volume = (
            phi
            * np.pi
            * radius**2
            * h
            * angle
            / 360.0
        )

        lower = max(
            float(hard_lower_bound),
            float(lower_fraction) * geometric_pore_volume,
        )

        upper = min(
            float(hard_upper_bound),
            float(upper_fraction) * geometric_pore_volume,
        )

        # If hard limits collapse the interval, fall back to the largest
        # physically useful interval allowed by the hard limits.
        if upper <= lower:
            lower = float(hard_lower_bound)
            upper = float(hard_upper_bound)

        if upper <= lower:
            raise ValueError(
                "Unable to construct a valid Fetkovich aquifer-volume interval."
            )

        return lower, upper, geometric_pore_volume

    # ==================================================================
    # SCHILTHUIS
    # ==================================================================

    def _run_schilthuis(self, config):

        calibrator = self._create_calibrator()

        return calibrator.calibrate_schilthuis(
            aquifer_pressure_psia=config.get(
                "aquifer_pressure_psia",
                self.pressure_psia[0],
            ),
            c_bounds=config.get(
                "c_bounds",
                (1e-6, 100.0),
            ),
        )

    # ==================================================================
    # FETKOVICH
    # ==================================================================

    def _run_fetkovich(self, config):

        calibrator = self._create_calibrator()

        volume_bounds = config.get(
            "aquifer_volume_bounds"
        )

        if volume_bounds is None:

            geometry_keys = [
                "aquifer_porosity",
                "aquifer_thickness_ft",
                "reservoir_radius_ft",
            ]

            if all(
                key in config
                for key in geometry_keys
            ):

                volume_bounds = (
                    self.estimate_fetkovich_volume_bounds(
                        aquifer_porosity=config[
                            "aquifer_porosity"
                        ],
                        aquifer_thickness_ft=config[
                            "aquifer_thickness_ft"
                        ],
                        reservoir_radius_ft=config[
                            "reservoir_radius_ft"
                        ],
                        encroachment_angle_deg=config.get(
                            "encroachment_angle_deg",
                            360.0,
                        ),
                        lower_fraction=config.get(
                            "fetkovich_volume_lower_fraction",
                            0.25,
                        ),
                        upper_fraction=config.get(
                            "fetkovich_volume_upper_fraction",
                            4.0,
                        ),
                        hard_lower_bound=config.get(
                            "fetkovich_volume_hard_lower_rb",
                            1.0e5,
                        ),
                        hard_upper_bound=config.get(
                            "fetkovich_volume_hard_upper_rb",
                            1.0e8,
                        ),
                    )[:2]
                )

        if volume_bounds is None:

            volume_bounds = (
                100_000.0,
                100_000_000.0,
            )

        return calibrator.calibrate_fetkovich(
            initial_aquifer_pressure_psia=config.get(
                "initial_aquifer_pressure_psia",
                config.get(
                    "aquifer_pressure_psia",
                    self.pressure_psia[0],
                ),
            ),
            total_compressibility_psi_inv=config[
                "total_compressibility_psi_inv"
            ],
            productivity_index_bounds=config.get(
                "productivity_index_bounds",
                (1.0, 1000.0),
            ),
            aquifer_volume_bounds=volume_bounds,
        )

    # ==================================================================
    # VAN EVERDINGEN-HURST
    # ==================================================================

    def _run_van_everdingen_hurst(self, config):

        required = [
            "aquifer_porosity",
            "aquifer_thickness_ft",
            "total_compressibility_psi_inv",
            "water_viscosity_cp",
            "reservoir_radius_ft",
        ]

        missing = [
            key
            for key in required
            if key not in config
        ]

        if missing:
            raise ValueError(
                "Missing V-E-H model parameters: "
                + ", ".join(missing)
            )

        calibrator = self._create_calibrator()

        return calibrator.calibrate_van_everdingen_hurst(
            aquifer_porosity=config[
                "aquifer_porosity"
            ],
            aquifer_thickness_ft=config[
                "aquifer_thickness_ft"
            ],
            total_compressibility_psi_inv=config[
                "total_compressibility_psi_inv"
            ],
            water_viscosity_cp=config[
                "water_viscosity_cp"
            ],
            reservoir_radius_ft=config[
                "reservoir_radius_ft"
            ],
            encroachment_angle_deg=config.get(
                "encroachment_angle_deg",
                360.0,
            ),
            permeability_bounds=config.get(
                "permeability_bounds",
                (1.0, 1000.0),
            ),
        )

    # ==================================================================
    # RUN ONE MODEL
    # ==================================================================

    def run_model(self, model_name, config=None):

        if config is None:
            config = {}

        name = model_name.lower().strip()

        if name == "schilthuis":
            return self._run_schilthuis(config)

        if name == "fetkovich":
            return self._run_fetkovich(config)

        if name in (
            "van_everdingen_hurst",
            "van-everdingen-hurst",
            "veh",
        ):
            return self._run_van_everdingen_hurst(config)

        raise ValueError(
            f"Unsupported aquifer model: {model_name}"
        )

    # ==================================================================
    # COMPARE MODELS
    # ==================================================================

    def compare_models(self, model_configs, models=None):

        if models is None:
            models = self.SUPPORTED_MODELS

        models = list(models)

        if not models:
            raise ValueError(
                "At least one aquifer model must be selected."
            )

        all_results = {}
        rows = []

        for model_name in models:

            normalized_name = model_name.lower().strip()

            if normalized_name not in self.SUPPORTED_MODELS:
                raise ValueError(
                    f"Unsupported aquifer model: {model_name}"
                )

            config = model_configs.get(
                normalized_name,
                {},
            )

            try:

                result = self.run_model(
                    normalized_name,
                    config,
                )

                all_results[normalized_name] = result

                rows.append(
                    {
                        "Model": result.model_name,
                        "Success": result.success,
                        "RMSE_rb": result.rmse_rb,
                        "MAE_rb": result.mae_rb,
                        "R2": result.r2,
                        "OOIP_STB": result.estimated_ooip_stb,
                        "Error": "",
                    }
                )

            except Exception as exc:

                all_results[normalized_name] = exc

                rows.append(
                    {
                        "Model": normalized_name,
                        "Success": False,
                        "RMSE_rb": np.inf,
                        "MAE_rb": np.inf,
                        "R2": -np.inf,
                        "OOIP_STB": np.nan,
                        "Error": str(exc),
                    }
                )

        comparison = pd.DataFrame(rows)

        valid = comparison[
            comparison["Success"]
            & np.isfinite(comparison["RMSE_rb"])
        ].copy()

        if valid.empty:
            raise RuntimeError(
                "No aquifer model produced a physically valid calibration result."
            )

        best_index = valid["RMSE_rb"].idxmin()
        best_display_name = valid.loc[
            best_index,
            "Model",
        ]

        best_result = None

        for result in all_results.values():

            if isinstance(result, Exception):
                continue

            if result.model_name == best_display_name:
                best_result = result
                break

        if best_result is None:
            raise RuntimeError(
                "Best calibration result could not be recovered."
            )

        return ModelSelectionResult(
            best_model=best_display_name,
            best_result=best_result,
            comparison=(
                comparison
                .sort_values(by="RMSE_rb")
                .reset_index(drop=True)
            ),
            all_results=all_results,
        )

    # ==================================================================
    # CONVENIENCE ALIAS
    # ==================================================================

    def select_best_model(self, model_configs, models=None):

        return self.compare_models(
            model_configs=model_configs,
            models=models,
        )


if __name__ == "__main__":

    print("=" * 70)
    print("PyMBAL - AQUIFER MODEL SELECTION ENGINE")
    print("=" * 70)

    print()
    print("Supported models:")

    for model in AquiferModelSelector.SUPPORTED_MODELS:
        print(f"  - {model}")
