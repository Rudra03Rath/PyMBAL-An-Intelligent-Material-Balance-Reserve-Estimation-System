"""
PyMBAL - Van Everdingen-Hurst Water Influx Model

Classical unsteady-state radial aquifer model.

Current implementation:
    - Infinite-acting radial aquifer
    - Constant aquifer properties
    - Constant-terminal-pressure influence solution
    - Pressure-history superposition
    - Tabulated dimensionless cumulative influx Q_D
    - Logarithmic interpolation in t_D

Core equations
--------------

Dimensionless time:

    t_D = 0.0002637 * k * t
          ---------------------
          phi * mu_w * c_t * r_e^2

where:

    k       = aquifer permeability, md
    t       = time, days
    phi     = aquifer porosity
    mu_w    = water viscosity, cp
    c_t     = total compressibility, psi^-1
    r_e     = reservoir/aquifer inner radius, ft

Aquifer influx constant:

    B = 1.119 * phi * c_t * h * r_e^2 * f

where:

    f = theta / 360

For a pressure history, the Van Everdingen-Hurst
superposition solution is:

    We(t_n) =
        B * sum[
            DeltaP_j *
            Q_D(t_D,n - t_D,j)
        ]

where:

    DeltaP_j = P_(j-1) - P_j

and Q_D is the dimensionless cumulative water
influx function.

References:
    Van Everdingen & Hurst (1949)
    Standard petroleum reservoir engineering
    formulations of unsteady-state aquifer influx.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class VanEverdingenHurstAquifer:
    """
    Infinite-acting radial Van Everdingen-Hurst aquifer.

    Parameters
    ----------
    aquifer_permeability_md:
        Aquifer permeability in md.

    aquifer_porosity:
        Aquifer porosity as fraction.

    aquifer_thickness_ft:
        Aquifer thickness in ft.

    total_compressibility_psi_inv:
        Effective total aquifer compressibility in psi^-1.

    water_viscosity_cp:
        Aquifer water viscosity in cp.

    reservoir_radius_ft:
        Reservoir/aquifer inner radius in ft.

    encroachment_angle_deg:
        Aquifer contact angle in degrees.
        360 degrees represents a full radial aquifer.

    """

    aquifer_permeability_md: float
    aquifer_porosity: float
    aquifer_thickness_ft: float
    total_compressibility_psi_inv: float
    water_viscosity_cp: float
    reservoir_radius_ft: float
    encroachment_angle_deg: float = 360.0

    def __post_init__(self) -> None:

        self._validate_parameters()

        # Fraction of full radial circumference
        self.encroachment_fraction = (
            self.encroachment_angle_deg / 360.0
        )

        # Van Everdingen-Hurst aquifer constant
        #
        # Units: rb/psi
        self.aquifer_constant_rb_per_psi = (
            1.119
            * self.aquifer_porosity
            * self.total_compressibility_psi_inv
            * self.aquifer_thickness_ft
            * self.reservoir_radius_ft**2
            * self.encroachment_fraction
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_parameters(self) -> None:

        if (
            not np.isfinite(
                self.aquifer_permeability_md
            )
            or self.aquifer_permeability_md <= 0
        ):
            raise ValueError(
                "Aquifer permeability must be > 0 md."
            )

        if (
            not np.isfinite(
                self.aquifer_porosity
            )
            or not (
                0 < self.aquifer_porosity < 1
            )
        ):
            raise ValueError(
                "Aquifer porosity must be between 0 and 1."
            )

        if (
            not np.isfinite(
                self.aquifer_thickness_ft
            )
            or self.aquifer_thickness_ft <= 0
        ):
            raise ValueError(
                "Aquifer thickness must be > 0 ft."
            )

        if (
            not np.isfinite(
                self.total_compressibility_psi_inv
            )
            or self.total_compressibility_psi_inv <= 0
        ):
            raise ValueError(
                "Total compressibility must be > 0 psi^-1."
            )

        if (
            not np.isfinite(
                self.water_viscosity_cp
            )
            or self.water_viscosity_cp <= 0
        ):
            raise ValueError(
                "Water viscosity must be > 0 cp."
            )

        if (
            not np.isfinite(
                self.reservoir_radius_ft
            )
            or self.reservoir_radius_ft <= 0
        ):
            raise ValueError(
                "Reservoir radius must be > 0 ft."
            )

        if (
            not np.isfinite(
                self.encroachment_angle_deg
            )
            or not (
                0 < self.encroachment_angle_deg <= 360
            )
        ):
            raise ValueError(
                "Encroachment angle must be "
                "greater than 0 and <= 360 degrees."
            )

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    def get_name(self) -> str:
        return (
            "Van Everdingen-Hurst "
            "Infinite-Acting Radial Aquifer"
        )

    def get_parameters(self) -> dict:

        return {
            "model": self.get_name(),

            "aquifer_permeability_md":
                self.aquifer_permeability_md,

            "aquifer_porosity":
                self.aquifer_porosity,

            "aquifer_thickness_ft":
                self.aquifer_thickness_ft,

            "total_compressibility_psi_inv":
                self.total_compressibility_psi_inv,

            "water_viscosity_cp":
                self.water_viscosity_cp,

            "reservoir_radius_ft":
                self.reservoir_radius_ft,

            "encroachment_angle_deg":
                self.encroachment_angle_deg,

            "aquifer_constant_rb_per_psi":
                self.aquifer_constant_rb_per_psi,
        }

    # ========================================================
    # DIMENSIONLESS TIME
    # ========================================================

    def dimensionless_time(
        self,
        time_days: Iterable[float],
    ) -> np.ndarray:
        """
        Calculate dimensionless time t_D.

        Field-unit formulation:

            t_D =
                0.0002637 k t
                ----------------
                phi mu c_t r_e^2
        """

        t = np.asarray(
            time_days,
            dtype=float,
        )

        if np.any(~np.isfinite(t)):
            raise ValueError(
                "Time history contains NaN or infinite values."
            )

        if np.any(t < 0):
            raise ValueError(
                "Time values cannot be negative."
            )

        denominator = (
            self.aquifer_porosity
            * self.water_viscosity_cp
            * self.total_compressibility_psi_inv
            * self.reservoir_radius_ft**2
        )

        return (
            0.0002637
            * self.aquifer_permeability_md
            * t
            / denominator
        )

    # ========================================================
    # DIMENSIONLESS CUMULATIVE INFLUX Q_D
    # ========================================================

    @staticmethod
    def dimensionless_influx(
        tD: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Return dimensionless cumulative influx Q_D(t_D).

        The classical Van Everdingen-Hurst solution was
        originally presented using dimensionless influence
        functions/tables.

        The values below represent the infinite-acting
        radial constant-terminal-pressure solution.

        Logarithmic interpolation is used between tabulated
        values.

        For very small t_D, the early-time asymptotic
        behavior is used.

        For values beyond the table, a late-time asymptotic
        approximation is used.
        """

        values = np.asarray(
            tD,
            dtype=float,
        )

        if np.any(~np.isfinite(values)):
            raise ValueError(
                "Dimensionless time contains NaN or infinite values."
            )

        if np.any(values < 0):
            raise ValueError(
                "Dimensionless time cannot be negative."
            )

        scalar_input = values.ndim == 0

        values = np.atleast_1d(values)

        result = np.zeros_like(
            values,
            dtype=float,
        )

        # ----------------------------------------------------
        # Classical infinite-acting radial Q_D table
        #
        # Values are representative of the standard
        # Van Everdingen-Hurst dimensionless solution.
        # ----------------------------------------------------

        td_table = np.array(
            [
                0.00,
                0.01,
                0.05,
                0.10,
                0.15,
                0.20,
                0.30,
                0.40,
                0.50,
                0.60,
                0.70,
                0.80,
                1.00,
                2.00,
                3.00,
                4.00,
                5.00,
                6.00,
                7.00,
                8.00,
                9.00,
                10.00,
                15.00,
                20.00,
                25.00,
                30.00,
                40.00,
                50.00,
                60.00,
                70.00,
                80.00,
                90.00,
                100.00,
                150.00,
                200.00,
                300.00,
                400.00,
                500.00,
                600.00,
                700.00,
                800.00,
                900.00,
                1000.00,
                1500.00,
                2000.00,
                5000.00,
                10000.00,
            ],
            dtype=float,
        )

        qd_table = np.array(
            [
                0.000,
                0.112,
                0.276,
                0.404,
                0.520,
                0.617,
                0.796,
                0.949,
                1.020,
                1.169,
                1.275,
                1.395,
                1.569,
                2.447,
                3.202,
                3.893,
                4.539,
                5.154,
                5.765,
                6.390,
                6.965,
                7.537,
                10.235,
                12.747,
                15.182,
                17.482,
                21.928,
                26.166,
                30.249,
                34.224,
                37.966,
                41.843,
                40.026,
                56.098,
                71.691,
                102.39,
                132.72,
                162.62,
                192.30,
                221.82,
                251.15,
                280.32,
                309.37,
                453.37,
                596.28,
                1253.10,
                2388.00,
            ],
            dtype=float,
        )

        # ----------------------------------------------------
        # Early-time approximation
        #
        # Q_D ≈ 2 sqrt(t_D / pi)
        #
        # This captures the correct t_D^0.5 behavior.
        # ----------------------------------------------------

        early_mask = values < 0.01

        result[
            early_mask
        ] = (
            2.0
            * np.sqrt(
                values[
                    early_mask
                ] / np.pi
            )
        )

        # ----------------------------------------------------
        # Tabulated region
        # ----------------------------------------------------

        table_mask = (
            (values >= 0.01)
            &
            (values <= 10000.0)
        )

        if np.any(table_mask):

            result[
                table_mask
            ] = np.interp(
                values[
                    table_mask
                ],
                td_table,
                qd_table,
            )

        # ----------------------------------------------------
        # Late-time approximation
        #
        # Approximation for the infinite-acting solution.
        #
        # The tabulated solution is preferred whenever
        # possible.
        # ----------------------------------------------------

        late_mask = values > 10000.0

        if np.any(late_mask):

            td_late = values[
                late_mask
            ]

            result[
                late_mask
            ] = (
                4.0
                * td_late
                /
                (
                    np.log(td_late)
                    + 0.80907
                )
            )

        if scalar_input:
            return float(
                result[0]
            )

        return result

    # ========================================================
    # PRESSURE-HISTORY VALIDATION
    # ========================================================

    @staticmethod
    def _validate_history(
        pressure_psia: Iterable[float],
        time_days: Iterable[float],
    ) -> tuple[np.ndarray, np.ndarray]:

        pressure = np.asarray(
            pressure_psia,
            dtype=float,
        )

        time = np.asarray(
            time_days,
            dtype=float,
        )

        if pressure.ndim != 1:
            raise ValueError(
                "Pressure history must be one-dimensional."
            )

        if time.ndim != 1:
            raise ValueError(
                "Time history must be one-dimensional."
            )

        if len(pressure) != len(time):
            raise ValueError(
                "Pressure and time histories "
                "must have the same length."
            )

        if len(pressure) < 2:
            raise ValueError(
                "At least two pressure-history points "
                "are required."
            )

        if np.any(~np.isfinite(pressure)):
            raise ValueError(
                "Pressure history contains NaN "
                "or infinite values."
            )

        if np.any(~np.isfinite(time)):
            raise ValueError(
                "Time history contains NaN "
                "or infinite values."
            )

        if np.any(pressure <= 0):
            raise ValueError(
                "Reservoir pressure must be > 0 psia."
            )

        if np.any(time < 0):
            raise ValueError(
                "Time cannot be negative."
            )

        if not np.isclose(
            time[0],
            0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "Time history must start at 0 days."
            )

        if np.any(
            np.diff(time) <= 0
        ):
            raise ValueError(
                "Time history must be strictly increasing."
            )

        return pressure, time

    # ========================================================
    # SUPERPOSITION
    # ========================================================

    def calculate_influx(
        self,
        pressure_psia: Iterable[float],
        time_days: Iterable[float],
    ) -> pd.DataFrame:
        """
        Calculate cumulative water influx using
        Van Everdingen-Hurst superposition.

        Parameters
        ----------
        pressure_psia:
            Reservoir pressure history at the
            reservoir-aquifer boundary.

        time_days:
            Corresponding time history in days.

        Returns
        -------
        pandas.DataFrame

        Columns:
            time_days
            pressure_psia
            dimensionless_time
            pressure_change_psia
            influx_rate_rb_per_day
            delta_We_rb
            We_rb
        """

        pressure, time = (
            self._validate_history(
                pressure_psia,
                time_days,
            )
        )

        # Dimensionless time
        tD = self.dimensionless_time(
            time
        )

        n = len(
            pressure
        )

        We = np.zeros(
            n,
            dtype=float,
        )

        # ----------------------------------------------------
        # Pressure-step superposition
        #
        # At each pressure change:
        #
        # Delta P_j = P_(j-1) - P_j
        #
        # The change acts from t_j onward.
        # ----------------------------------------------------

        pressure_changes = np.zeros(
            n,
            dtype=float,
        )

        for j in range(
            1,
            n,
        ):

            pressure_changes[j] = (
                pressure[j - 1]
                - pressure[j]
            )

        # ----------------------------------------------------
        # Calculate cumulative influx
        # ----------------------------------------------------

        for n_idx in range(
            1,
            n,
        ):

            cumulative = 0.0

            for j in range(
                1,
                n_idx + 1,
            ):

                elapsed_tD = (
                    tD[n_idx]
                    - tD[j]
                )

                if elapsed_tD <= 0:
                    continue

                qd = self.dimensionless_influx(
                    elapsed_tD
                )

                cumulative += (
                    pressure_changes[j]
                    * qd
                )

            We[n_idx] = (
                self.aquifer_constant_rb_per_psi
                * cumulative
            )

        # ----------------------------------------------------
        # Calculate incremental influx
        # ----------------------------------------------------

        delta_We = np.zeros(
            n,
            dtype=float,
        )

        delta_We[1:] = (
            np.diff(We)
        )

        # ----------------------------------------------------
        # Average influx rate over each interval
        # ----------------------------------------------------

        influx_rate = np.zeros(
            n,
            dtype=float,
        )

        dt = np.diff(
            time
        )

        influx_rate[1:] = (
            delta_We[1:]
            / dt
        )

        # ----------------------------------------------------
        # Build result
        # ----------------------------------------------------

        result = pd.DataFrame(
            {
                "time_days":
                    time,

                "pressure_psia":
                    pressure,

                "dimensionless_time":
                    tD,

                "pressure_change_psia":
                    pressure_changes,

                "influx_rate_rb_per_day":
                    influx_rate,

                "delta_We_rb":
                    delta_We,

                "We_rb":
                    We,
            }
        )

        return result