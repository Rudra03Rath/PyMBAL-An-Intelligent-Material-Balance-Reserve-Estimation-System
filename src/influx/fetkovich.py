"""
PyMBAL - Fetkovich Water Influx Model

Finite-aquifer Fetkovich model.

The model represents the aquifer as a finite water-storage system
connected to the reservoir through an aquifer productivity index J.

Model parameters
----------------
initial_aquifer_pressure_psia
    Initial aquifer pressure, psia.

initial_water_volume_rb
    Initial aquifer water volume, rb.

total_compressibility_psi_inv
    Total aquifer compressibility, psi^-1.

productivity_index_rb_per_day_psi
    Aquifer productivity index, rb/(day*psi).

Governing relationships
-----------------------

Aquifer storage capacity:

    Ca = Wi * ct

Water influx rate:

    qw = J * (Pa - P)

Aquifer material balance:

    dWe = Ca * (Pa_old - Pa_new)

For a time step dt, using the new aquifer pressure
implicitly:

    dWe = qw * dt

    qw = J * (Pa_new - P)

    Pa_new = Pa_old - dWe / Ca

Combining these:

    qw = J * (Pa_old - P)
         / (1 + J*dt/Ca)

The formulation gives a stable finite-aquifer response and ensures
that cumulative influx reduces aquifer pressure.

Output
------

A pandas DataFrame containing:

    time_days
    pressure_psia
    aquifer_pressure_psia
    pressure_difference_psia
    influx_rate_rb_per_day
    delta_We_rb
    We_rb
"""


import numpy as np
import pandas as pd

from .base import WaterInfluxModel


class FetkovichAquifer(WaterInfluxModel):
    """
    Finite-aquifer Fetkovich water-influx model.
    """

    def __init__(
        self,
        initial_aquifer_pressure_psia: float,
        initial_water_volume_rb: float,
        total_compressibility_psi_inv: float,
        productivity_index_rb_per_day_psi: float,
    ):
        """
        Initialize the Fetkovich aquifer.

        Parameters
        ----------
        initial_aquifer_pressure_psia : float
            Initial aquifer pressure, psia.

        initial_water_volume_rb : float
            Initial aquifer water volume, rb.

        total_compressibility_psi_inv : float
            Total aquifer compressibility, psi^-1.

        productivity_index_rb_per_day_psi : float
            Aquifer productivity index,
            rb/(day*psi).
        """

        # ----------------------------------------------------
        # Validate parameters
        # ----------------------------------------------------

        if (
            not np.isfinite(
                initial_aquifer_pressure_psia
            )
            or initial_aquifer_pressure_psia <= 0
        ):
            raise ValueError(
                "Initial aquifer pressure must be "
                "greater than zero."
            )

        if (
            not np.isfinite(
                initial_water_volume_rb
            )
            or initial_water_volume_rb <= 0
        ):
            raise ValueError(
                "Initial aquifer water volume must be "
                "greater than zero."
            )

        if (
            not np.isfinite(
                total_compressibility_psi_inv
            )
            or total_compressibility_psi_inv <= 0
        ):
            raise ValueError(
                "Total aquifer compressibility must be "
                "greater than zero."
            )

        if (
            not np.isfinite(
                productivity_index_rb_per_day_psi
            )
            or productivity_index_rb_per_day_psi <= 0
        ):
            raise ValueError(
                "Productivity index must be "
                "greater than zero."
            )

        # ----------------------------------------------------
        # Store parameters
        # ----------------------------------------------------

        self.initial_aquifer_pressure_psia = float(
            initial_aquifer_pressure_psia
        )

        self.initial_water_volume_rb = float(
            initial_water_volume_rb
        )

        self.total_compressibility_psi_inv = float(
            total_compressibility_psi_inv
        )

        self.productivity_index_rb_per_day_psi = float(
            productivity_index_rb_per_day_psi
        )

        # ----------------------------------------------------
        # Aquifer storage capacity
        #
        # Ca = Wi * ct
        #
        # Units:
        # rb * psi^-1 = rb/psi
        # ----------------------------------------------------

        self.storage_capacity_rb_per_psi = (
            self.initial_water_volume_rb
            *
            self.total_compressibility_psi_inv
        )

    # ========================================================
    # MODEL NAME
    # ========================================================

    def get_name(self) -> str:
        """
        Return model name.
        """

        return "Fetkovich Finite Aquifer"

    # ========================================================
    # WATER INFLUX CALCULATION
    # ========================================================

    def calculate_influx(
        self,
        pressure_psia,
        time_days,
    ) -> pd.DataFrame:
        """
        Calculate water influx using the Fetkovich model.

        Parameters
        ----------
        pressure_psia : array-like
            Reservoir pressure history, psia.

        time_days : array-like
            Time history, days.

        Returns
        -------
        pandas.DataFrame
            Water-influx history.
        """

        # ----------------------------------------------------
        # Convert inputs to arrays
        # ----------------------------------------------------

        pressure = np.asarray(
            pressure_psia,
            dtype=float,
        )

        time = np.asarray(
            time_days,
            dtype=float,
        )

        # ----------------------------------------------------
        # Basic input validation
        # ----------------------------------------------------

        if pressure.ndim != 1:

            raise ValueError(
                "Reservoir pressure must be a "
                "one-dimensional sequence."
            )

        if time.ndim != 1:

            raise ValueError(
                "Time history must be a "
                "one-dimensional sequence."
            )

        if len(pressure) != len(time):

            raise ValueError(
                "Pressure and time histories must "
                "have the same length."
            )

        if len(pressure) == 0:

            raise ValueError(
                "Pressure and time histories "
                "cannot be empty."
            )

        if not np.all(
            np.isfinite(pressure)
        ):

            raise ValueError(
                "Pressure history contains "
                "invalid values."
            )

        if not np.all(
            np.isfinite(time)
        ):

            raise ValueError(
                "Time history contains "
                "invalid values."
            )

        if np.any(
            pressure <= 0
        ):

            raise ValueError(
                "Reservoir pressure must be "
                "greater than zero."
            )

        # ----------------------------------------------------
        # Time must be monotonic
        # ----------------------------------------------------

        if len(time) > 1:

            if np.any(
                np.diff(time) < 0
            ):

                raise ValueError(
                    "Time history must be "
                    "monotonically increasing."
                )

        # ----------------------------------------------------
        # Initialize output arrays
        # ----------------------------------------------------

        n = len(
            pressure
        )

        aquifer_pressure = np.zeros(
            n,
            dtype=float,
        )

        influx_rate = np.zeros(
            n,
            dtype=float,
        )

        delta_We = np.zeros(
            n,
            dtype=float,
        )

        We = np.zeros(
            n,
            dtype=float,
        )

        pressure_difference = np.zeros(
            n,
            dtype=float,
        )

        # ----------------------------------------------------
        # Initial aquifer state
        # ----------------------------------------------------

        aquifer_pressure[0] = (
            self.initial_aquifer_pressure_psia
        )

        pressure_difference[0] = max(
            aquifer_pressure[0]
            -
            pressure[0],
            0.0,
        )

        # ====================================================
        # TIME-STEP CALCULATION
        # ====================================================

        for i in range(1, n):

            # ------------------------------------------------
            # Time step
            # ------------------------------------------------

            dt = (
                time[i]
                -
                time[i - 1]
            )

            if dt < 0:

                raise ValueError(
                    "Time history must be "
                    "monotonically increasing."
                )

            # ------------------------------------------------
            # Previous aquifer pressure
            # ------------------------------------------------

            Pa_old = (
                aquifer_pressure[i - 1]
            )

            # ------------------------------------------------
            # Current reservoir pressure
            # ------------------------------------------------

            P = (
                pressure[i]
            )

            # ------------------------------------------------
            # Pressure drawdown available to aquifer
            # ------------------------------------------------

            pressure_difference[i] = max(
                Pa_old - P,
                0.0,
            )

            # ------------------------------------------------
            # Zero-duration step
            # ------------------------------------------------

            if dt == 0:

                aquifer_pressure[i] = Pa_old

                influx_rate[i] = 0.0

                delta_We[i] = 0.0

                We[i] = We[i - 1]

                continue

            # ------------------------------------------------
            # Fetkovich implicit rate equation
            #
            # q = J(Pa_new - P)
            #
            # Pa_new = Pa_old - q*dt/Ca
            #
            # Therefore:
            #
            # q =
            # J(Pa_old - P)
            # ----------------
            # 1 + J*dt/Ca
            # ------------------------------------------------

            numerator = (
                self.productivity_index_rb_per_day_psi
                *
                max(
                    Pa_old - P,
                    0.0,
                )
            )

            denominator = (
                1.0
                +
                (
                    self.productivity_index_rb_per_day_psi
                    *
                    dt
                    /
                    self.storage_capacity_rb_per_psi
                )
            )

            q = (
                numerator
                /
                denominator
            )

            # ------------------------------------------------
            # Water influx during current time step
            # ------------------------------------------------

            dWe = (
                q
                *
                dt
            )

            # ------------------------------------------------
            # Prevent numerical over-withdrawal
            #
            # Aquifer pressure cannot fall below
            # the current reservoir pressure in this
            # formulation.
            # ------------------------------------------------

            max_dWe = (
                self.storage_capacity_rb_per_psi
                *
                max(
                    Pa_old - P,
                    0.0,
                )
            )

            dWe = min(
                dWe,
                max_dWe,
            )

            # ------------------------------------------------
            # Update aquifer pressure
            # ------------------------------------------------

            Pa_new = (
                Pa_old
                -
                dWe
                /
                self.storage_capacity_rb_per_psi
            )

            # Numerical protection
            Pa_new = max(
                Pa_new,
                P,
            )

            # ------------------------------------------------
            # Store results
            # ------------------------------------------------

            influx_rate[i] = (
                dWe
                /
                dt
            )

            delta_We[i] = (
                dWe
            )

            We[i] = (
                We[i - 1]
                +
                dWe
            )

            aquifer_pressure[i] = (
                Pa_new
            )

        # ====================================================
        # BUILD RESULT DATAFRAME
        # ====================================================

        result = pd.DataFrame(
            {
                "time_days": time,

                "pressure_psia": pressure,

                "aquifer_pressure_psia":
                    aquifer_pressure,

                "pressure_difference_psia":
                    np.maximum(
                        aquifer_pressure
                        -
                        pressure,
                        0.0,
                    ),

                "influx_rate_rb_per_day":
                    influx_rate,

                "delta_We_rb":
                    delta_We,

                "We_rb":
                    We,
            }
        )

        return result

    # ========================================================
    # MODEL SUMMARY
    # ========================================================

    def get_parameters(self) -> dict:
        """
        Return model parameters.
        """

        return {

            "model":
                self.get_name(),

            "initial_aquifer_pressure_psia":
                self.initial_aquifer_pressure_psia,

            "initial_water_volume_rb":
                self.initial_water_volume_rb,

            "total_compressibility_psi_inv":
                self.total_compressibility_psi_inv,

            "productivity_index_rb_per_day_psi":
                self.productivity_index_rb_per_day_psi,

            "storage_capacity_rb_per_psi":
                self.storage_capacity_rb_per_psi,
        }