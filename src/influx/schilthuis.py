"""
PyMBAL - Schilthuis Steady-State Water Influx Model

Model:

    q_w = C (Pe - P)

where:

    q_w = water influx rate, rb/day
    C   = aquifer influx constant,
          rb/(day*psi)
    Pe  = external aquifer pressure, psia
    P   = reservoir pressure, psia

Cumulative influx is obtained by numerical integration:

    We(t) = integral[q_w dt]

For discrete pressure data:

    dWe_i = C (Pe - P_i) dt_i

and:

    We_i = We_(i-1) + dWe_i

This implementation assumes:

    - steady-state aquifer behavior
    - constant external aquifer pressure
    - constant influx coefficient C
    - no water influx before the first time point
    - reservoir pressure is supplied as a history
"""

import numpy as np
import pandas as pd

from .base import WaterInfluxModel


class SchilthuisAquifer(WaterInfluxModel):
    """
    Schilthuis steady-state aquifer model.

    Parameters
    ----------
    aquifer_pressure_psia : float
        Constant external aquifer pressure, psia.

    influx_constant_rb_per_day_psi : float
        Aquifer influx coefficient C with units:

            rb / (day * psi)

        Therefore:

            q_w = C (Pe - P)

        gives q_w in rb/day.
    """

    def __init__(
        self,
        aquifer_pressure_psia,
        influx_constant_rb_per_day_psi,
    ):

        super().__init__(
            name="Schilthuis Steady-State"
        )

        # ----------------------------------------------------
        # Aquifer pressure
        # ----------------------------------------------------

        self.aquifer_pressure_psia = (
            self.validate_positive_parameter(
                aquifer_pressure_psia,
                "aquifer_pressure_psia",
            )
        )

        # ----------------------------------------------------
        # Influx coefficient
        # ----------------------------------------------------

        self.influx_constant_rb_per_day_psi = (
            self.validate_positive_parameter(
                influx_constant_rb_per_day_psi,
                "influx_constant_rb_per_day_psi",
            )
        )

    # ========================================================
    # INFLOW CALCULATION
    # ========================================================

    def calculate_influx(
        self,
        pressure_psia,
        time_days,
        **kwargs
    ):
        """
        Calculate cumulative water influx.

        Parameters
        ----------
        pressure_psia : array-like
            Reservoir pressure history, psia.

        time_days : array-like
            Time history, days.

        Returns
        -------
        pandas.DataFrame

            Columns:

                time_days
                pressure_psia
                pressure_difference_psia
                influx_rate_rb_per_day
                delta_We_rb
                We_rb
        """

        # ====================================================
        # VALIDATE INPUTS
        # ====================================================

        pressure = (
            self.validate_pressure_history(
                pressure_psia
            )
        )

        time = (
            self.validate_time_history(
                time_days
            )
        )

        if len(pressure) != len(time):

            raise ValueError(
                "Pressure and time histories must "
                "have the same length."
            )

        # ====================================================
        # PRESSURE DIFFERENCE
        # ====================================================

        pressure_difference = (

            self.aquifer_pressure_psia

            -

            pressure
        )

        # ----------------------------------------------------
        # Negative pressure difference means the reservoir
        # pressure is above the assumed aquifer pressure.
        #
        # That does not produce positive water influx.
        #
        # Therefore clamp the pressure difference at zero.
        # ----------------------------------------------------

        effective_pressure_difference = (
            pressure_difference.clip(
                lower=0.0
            )
        )

        # ====================================================
        # INFLOW RATE
        # ====================================================

        influx_rate = (

            self.influx_constant_rb_per_day_psi

            *

            effective_pressure_difference
        )

        # ====================================================
        # TIME INTERVALS
        # ====================================================

        dt = (
            time.diff()
            .fillna(0.0)
        )

        # ====================================================
        # INCREMENTAL WATER INFLUX
        # ====================================================

        delta_We = (

            influx_rate

            *

            dt
        )

        # ====================================================
        # CUMULATIVE WATER INFLUX
        # ====================================================

        We = (
            delta_We
            .cumsum()
        )

        # ====================================================
        # BUILD RESULT
        # ====================================================

        result = pd.DataFrame(
            {
                "time_days":
                    time,

                "pressure_psia":
                    pressure,

                "aquifer_pressure_psia":
                    self.aquifer_pressure_psia,

                "pressure_difference_psia":
                    effective_pressure_difference,

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

    def summary(self):

        return {

            "model":
                self.name,

            "aquifer_pressure_psia":
                self.aquifer_pressure_psia,

            "influx_constant_rb_per_day_psi":
                self.influx_constant_rb_per_day_psi,

            "assumption":
                "Steady-state aquifer",

        }