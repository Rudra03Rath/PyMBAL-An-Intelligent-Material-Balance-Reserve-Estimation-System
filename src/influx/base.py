"""
PyMBAL - Water Influx Base Interface

Defines the common interface that all aquifer / water-influx
models must follow.

Current planned implementations:

    1. Schilthuis
    2. Fetkovich
    3. van Everdingen-Hurst

The material-balance engine should interact with aquifer
models through this interface rather than depending on a
specific aquifer formulation.
"""

from abc import ABC, abstractmethod

import pandas as pd


class WaterInfluxModel(ABC):
    """
    Abstract base class for water-influx models.

    Every water-influx model must provide:

        - a model name
        - a method to calculate cumulative water influx
        - basic input validation
    """

    def __init__(self, name):
        """
        Parameters
        ----------
        name : str
            Human-readable model name.
        """

        if not isinstance(name, str):
            raise TypeError(
                "Water influx model name must be a string."
            )

        if not name.strip():
            raise ValueError(
                "Water influx model name cannot be empty."
            )

        self.name = name

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    def get_name(self):
        """
        Return the model name.
        """

        return self.name

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    @staticmethod
    def validate_pressure_history(
        pressure_psia
    ):
        """
        Validate a reservoir pressure history.

        Parameters
        ----------
        pressure_psia : array-like or pandas Series
            Reservoir pressure history in psia.

        Returns
        -------
        pandas.Series
            Validated pressure series.
        """

        pressure = pd.Series(
            pressure_psia,
            dtype=float
        )

        if pressure.empty:
            raise ValueError(
                "Pressure history cannot be empty."
            )

        if pressure.isna().any():
            raise ValueError(
                "Pressure history contains missing values."
            )

        if not (pressure > 0).all():
            raise ValueError(
                "All reservoir pressures must be "
                "greater than zero."
            )

        return pressure.reset_index(
            drop=True
        )

    @staticmethod
    def validate_time_history(
        time_days
    ):
        """
        Validate a time history.

        Parameters
        ----------
        time_days : array-like or pandas Series
            Time history in days.

        Returns
        -------
        pandas.Series
            Validated time series.
        """

        time = pd.Series(
            time_days,
            dtype=float
        )

        if time.empty:
            raise ValueError(
                "Time history cannot be empty."
            )

        if time.isna().any():
            raise ValueError(
                "Time history contains missing values."
            )

        if (time < 0).any():
            raise ValueError(
                "Time values cannot be negative."
            )

        if len(time) > 1:

            dt = time.diff().iloc[1:]

            if not (dt > 0).all():
                raise ValueError(
                    "Time history must be strictly "
                    "increasing."
                )

        return time.reset_index(
            drop=True
        )

    @staticmethod
    def validate_positive_parameter(
        value,
        name
    ):
        """
        Validate a strictly positive model parameter.
        """

        try:
            value = float(value)
        except (TypeError, ValueError):

            raise TypeError(
                f"{name} must be numeric."
            )

        if value <= 0:

            raise ValueError(
                f"{name} must be greater than zero."
            )

        return value

    # ========================================================
    # WATER INFLUX CALCULATION
    # ========================================================

    @abstractmethod
    def calculate_influx(
        self,
        pressure_psia,
        time_days,
        **kwargs
    ):
        """
        Calculate cumulative water influx.

        Every child model must implement this method.

        Returns
        -------
        pandas.DataFrame

            The returned DataFrame must contain at least:

                time_days
                pressure_psia
                We_rb

            Additional model-specific diagnostic columns
            may also be included.
        """

        raise NotImplementedError

    # ========================================================
    # MODEL SUMMARY
    # ========================================================

    def summary(self):
        """
        Return basic model information.
        """

        return {
            "model": self.name
        }

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(self):

        return (
            f"{self.__class__.__name__}"
            f"(name='{self.name}')"
        )