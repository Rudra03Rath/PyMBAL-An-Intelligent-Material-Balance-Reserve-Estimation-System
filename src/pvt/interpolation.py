"""
PyMBAL - PVT Interpolation Module

Provides interpolation utilities for laboratory PVT data.

Currently supported:
    - Linear interpolation
    - Multiple-property interpolation
    - DataFrame-based PVT interpolation

Typical use:

    Laboratory PVT data
            |
            v
       Interpolation
            |
            v
    Reservoir pressure history
            |
            v
    PVT properties at required pressures
"""

import numpy as np
import pandas as pd


# ============================================================
# Validation
# ============================================================

def _validate_input_arrays(x, y):
    """
    Validate pressure and property arrays.

    Parameters
    ----------
    x : array-like
        Independent variable, normally pressure.

    y : array-like
        PVT property values.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1:
        raise ValueError(
            "Pressure data must be one-dimensional."
        )

    if y.ndim != 1:
        raise ValueError(
            "PVT property data must be one-dimensional."
        )

    if len(x) != len(y):
        raise ValueError(
            "Pressure and property arrays must have "
            "the same length."
        )

    if len(x) < 2:
        raise ValueError(
            "At least two PVT data points are required "
            "for interpolation."
        )

    if np.any(~np.isfinite(x)):
        raise ValueError(
            "Pressure data contains invalid values."
        )

    if np.any(~np.isfinite(y)):
        raise ValueError(
            "PVT property data contains invalid values."
        )

    # Pressure must be strictly increasing
    if np.any(np.diff(x) <= 0):
        raise ValueError(
            "Pressure values must be strictly increasing "
            "and contain no duplicates."
        )

    return x, y


# ============================================================
# Linear Interpolation
# ============================================================

def linear_interpolate(
    pressure,
    pvt_pressure,
    pvt_values,
    allow_extrapolation=False
):
    """
    Linearly interpolate a PVT property.

    Parameters
    ----------
    pressure : float or array-like
        Pressure(s) at which the property is required.

    pvt_pressure : array-like
        Laboratory PVT pressures.

    pvt_values : array-like
        Laboratory PVT property values.

    allow_extrapolation : bool, optional
        If False, pressures outside the laboratory range
        raise an error.

        If True, linear extrapolation is performed.

    Returns
    -------
    float or numpy.ndarray
        Interpolated PVT property.

    Notes
    -----
    Linear interpolation is:

        y = y1 + (x - x1)/(x2 - x1) * (y2 - y1)

    """

    pvt_pressure, pvt_values = _validate_input_arrays(
        pvt_pressure,
        pvt_values
    )

    pressure = np.asarray(
        pressure,
        dtype=float
    )

    if np.any(~np.isfinite(pressure)):
        raise ValueError(
            "Requested pressure contains invalid values."
        )

    min_pressure = pvt_pressure.min()
    max_pressure = pvt_pressure.max()

    # --------------------------------------------------------
    # Check extrapolation
    # --------------------------------------------------------

    outside = (
        (pressure < min_pressure)
        | (pressure > max_pressure)
    )

    if np.any(outside) and not allow_extrapolation:

        raise ValueError(
            "Requested pressure lies outside the "
            "laboratory PVT pressure range.\n"
            f"Available range: "
            f"{min_pressure:.2f} - "
            f"{max_pressure:.2f} psi"
        )

    # --------------------------------------------------------
    # Interpolation
    # --------------------------------------------------------

    if allow_extrapolation:

        result = np.interp(
            pressure,
            pvt_pressure,
            pvt_values
        )

        # np.interp does not extrapolate. It clips to
        # boundary values, so perform explicit linear
        # extrapolation where necessary.

        below = pressure < min_pressure
        above = pressure > max_pressure

        # Lower extrapolation
        if np.any(below):

            slope_low = (
                (pvt_values[1] - pvt_values[0])
                / (pvt_pressure[1] - pvt_pressure[0])
            )

            result[below] = (
                pvt_values[0]
                + slope_low
                * (pressure[below] - pvt_pressure[0])
            )

        # Upper extrapolation
        if np.any(above):

            slope_high = (
                (pvt_values[-1] - pvt_values[-2])
                / (pvt_pressure[-1] - pvt_pressure[-2])
            )

            result[above] = (
                pvt_values[-1]
                + slope_high
                * (
                    pressure[above]
                    - pvt_pressure[-1]
                )
            )

    else:

        result = np.interp(
            pressure,
            pvt_pressure,
            pvt_values
        )

    # Return scalar when scalar input was supplied
    if np.ndim(pressure) == 0:
        return float(result)

    return result


# ============================================================
# Multiple PVT Properties
# ============================================================

def interpolate_pvt(
    pressure,
    pvt_pressure,
    pvt_data,
    allow_extrapolation=False
):
    """
    Interpolate multiple PVT properties simultaneously.

    Parameters
    ----------
    pressure : float or array-like
        Target reservoir pressure(s).

    pvt_pressure : array-like
        Laboratory PVT pressure data.

    pvt_data : dict
        Dictionary containing PVT properties.

        Example:

            {
                "Bo_rb_per_STB": [...],
                "Bg_rb_per_scf": [...],
                "Rs_scf_per_STB": [...],
                "Bw_rb_per_STB": [...]
            }

    allow_extrapolation : bool, optional
        Allow extrapolation outside laboratory range.

    Returns
    -------
    dict
        Dictionary containing interpolated properties.
    """

    pvt_pressure = np.asarray(
        pvt_pressure,
        dtype=float
    )

    if not isinstance(pvt_data, dict):
        raise TypeError(
            "pvt_data must be a dictionary."
        )

    result = {}

    for property_name, values in pvt_data.items():

        result[property_name] = linear_interpolate(
            pressure=pressure,
            pvt_pressure=pvt_pressure,
            pvt_values=values,
            allow_extrapolation=allow_extrapolation
        )

    return result


# ============================================================
# DataFrame-Based Interpolation
# ============================================================

def interpolate_pvt_dataframe(
    target_pressure,
    laboratory_df,
    pressure_column="Pressure_psia",
    pvt_columns=None,
    allow_extrapolation=False
):
    """
    Interpolate laboratory PVT properties directly
    from a pandas DataFrame.

    Parameters
    ----------
    target_pressure : float or array-like
        Required reservoir pressure(s).

    laboratory_df : pandas.DataFrame
        Laboratory PVT dataset.

    pressure_column : str
        Name of the pressure column.

    pvt_columns : list, optional
        PVT property columns to interpolate.

        If None, the following are used when available:

            Bo_rb_per_STB
            Bg_rb_per_scf
            Bw_rb_per_STB
            Rs_scf_per_STB

    allow_extrapolation : bool
        Allow pressure extrapolation.

    Returns
    -------
    pandas.DataFrame
        Interpolated PVT data.
    """

    if not isinstance(
        laboratory_df,
        pd.DataFrame
    ):
        raise TypeError(
            "laboratory_df must be a pandas DataFrame."
        )

    if pressure_column not in laboratory_df.columns:
        raise ValueError(
            f"Pressure column '{pressure_column}' "
            "not found in laboratory data."
        )

    # --------------------------------------------------------
    # Default PVT properties
    # --------------------------------------------------------

    if pvt_columns is None:

        possible_columns = [
            "Bo_rb_per_STB",
            "Bg_rb_per_scf",
            "Bw_rb_per_STB",
            "Rs_scf_per_STB"
        ]

        pvt_columns = [
            column
            for column in possible_columns
            if column in laboratory_df.columns
        ]

    if not pvt_columns:

        raise ValueError(
            "No PVT property columns were found."
        )

    # --------------------------------------------------------
    # Sort laboratory data by pressure
    # --------------------------------------------------------

    laboratory_df = (
        laboratory_df
        .sort_values(pressure_column)
        .reset_index(drop=True)
    )

    pvt_pressure = laboratory_df[
        pressure_column
    ].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Build property dictionary
    # --------------------------------------------------------

    pvt_data = {}

    for column in pvt_columns:

        pvt_data[column] = laboratory_df[
            column
        ].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Interpolate
    # --------------------------------------------------------

    interpolated = interpolate_pvt(
        pressure=target_pressure,
        pvt_pressure=pvt_pressure,
        pvt_data=pvt_data,
        allow_extrapolation=allow_extrapolation
    )

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    target_pressure = np.asarray(
        target_pressure,
        dtype=float
    )

    if target_pressure.ndim == 0:

        target_pressure = np.array([
            target_pressure
        ])

    result = pd.DataFrame({
        pressure_column: target_pressure
    })

    for column, values in interpolated.items():

        values = np.asarray(values)

        if values.ndim == 0:
            values = np.array([values])

        result[column] = values

    return result