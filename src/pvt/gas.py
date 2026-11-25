"""
PyMBAL - Gas PVT Correlations

This module contains gas PVT calculations.

Currently implemented:
    - Sutton pseudo-critical properties
    - Pseudo-reduced pressure
    - Pseudo-reduced temperature
    - Hall-Yarborough Z-factor
    - Gas formation volume factor (Bg)

Field units:
    Pressure       : psia
    Temperature    : °F
    Gas gravity    : specific gravity relative to air
    Bg              : rb/scf
"""

import numpy as np


# ============================================================
# Sutton Pseudo-Critical Properties
# ============================================================

def sutton_pseudo_critical(gas_gravity):
    """
    Calculate pseudo-critical pressure and temperature
    using Sutton correlations.

    Parameters
    ----------
    gas_gravity : float or array-like
        Gas specific gravity relative to air.

    Returns
    -------
    tuple
        (Ppc, Tpc)

        Ppc : pseudo-critical pressure, psia
        Tpc : pseudo-critical temperature, °R
    """

    gas_gravity = np.asarray(gas_gravity, dtype=float)

    if np.any(gas_gravity <= 0):
        raise ValueError("Gas gravity must be greater than zero.")

    # Sutton correlations
    tpc = 169.2 + 349.5 * gas_gravity - 74.0 * gas_gravity ** 2

    ppc = 756.8 - 131.0 * gas_gravity - 3.6 * gas_gravity ** 2

    return ppc, tpc


# ============================================================
# Pseudo-Reduced Properties
# ============================================================

def pseudo_reduced_pressure(pressure, ppc):
    """
    Calculate pseudo-reduced pressure.

        Ppr = P / Ppc

    Parameters
    ----------
    pressure : float or array-like
        Pressure, psia.

    ppc : float or array-like
        Pseudo-critical pressure, psia.

    Returns
    -------
    float or numpy.ndarray
        Pseudo-reduced pressure.
    """

    pressure = np.asarray(pressure, dtype=float)
    ppc = np.asarray(ppc, dtype=float)

    if np.any(pressure <= 0):
        raise ValueError("Pressure must be greater than zero.")

    if np.any(ppc <= 0):
        raise ValueError(
            "Pseudo-critical pressure must be greater than zero."
        )

    return pressure / ppc


def pseudo_reduced_temperature(temperature, tpc):
    """
    Calculate pseudo-reduced temperature.

        Tpr = T / Tpc

    Parameters
    ----------
    temperature : float
        Temperature, °F.

    tpc : float
        Pseudo-critical temperature, °R.

    Returns
    -------
    float
        Pseudo-reduced temperature.
    """

    temperature_rankine = np.asarray(temperature, dtype=float) + 459.67
    tpc = np.asarray(tpc, dtype=float)

    if np.any(temperature_rankine <= 0):
        raise ValueError("Temperature must be above absolute zero.")

    if np.any(tpc <= 0):
        raise ValueError(
            "Pseudo-critical temperature must be greater than zero."
        )

    return temperature_rankine / tpc


# ============================================================
# Hall-Yarborough Z-Factor
# ============================================================

def hall_yarborough_z(
    pressure,
    temperature,
    gas_gravity,
    tolerance=1e-8,
    max_iterations=100
):
    """
    Calculate gas compressibility factor using
    the Hall-Yarborough correlation.

    The correlation solves iteratively for reduced density.

    Parameters
    ----------
    pressure : float or array-like
        Pressure, psia.

    temperature : float
        Temperature, °F.

    gas_gravity : float
        Gas specific gravity.

    tolerance : float, optional
        Convergence tolerance.

    max_iterations : int, optional
        Maximum number of iterations.

    Returns
    -------
    float or numpy.ndarray
        Compressibility factor, Z.
    """

    pressure = np.asarray(pressure, dtype=float)

    ppc, tpc = sutton_pseudo_critical(gas_gravity)

    ppr = pseudo_reduced_pressure(
        pressure,
        ppc
    )

    tpr = pseudo_reduced_temperature(
        temperature,
        tpc
    )

    # Reciprocal pseudo-reduced temperature
    t = 1.0 / tpr

    # Hall-Yarborough coefficients
    A1 = 0.06125 * t * np.exp(
        -1.2 * (1.0 - t) ** 2
    )

    A2 = (
        14.76 * t
        - 9.76 * t ** 2
        + 4.58 * t ** 3
    )

    A3 = (
        90.7 * t
        - 242.2 * t ** 2
        + 42.4 * t ** 3
    )

    A4 = 2.18 + 2.82 * t

    # --------------------------------------------------------
    # Initial guess
    # --------------------------------------------------------

    y = np.full_like(
        ppr,
        0.001,
        dtype=float
    )

    # --------------------------------------------------------
    # Newton-Raphson iteration
    # --------------------------------------------------------

    converged = np.zeros_like(
        ppr,
        dtype=bool
    )

    for _ in range(max_iterations):

        denominator = (1.0 - y)

        f = (
            -A1 * ppr
            + (
                y
                + y ** 2
                + y ** 3
                - y ** 4
            ) / denominator ** 3
            - A2 * y ** 2
            + A3 * y ** A4
        )

        # Numerical derivative
        dy = 1e-6

        y_plus = y + dy

        denominator_plus = 1.0 - y_plus

        f_plus = (
            -A1 * ppr
            + (
                y_plus
                + y_plus ** 2
                + y_plus ** 3
                - y_plus ** 4
            ) / denominator_plus ** 3
            - A2 * y_plus ** 2
            + A3 * y_plus ** A4
        )

        derivative = (f_plus - f) / dy

        # Newton-Raphson update
        y_new = y - f / derivative

        # Keep reduced density physically meaningful
        y_new = np.clip(
            y_new,
            1e-12,
            0.999
        )

        difference = np.abs(y_new - y)

        newly_converged = difference < tolerance

        converged = converged | newly_converged

        y = y_new

        if np.all(converged):
            break

    if not np.all(converged):
        raise RuntimeError(
            "Hall-Yarborough Z-factor calculation "
            "did not converge."
        )

    # --------------------------------------------------------
    # Calculate Z
    # --------------------------------------------------------

    z = (
        A1 * ppr
    ) / y

    if np.any(z <= 0):
        raise ValueError(
            "Calculated Z-factor contains "
            "non-positive values."
        )

    return z


# ============================================================
# Gas Formation Volume Factor
# ============================================================

def gas_fvf(
    pressure,
    temperature,
    z
):
    """
    Calculate gas formation volume factor.

        Bg = 0.00504 * Z * T / P

    Field units:
        Bg : rb/scf
        T  : °R
        P  : psia

    Parameters
    ----------
    pressure : float or array-like
        Pressure, psia.

    temperature : float
        Temperature, °F.

    z : float or array-like
        Gas compressibility factor.

    Returns
    -------
    float or numpy.ndarray
        Gas formation volume factor, rb/scf.
    """

    pressure = np.asarray(pressure, dtype=float)
    z = np.asarray(z, dtype=float)

    if np.any(pressure <= 0):
        raise ValueError(
            "Pressure must be greater than zero."
        )

    if np.any(z <= 0):
        raise ValueError(
            "Z-factor must be greater than zero."
        )

    temperature_rankine = temperature + 459.67

    bg = (
        0.00504
        * z
        * temperature_rankine
        / pressure
    )

    return bg


# ============================================================
# Complete Gas PVT Calculation
# ============================================================

def gas_pvt(
    pressure,
    temperature,
    gas_gravity
):
    """
    Calculate a complete set of gas PVT properties.

    Parameters
    ----------
    pressure : float or array-like
        Pressure, psia.

    temperature : float
        Temperature, °F.

    gas_gravity : float
        Gas specific gravity.

    Returns
    -------
    dict
        Dictionary containing:

            Ppc
            Tpc
            Ppr
            Tpr
            Z
            Bg
    """

    ppc, tpc = sutton_pseudo_critical(
        gas_gravity
    )

    ppr = pseudo_reduced_pressure(
        pressure,
        ppc
    )

    tpr = pseudo_reduced_temperature(
        temperature,
        tpc
    )

    z = hall_yarborough_z(
        pressure,
        temperature,
        gas_gravity
    )

    bg = gas_fvf(
        pressure,
        temperature,
        z
    )

    return {
        "Ppc_psia": ppc,
        "Tpc_R": tpc,
        "Ppr": ppr,
        "Tpr": tpr,
        "Z": z,
        "Bg_rb_per_scf": bg
    }