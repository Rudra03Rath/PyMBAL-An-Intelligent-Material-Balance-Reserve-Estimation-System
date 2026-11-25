"""
PyMBAL - Water PVT Correlations

This module contains correlations for calculating
formation-water PVT properties.

Currently implemented:
    - McCain Water Formation Volume Factor (Bw)
    - McCain Water Viscosity at Atmospheric Pressure
    - McCain Water Viscosity at Reservoir Pressure

Field units:
    Pressure       : psia
    Temperature    : °F
    Salinity       : weight percent (%)
    Bw             : rb/STB
    Viscosity      : cP
"""

import numpy as np


# ============================================================
# Utility Validation
# ============================================================

def _validate_pressure(pressure):
    """
    Validate pressure input.
    """

    pressure = np.asarray(pressure, dtype=float)

    if np.any(pressure <= 0):
        raise ValueError(
            "Pressure must be greater than zero."
        )

    return pressure


def _validate_temperature(temperature):
    """
    Validate reservoir temperature.
    """

    temperature = np.asarray(
        temperature,
        dtype=float
    )

    if np.any(temperature <= -459.67):
        raise ValueError(
            "Temperature must be above absolute zero."
        )

    return temperature


def _validate_salinity(salinity):
    """
    Validate water salinity.

    Salinity is expressed as weight percent.
    """

    salinity = np.asarray(
        salinity,
        dtype=float
    )

    if np.any(salinity < 0):
        raise ValueError(
            "Salinity cannot be negative."
        )

    return salinity


# ============================================================
# McCain Water Formation Volume Factor
# ============================================================

def mccain_bw(
    pressure,
    temperature
):
    """
    Calculate water formation volume factor using
    McCain's correlation.

    Equation:

        Bw = (1 + ΔVwt)(1 + ΔVwp)

    where:

        ΔVwt =
            -1.00010e-2
            + 1.33391e-4*T
            + 5.50654e-7*T²

        ΔVwp =
            -1.95301e-9*P*T
            - 1.72834e-13*P²*T
            - 3.58922e-7*P
            - 2.25341e-10*P²

    Parameters
    ----------
    pressure : float or array-like
        Reservoir pressure, psia.

    temperature : float
        Reservoir temperature, °F.

    Returns
    -------
    float or numpy.ndarray
        Water formation volume factor, rb/STB.
    """

    pressure = _validate_pressure(pressure)
    temperature = _validate_temperature(
        temperature
    )

    # --------------------------------------------------------
    # Temperature contribution
    # --------------------------------------------------------

    delta_vwt = (
        -1.00010e-2
        + 1.33391e-4 * temperature
        + 5.50654e-7 * temperature ** 2
    )

    # --------------------------------------------------------
    # Pressure contribution
    # --------------------------------------------------------

    delta_vwp = (
        -1.95301e-9 * pressure * temperature
        - 1.72834e-13 * pressure ** 2 * temperature
        - 3.58922e-7 * pressure
        - 2.25341e-10 * pressure ** 2
    )

    # --------------------------------------------------------
    # Water FVF
    # --------------------------------------------------------

    bw = (
        (1.0 + delta_vwt)
        * (1.0 + delta_vwp)
    )

    if np.any(bw <= 0):
        raise ValueError(
            "Calculated Bw contains non-positive values. "
            "Check pressure and temperature."
        )

    return bw


# ============================================================
# McCain Water Viscosity at Atmospheric Pressure
# ============================================================

def mccain_water_viscosity_atm(
    temperature,
    salinity
):
    """
    Calculate water viscosity at atmospheric pressure
    using McCain's correlation.

    The correlation is:

        μw1 = A * T^B

    where A and B are functions of salinity.

    Parameters
    ----------
    temperature : float
        Reservoir temperature, °F.

    salinity : float
        Water salinity, weight percent.

    Returns
    -------
    float
        Water viscosity at atmospheric pressure, cP.
    """

    temperature = _validate_temperature(
        temperature
    )

    salinity = _validate_salinity(
        salinity
    )

    # Temperature must be positive for T^B
    if np.any(temperature <= 0):
        raise ValueError(
            "Temperature must be greater than 0 °F "
            "for this correlation."
        )

    # --------------------------------------------------------
    # McCain coefficient A
    # --------------------------------------------------------

    A = (
        109.574
        - 8.40564 * salinity
        + 0.313314 * salinity ** 2
        + 8.72213e-3 * salinity ** 3
    )

    # --------------------------------------------------------
    # McCain coefficient B
    # --------------------------------------------------------

    B = (
        -1.12166
        + 2.63951e-2 * salinity
        - 6.79461e-4 * salinity ** 2
        - 5.47119e-5 * salinity ** 3
        + 1.55586e-6 * salinity ** 4
    )

    # --------------------------------------------------------
    # Atmospheric-pressure viscosity
    # --------------------------------------------------------

    mu_w1 = A * temperature ** B

    if np.any(mu_w1 <= 0):
        raise ValueError(
            "Calculated water viscosity is non-positive."
        )

    return mu_w1


# ============================================================
# McCain Water Viscosity at Reservoir Pressure
# ============================================================

def mccain_water_viscosity(
    pressure,
    temperature,
    salinity
):
    """
    Calculate water viscosity at reservoir pressure
    using McCain's correlation.

    Equation:

        μw = μw1 *
             (
                 0.9994
                 + 4.0295e-5*P
                 + 3.1062e-9*P²
             )

    Parameters
    ----------
    pressure : float or array-like
        Reservoir pressure, psia.

    temperature : float
        Reservoir temperature, °F.

    salinity : float
        Water salinity, weight percent.

    Returns
    -------
    float or numpy.ndarray
        Water viscosity at reservoir pressure, cP.
    """

    pressure = _validate_pressure(
        pressure
    )

    mu_w1 = mccain_water_viscosity_atm(
        temperature,
        salinity
    )

    pressure_correction = (
        0.9994
        + 4.0295e-5 * pressure
        + 3.1062e-9 * pressure ** 2
    )

    mu_w = (
        mu_w1
        * pressure_correction
    )

    if np.any(mu_w <= 0):
        raise ValueError(
            "Calculated reservoir water viscosity "
            "is non-positive."
        )

    return mu_w


# ============================================================
# Complete Water PVT Calculation
# ============================================================

def water_pvt(
    pressure,
    temperature,
    salinity
):
    """
    Calculate a complete set of water PVT properties.

    Parameters
    ----------
    pressure : float or array-like
        Reservoir pressure, psia.

    temperature : float
        Reservoir temperature, °F.

    salinity : float
        Water salinity, weight percent.

    Returns
    -------
    dict
        Dictionary containing:

            Bw
            μw_atm
            μw
    """

    bw = mccain_bw(
        pressure=pressure,
        temperature=temperature
    )

    mu_w1 = mccain_water_viscosity_atm(
        temperature=temperature,
        salinity=salinity
    )

    mu_w = mccain_water_viscosity(
        pressure=pressure,
        temperature=temperature,
        salinity=salinity
    )

    return {
        "Bw_rb_per_STB": bw,
        "mu_w_atm_cp": mu_w1,
        "mu_w_cp": mu_w
    }