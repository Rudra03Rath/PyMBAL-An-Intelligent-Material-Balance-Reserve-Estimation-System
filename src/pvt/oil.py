"""
PyMBAL - Oil PVT Correlations

This module contains correlations for calculating
oil PVT properties.

Currently implemented:
    - Standing Solution Gas-Oil Ratio (Rs)
    - Standing Bubble Point Pressure (Pb)
    - Standing Oil Formation Volume Factor (Bo)

Field units:
    Pressure       : psia
    Temperature    : °F
    API gravity    : degrees API
    Gas gravity    : specific gravity relative to air
    Rs             : scf/STB
    Bo             : rb/STB
"""

import numpy as np


# ============================================================
# Utility Functions
# ============================================================

def _validate_positive(value, name):
    """
    Check that a parameter contains only positive values.
    """

    if np.any(np.asarray(value) <= 0):
        raise ValueError(f"{name} must be greater than zero.")


def api_to_specific_gravity(api):
    """
    Convert oil API gravity to oil specific gravity.

    Parameters
    ----------
    api : float or array-like
        Oil gravity in degrees API.

    Returns
    -------
    float or numpy.ndarray
        Oil specific gravity.
    """

    api = np.asarray(api, dtype=float)

    if np.any(api <= -131.5):
        raise ValueError(
            "API gravity produces an invalid oil specific gravity."
        )

    return 141.5 / (131.5 + api)


# ============================================================
# Standing Solution Gas-Oil Ratio
# ============================================================

def standing_rs(
    pressure,
    temperature,
    api,
    gas_gravity
):
    """
    Calculate solution gas-oil ratio using Standing's correlation.

    Standing's correlation:

        Rs = γg [
                (P / 18.2 + 1.4)
                * 10^(0.0125*API - 0.00091*T)
             ]^1.2048

    Parameters
    ----------
    pressure : float or array-like
        Pressure, psia.

    temperature : float
        Reservoir temperature, °F.

    api : float
        Stock-tank oil API gravity.

    gas_gravity : float
        Solution gas specific gravity.

    Returns
    -------
    float or numpy.ndarray
        Solution gas-oil ratio, scf/STB.
    """

    _validate_positive(pressure, "Pressure")
    _validate_positive(temperature + 460, "Temperature in °R")
    _validate_positive(gas_gravity, "Gas gravity")

    pressure = np.asarray(pressure, dtype=float)

    exponent = (
        0.0125 * api
        - 0.00091 * temperature
    )

    rs = gas_gravity * (
        (pressure / 18.2 + 1.4)
        * 10 ** exponent
    ) ** 1.2048

    return rs


# ============================================================
# Standing Bubble Point Pressure
# ============================================================

def standing_bubble_point(
    rs,
    temperature,
    api,
    gas_gravity
):
    """
    Calculate bubble point pressure using Standing's correlation.

    Equation:

        Pb = 18.2 [
                (Rs / γg)^0.83
                * 10^(0.00091*T - 0.0125*API)
                - 1.4
             ]

    Parameters
    ----------
    rs : float or array-like
        Solution gas-oil ratio at bubble point, scf/STB.

    temperature : float
        Reservoir temperature, °F.

    api : float
        Stock-tank oil API gravity.

    gas_gravity : float
        Gas specific gravity.

    Returns
    -------
    float or numpy.ndarray
        Bubble point pressure, psia.
    """

    _validate_positive(rs, "Solution GOR")
    _validate_positive(temperature + 460, "Temperature in °R")
    _validate_positive(gas_gravity, "Gas gravity")

    rs = np.asarray(rs, dtype=float)

    exponent = (
        0.00091 * temperature
        - 0.0125 * api
    )

    pb = 18.2 * (
        (rs / gas_gravity) ** 0.83
        * 10 ** exponent
        - 1.4
    )

    if np.any(pb <= 0):
        raise ValueError(
            "Calculated bubble point pressure is not positive. "
            "Check the input PVT data."
        )

    return pb


# ============================================================
# Standing Oil Formation Volume Factor
# ============================================================

def standing_bo(
    rs,
    temperature,
    api,
    gas_gravity
):
    """
    Calculate oil formation volume factor using
    Standing's correlation.

    Equation:

        Bo = 0.9759
             + 0.00012 *
               [
                   Rs * sqrt(γg / γo)
                   + 1.25*T
               ]^1.2

    Parameters
    ----------
    rs : float or array-like
        Solution gas-oil ratio, scf/STB.

    temperature : float
        Reservoir temperature, °F.

    api : float
        Stock-tank oil API gravity.

    gas_gravity : float
        Solution gas specific gravity.

    Returns
    -------
    float or numpy.ndarray
        Oil formation volume factor, rb/STB.
    """

    _validate_positive(rs, "Solution GOR")
    _validate_positive(temperature + 460, "Temperature in °R")
    _validate_positive(gas_gravity, "Gas gravity")

    rs = np.asarray(rs, dtype=float)

    oil_gravity = api_to_specific_gravity(api)

    term = (
        rs * np.sqrt(gas_gravity / oil_gravity)
        + 1.25 * temperature
    )

    bo = 0.9759 + 0.00012 * term ** 1.2

    return bo


# ============================================================
# Standing PVT Package
# ============================================================

def standing_pvt(
    pressure,
    temperature,
    api,
    gas_gravity
):
    """
    Calculate a complete set of Standing oil PVT properties.

    Parameters
    ----------
    pressure : float or array-like
        Pressure, psia.

    temperature : float
        Reservoir temperature, °F.

    api : float
        Oil API gravity.

    gas_gravity : float
        Gas specific gravity.

    Returns
    -------
    dict
        Dictionary containing:
            Rs
            Bo
            Pb
    """

    rs = standing_rs(
        pressure=pressure,
        temperature=temperature,
        api=api,
        gas_gravity=gas_gravity
    )

    pb = standing_bubble_point(
        rs=rs,
        temperature=temperature,
        api=api,
        gas_gravity=gas_gravity
    )

    bo = standing_bo(
        rs=rs,
        temperature=temperature,
        api=api,
        gas_gravity=gas_gravity
    )

    return {
        "Rs_scf_per_STB": rs,
        "Bo_rb_per_STB": bo,
        "Pb_psia": pb
    }