"""
PyMBAL - Expansion Terms

Calculates the expansion terms used in the
Havlena-Ode material balance formulation.

General equation:

    F = N(Eo + mEg + Efw) + We

Definitions
-----------

Oil expansion:

    Eo = (Bo - Boi) + (Rsi - Rs)Bg

Gas-cap expansion:

    Eg = Boi * (Bg/Bgi - 1)

Formation and connate-water expansion:

    Efw = (1 + m)Boi *
          [(cf + cw*Swi)/(1-Swi)] * (Pi-P)

Units
-----

Bo, Boi, Bw, Bwi : rb/STB
Bg, Bgi          : rb/scf
Rs, Rsi          : scf/STB
cf, cw           : 1/psi
Pressure         : psi

Expansion terms are expressed on an
initial oil-zone reservoir-volume basis
(rb/STB).
"""

import numpy as np
import pandas as pd


# ============================================================
# Required columns
# ============================================================

PVT_COLUMNS = [
    "Pressure_psia",
    "Bo_rb_per_STB",
    "Bg_rb_per_scf",
    "Rs_scf_per_STB",
]


# ============================================================
# Validation
# ============================================================

def _validate_pvt_columns(df):
    """
    Validate pressure-dependent PVT columns.
    """

    missing = [
        column
        for column in PVT_COLUMNS
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns required for "
            f"expansion calculations: {missing}"
        )


def _validate_initial_state(
    Boi,
    Bgi,
    Rsi,
    Pi,
    Swi,
    cf,
    cw,
    m
):
    """
    Validate initial reservoir parameters.
    """

    if Boi <= 0:

        raise ValueError(
            "Boi must be greater than zero."
        )

    if Bgi <= 0:

        raise ValueError(
            "Bgi must be greater than zero."
        )

    if Rsi < 0:

        raise ValueError(
            "Rsi cannot be negative."
        )

    if Pi <= 0:

        raise ValueError(
            "Initial pressure Pi must be "
            "greater than zero."
        )

    if not 0 < Swi < 1:

        raise ValueError(
            "Swi must be between 0 and 1."
        )

    if cf < 0:

        raise ValueError(
            "Formation compressibility cf "
            "cannot be negative."
        )

    if cw < 0:

        raise ValueError(
            "Water compressibility cw "
            "cannot be negative."
        )

    if m < 0:

        raise ValueError(
            "Gas-cap ratio m cannot be negative."
        )


# ============================================================
# Oil Expansion
# ============================================================

def calculate_oil_expansion(
    df,
    Boi,
    Rsi
):
    """
    Calculate oil expansion term.

        Eo = (Bo - Boi) + (Rsi - Rs)Bg

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:

            Bo_rb_per_STB
            Bg_rb_per_scf
            Rs_scf_per_STB

    Boi : float
        Initial oil FVF, rb/STB.

    Rsi : float
        Initial solution GOR, scf/STB.

    Returns
    -------
    pandas.Series
        Oil expansion, rb/STB.
    """

    _validate_pvt_columns(df)

    if Boi <= 0:

        raise ValueError(
            "Boi must be greater than zero."
        )

    if Rsi < 0:

        raise ValueError(
            "Rsi cannot be negative."
        )

    Bo = df[
        "Bo_rb_per_STB"
    ]

    Bg = df[
        "Bg_rb_per_scf"
    ]

    Rs = df[
        "Rs_scf_per_STB"
    ]

    Eo = (
        (Bo - Boi)
        +
        (Rsi - Rs) * Bg
    )

    return Eo.rename(
        "Eo_rb_per_STB"
    )


# ============================================================
# Gas-Cap Expansion
# ============================================================

def calculate_gas_cap_expansion(
    df,
    Boi,
    Bgi
):
    """
    Calculate gas-cap expansion term.

        Eg = Boi * (Bg/Bgi - 1)

    Parameters
    ----------
    df : pandas.DataFrame

    Boi : float
        Initial oil FVF, rb/STB.

    Bgi : float
        Initial gas FVF, rb/scf.

    Returns
    -------
    pandas.Series
        Gas-cap expansion, rb/STB.
    """

    _validate_pvt_columns(df)

    if Boi <= 0:

        raise ValueError(
            "Boi must be greater than zero."
        )

    if Bgi <= 0:

        raise ValueError(
            "Bgi must be greater than zero."
        )

    Bg = df[
        "Bg_rb_per_scf"
    ]

    Eg = (
        Boi
        *
        (
            Bg / Bgi
            - 1.0
        )
    )

    return Eg.rename(
        "Eg_rb_per_STB"
    )


# ============================================================
# Formation + Connate Water Expansion
# ============================================================

def calculate_water_rock_expansion(
    df,
    Boi,
    Pi,
    Swi,
    cf,
    cw,
    m
):
    """
    Calculate formation and connate-water expansion.

        Efw = (1 + m) Boi
              [(cf + cw*Swi)/(1-Swi)]
              (Pi-P)

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain Pressure_psia.

    Boi : float
        Initial oil FVF, rb/STB.

    Pi : float
        Initial reservoir pressure, psi.

    Swi : float
        Initial water saturation.

    cf : float
        Formation compressibility, 1/psi.

    cw : float
        Water compressibility, 1/psi.

    m : float
        Initial gas-cap ratio.

    Returns
    -------
    pandas.Series
        Formation and connate-water expansion.
    """

    _validate_pvt_columns(df)

    _validate_initial_state(
        Boi=Boi,
        Bgi=1.0,      # Not used here
        Rsi=0.0,      # Not used here
        Pi=Pi,
        Swi=Swi,
        cf=cf,
        cw=cw,
        m=m
    )

    P = df[
        "Pressure_psia"
    ]

    delta_P = (
        Pi - P
    )

    # --------------------------------------------------------
    # Pressure should not exceed initial pressure
    # for a normal depletion calculation.
    # --------------------------------------------------------

    if (delta_P < -1e-8).any():

        bad_rows = df.index[
            delta_P < -1e-8
        ].tolist()

        raise ValueError(
            "Reservoir pressure exceeds initial "
            f"pressure Pi at rows: {bad_rows}."
        )

    expansion = (
        (1.0 + m)
        * Boi
        *
        (
            (cf + cw * Swi)
            /
            (1.0 - Swi)
        )
        *
        delta_P
    )

    return expansion.rename(
        "Efw_rb_per_STB"
    )


# ============================================================
# Total Expansion
# ============================================================

def calculate_total_expansion(
    df,
    Boi,
    Bgi,
    Rsi,
    Pi,
    Swi,
    cf,
    cw,
    m
):
    """
    Calculate all expansion terms.

        E = Eo + mEg + Efw

    Returns
    -------
    pandas.DataFrame
        Original dataframe plus:

            Eo_rb_per_STB
            Eg_rb_per_STB
            Efw_rb_per_STB
            mEg_rb_per_STB
            Etotal_rb_per_STB
    """

    _validate_pvt_columns(df)

    _validate_initial_state(
        Boi=Boi,
        Bgi=Bgi,
        Rsi=Rsi,
        Pi=Pi,
        Swi=Swi,
        cf=cf,
        cw=cw,
        m=m
    )

    result = df.copy()

    # --------------------------------------------------------
    # Individual expansion terms
    # --------------------------------------------------------

    result[
        "Eo_rb_per_STB"
    ] = calculate_oil_expansion(
        df,
        Boi=Boi,
        Rsi=Rsi
    )

    result[
        "Eg_rb_per_STB"
    ] = calculate_gas_cap_expansion(
        df,
        Boi=Boi,
        Bgi=Bgi
    )

    result[
        "Efw_rb_per_STB"
    ] = calculate_water_rock_expansion(
        df,
        Boi=Boi,
        Pi=Pi,
        Swi=Swi,
        cf=cf,
        cw=cw,
        m=m
    )

    # --------------------------------------------------------
    # Gas-cap contribution
    # --------------------------------------------------------

    result[
        "mEg_rb_per_STB"
    ] = (
        m
        *
        result[
            "Eg_rb_per_STB"
        ]
    )

    # --------------------------------------------------------
    # Total expansion
    # --------------------------------------------------------

    result[
        "Etotal_rb_per_STB"
    ] = (
        result[
            "Eo_rb_per_STB"
        ]
        +
        result[
            "mEg_rb_per_STB"
        ]
        +
        result[
            "Efw_rb_per_STB"
        ]
    )

    return result