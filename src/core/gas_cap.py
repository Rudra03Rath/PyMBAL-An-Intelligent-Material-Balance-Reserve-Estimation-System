"""
PyMBAL - Gas Cap Expansion Model

Gas-cap expansion calculation used by the material-balance workflow.

Formulation consistent with expansion.py:

    Eg = Boi * (Bg / Bgi - 1)

Gas-cap contribution:

    mEg = m * Eg
"""

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "Bg_rb_per_scf",
]


def _validate_dataframe(df):
    """Validate dataframe and required PVT columns."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required gas-cap PVT columns: "
            f"{missing}"
        )


def _validate_positive(value, name):
    """Validate a positive finite scalar."""

    if not np.isscalar(value):
        raise TypeError(
            f"{name} must be a scalar value."
        )

    if not np.isfinite(value):
        raise ValueError(
            f"{name} must be finite."
        )

    if value <= 0:
        raise ValueError(
            f"{name} must be greater than zero."
        )


def _validate_gas_cap_ratio(m):
    """Validate initial gas-cap ratio."""

    if not np.isscalar(m):
        raise TypeError(
            "m must be a scalar value."
        )

    if not np.isfinite(m):
        raise ValueError(
            "Gas-cap ratio m must be finite."
        )

    if m < 0:
        raise ValueError(
            "Gas-cap ratio m cannot be negative."
        )


def _get_bg(df):
    """Extract and validate gas formation volume factor."""

    bg = pd.to_numeric(
        df["Bg_rb_per_scf"],
        errors="coerce",
    )

    values = bg.to_numpy(
        dtype=float
    )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "Bg_rb_per_scf contains "
            "non-finite values."
        )

    if (bg <= 0).any():

        bad_rows = df.index[
            bg <= 0
        ].tolist()

        raise ValueError(
            "Bg_rb_per_scf must be greater "
            "than zero. "
            f"Invalid rows: {bad_rows}"
        )

    return bg


# ============================================================
# Gas-Cap Expansion
# ============================================================

def calculate_gas_cap_expansion(
    df,
    Boi,
    Bgi,
):
    """
    Calculate gas-cap expansion.

        Eg = Boi * (Bg/Bgi - 1)

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain Bg_rb_per_scf.

    Boi : float
        Initial oil FVF, rb/STB.

    Bgi : float
        Initial gas FVF, rb/scf.

    Returns
    -------
    pandas.Series
        Gas-cap expansion, rb/STB.
    """

    _validate_dataframe(df)

    _validate_positive(
        Boi,
        "Boi",
    )

    _validate_positive(
        Bgi,
        "Bgi",
    )

    Bg = _get_bg(df)

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
# Gas-Cap Contribution
# ============================================================

def calculate_gas_cap_contribution(
    df,
    Boi,
    Bgi,
    m,
):
    """
    Calculate gas-cap contribution.

        mEg = m * Eg
    """

    _validate_gas_cap_ratio(m)

    Eg = calculate_gas_cap_expansion(
        df,
        Boi=Boi,
        Bgi=Bgi,
    )

    return (
        m * Eg
    ).rename(
        "mEg_rb_per_STB"
    )


# ============================================================
# Gas-Cap Terms
# ============================================================

def calculate_gas_cap_terms(
    df,
    Boi,
    Bgi,
    m,
):
    """
    Calculate both Eg and mEg.

    Returns the original dataframe plus:

        Eg_rb_per_STB
        mEg_rb_per_STB
    """

    _validate_gas_cap_ratio(m)

    result = df.copy()

    result[
        "Eg_rb_per_STB"
    ] = calculate_gas_cap_expansion(
        df,
        Boi=Boi,
        Bgi=Bgi,
    )

    result[
        "mEg_rb_per_STB"
    ] = (
        m
        *
        result["Eg_rb_per_STB"]
    )

    return result


# ============================================================
# Convenience Model Class
# ============================================================

class GasCapModel:
    """
    Stateful wrapper for gas-cap calculations.

    This class evaluates a supplied m.
    It does NOT estimate m.
    """

    def __init__(
        self,
        Boi,
        Bgi,
        m=0.0,
    ):

        _validate_positive(
            Boi,
            "Boi",
        )

        _validate_positive(
            Bgi,
            "Bgi",
        )

        _validate_gas_cap_ratio(
            m
        )

        self.Boi = float(Boi)
        self.Bgi = float(Bgi)
        self.m = float(m)

    def calculate_expansion(
        self,
        df,
    ):
        """Return Eg."""

        return calculate_gas_cap_expansion(
            df,
            Boi=self.Boi,
            Bgi=self.Bgi,
        )

    def calculate_contribution(
        self,
        df,
    ):
        """Return mEg."""

        return calculate_gas_cap_contribution(
            df,
            Boi=self.Boi,
            Bgi=self.Bgi,
            m=self.m,
        )

    def calculate(
        self,
        df,
    ):
        """Return dataframe containing Eg and mEg."""

        return calculate_gas_cap_terms(
            df,
            Boi=self.Boi,
            Bgi=self.Bgi,
            m=self.m,
        )