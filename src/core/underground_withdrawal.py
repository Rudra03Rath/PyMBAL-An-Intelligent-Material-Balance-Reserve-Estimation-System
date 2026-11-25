"""
PyMBAL - Underground Withdrawal

Calculates the underground withdrawal term, F, for
black-oil material balance calculations.

For an oil reservoir:

    F = Np [Bo + (Rp - Rs) Bg] + Wp Bw

Equivalent form:

    F = Np Bo + (Gp - Np Rs) Bg + Wp Bw

Field units:
    Np : STB
    Gp : Mscf
    Wp : STB
    Bo : rb/STB
    Bg : rb/scf
    Bw : rb/STB
    Rs : scf/STB

Therefore:

    F : reservoir bbl (rb)
"""

import numpy as np
import pandas as pd


# ============================================================
# Validation
# ============================================================

def _validate_columns(df):
    """
    Validate columns required for underground withdrawal.
    """

    required_columns = [
        "Np_STB",
        "Gp_Mscf",
        "Wp_STB",
        "Bo_rb_per_STB",
        "Bg_rb_per_scf",
        "Bw_rb_per_STB",
        "Rs_scf_per_STB",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns required for "
            f"underground withdrawal: {missing}"
        )


def _validate_values(df):
    """
    Validate numerical values used in F calculation.
    """

    columns = [
        "Np_STB",
        "Gp_Mscf",
        "Wp_STB",
        "Bo_rb_per_STB",
        "Bg_rb_per_scf",
        "Bw_rb_per_STB",
        "Rs_scf_per_STB",
    ]

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    for column in columns:

        if df[column].isna().any():

            raise ValueError(
                f"Column '{column}' contains "
                "missing values."
            )

    # --------------------------------------------------------
    # Production volumes cannot be negative
    # --------------------------------------------------------

    production_columns = [
        "Np_STB",
        "Gp_Mscf",
        "Wp_STB",
    ]

    for column in production_columns:

        if (
            df[column] < 0
        ).any():

            raise ValueError(
                f"Column '{column}' contains "
                "negative production values."
            )

    # --------------------------------------------------------
    # PVT properties must be positive
    # --------------------------------------------------------

    pvt_columns = [
        "Bo_rb_per_STB",
        "Bg_rb_per_scf",
        "Bw_rb_per_STB",
    ]

    for column in pvt_columns:

        if (
            df[column] <= 0
        ).any():

            raise ValueError(
                f"Column '{column}' contains "
                "non-positive PVT values."
            )

    # --------------------------------------------------------
    # Rs cannot be negative
    # --------------------------------------------------------

    if (
        df["Rs_scf_per_STB"] < 0
    ).any():

        raise ValueError(
            "Rs_scf_per_STB cannot be negative."
        )


# ============================================================
# Produced Gas-Oil Ratio
# ============================================================

def calculate_producing_gor(df):
    """
    Calculate cumulative producing gas-oil ratio.

        Rp = Gp / Np

    Since the dataset stores Gp in Mscf:

        Rp = (Gp_Mscf * 1000) / Np_STB

    For the initial reservoir reference row:

        Np = 0
        Gp = 0

    Rp is set to zero because no production has occurred
    yet and Rp is not physically defined for that row.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.Series
        Producing GOR, scf/STB.
    """

    _validate_columns(df)
    _validate_values(df)

    np_oil = (
        df["Np_STB"]
        .to_numpy(
            dtype=float
        )
    )

    gp_scf = (
        df["Gp_Mscf"]
        .to_numpy(
            dtype=float
        )
        *
        1000.0
    )

    rp = np.zeros_like(
        np_oil,
        dtype=float
    )

    # --------------------------------------------------------
    # Only calculate Rp where oil production exists.
    #
    # This prevents:
    #
    #     0 / 0
    #
    # at the initial reservoir state.
    # --------------------------------------------------------

    valid = (
        np_oil > 0
    )

    rp[valid] = (
        gp_scf[valid]
        /
        np_oil[valid]
    )

    return pd.Series(
        rp,
        index=df.index,
        name="Rp_scf_per_STB"
    )


# ============================================================
# Oil Underground Withdrawal
# ============================================================

def calculate_oil_withdrawal(df):
    """
    Calculate underground withdrawal associated
    with produced oil.

        Fo = Np * Bo

    Returns
    -------
    pandas.Series
        Oil withdrawal, rb.
    """

    _validate_columns(df)
    _validate_values(df)

    Fo = (
        df["Np_STB"]
        *
        df["Bo_rb_per_STB"]
    )

    return Fo.rename(
        "Fo_rb"
    )


# ============================================================
# Free-Gas Underground Withdrawal
# ============================================================

def calculate_gas_withdrawal(df):
    """
    Calculate underground withdrawal associated
    with free gas produced from the reservoir.

        Fg = (Gp - Np Rs) Bg

    Where:

        Gp = cumulative produced gas, scf
        Np = cumulative produced oil, STB
        Rs = solution GOR, scf/STB
        Bg = gas FVF, rb/scf

    Returns
    -------
    pandas.Series
        Free-gas withdrawal, rb.
    """

    _validate_columns(df)
    _validate_values(df)

    np_oil = (
        df["Np_STB"]
    )

    gp_scf = (
        df["Gp_Mscf"]
        *
        1000.0
    )

    dissolved_gas = (
        np_oil
        *
        df["Rs_scf_per_STB"]
    )

    free_gas = (
        gp_scf
        -
        dissolved_gas
    )

    # --------------------------------------------------------
    # Physical check
    # --------------------------------------------------------

    if (
        free_gas < -1e-8
    ).any():

        bad_rows = df.index[
            free_gas < -1e-8
        ].tolist()

        raise ValueError(
            "Calculated free gas production is negative "
            f"at rows: {bad_rows}. "
            "Check Gp, Np and Rs."
        )

    # --------------------------------------------------------
    # Numerical protection
    # --------------------------------------------------------

    free_gas = free_gas.clip(
        lower=0
    )

    Fg = (
        free_gas
        *
        df["Bg_rb_per_scf"]
    )

    return Fg.rename(
        "Fg_rb"
    )


# ============================================================
# Water Underground Withdrawal
# ============================================================

def calculate_water_withdrawal(df):
    """
    Calculate underground withdrawal associated
    with produced water.

        Fw = Wp * Bw

    Returns
    -------
    pandas.Series
        Water withdrawal, rb.
    """

    _validate_columns(df)
    _validate_values(df)

    Fw = (
        df["Wp_STB"]
        *
        df["Bw_rb_per_STB"]
    )

    return Fw.rename(
        "Fw_rb"
    )


# ============================================================
# Total Underground Withdrawal
# ============================================================

def calculate_underground_withdrawal(df):
    """
    Calculate total underground withdrawal.

        F = Fo + Fg + Fw

    Equivalent to:

        F = Np [Bo + (Rp - Rs) Bg] + Wp Bw

    Returns
    -------
    pandas.DataFrame
        Original DataFrame with:

            Rp_scf_per_STB
            Fo_rb
            Fg_rb
            Fw_rb
            F_rb
    """

    _validate_columns(df)
    _validate_values(df)

    result = df.copy()

    # --------------------------------------------------------
    # Producing GOR
    # --------------------------------------------------------

    result[
        "Rp_scf_per_STB"
    ] = calculate_producing_gor(
        df
    )

    # --------------------------------------------------------
    # Individual withdrawal terms
    # --------------------------------------------------------

    result[
        "Fo_rb"
    ] = calculate_oil_withdrawal(
        df
    )

    result[
        "Fg_rb"
    ] = calculate_gas_withdrawal(
        df
    )

    result[
        "Fw_rb"
    ] = calculate_water_withdrawal(
        df
    )

    # --------------------------------------------------------
    # Total underground withdrawal
    # --------------------------------------------------------

    result[
        "F_rb"
    ] = (
        result["Fo_rb"]
        +
        result["Fg_rb"]
        +
        result["Fw_rb"]
    )

    return result


# ============================================================
# Direct Formula Verification
# ============================================================

def calculate_underground_withdrawal_direct(df):
    """
    Calculate F directly using:

        F = Np [Bo + (Rp - Rs) Bg] + Wp Bw

    This function is intended for independent
    verification of the decomposed calculation.

    Returns
    -------
    pandas.Series
        Total underground withdrawal, rb.
    """

    _validate_columns(df)
    _validate_values(df)

    rp = calculate_producing_gor(
        df
    )

    F = (

        df["Np_STB"]

        *

        (
            df["Bo_rb_per_STB"]

            +

            (
                rp
                -
                df["Rs_scf_per_STB"]
            )

            *

            df["Bg_rb_per_scf"]
        )

        +

        df["Wp_STB"]

        *

        df["Bw_rb_per_STB"]
    )

    return F.rename(
        "F_direct_rb"
    )


# ============================================================
# Verification Helper
# ============================================================

def verify_underground_withdrawal(
    df,
    tolerance=1e-8,
):
    """
    Verify the decomposed and direct calculations
    of underground withdrawal.

    Returns
    -------
    bool
        True if both calculations agree within tolerance.
    """

    result = (
        calculate_underground_withdrawal(
            df
        )
    )

    direct = (
        calculate_underground_withdrawal_direct(
            df
        )
    )

    difference = (
        result["F_rb"]
        -
        direct
    )

    max_difference = np.max(
        np.abs(
            difference.to_numpy()
        )
    )

    if max_difference > tolerance:

        raise AssertionError(
            "Underground withdrawal verification failed. "
            f"Maximum difference = "
            f"{max_difference:.6e} rb."
        )

    return True