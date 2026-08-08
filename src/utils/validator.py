"""
PyMBAL - Data Validator

Validates reservoir production data according to
the selected PVT workflow.
"""

import pandas as pd


class DataValidator:

    # ---------------------------------------------------------
    # Columns required for every dataset
    # ---------------------------------------------------------

    BASE_COLUMNS = [
        "Date",
        "Days",
        "Np_STB",
        "Gp_Mscf",
        "Wp_STB",
        "Pressure_psia",
    ]

    # ---------------------------------------------------------
    # Inputs required for correlation-based PVT
    # ---------------------------------------------------------

    PVT_INPUT_COLUMNS = [
        "Temperature_F",
        "Oil_API",
        "Gas_Gravity",
        "Water_Salinity_wt_pct",
    ]

    # ---------------------------------------------------------
    # PVT properties
    # ---------------------------------------------------------

    PVT_COLUMNS = [
        "Bo_rb_per_STB",
        "Bg_rb_per_scf",
        "Bw_rb_per_STB",
        "Rs_scf_per_STB",
    ]

    # =========================================================
    # Base validation
    # =========================================================

    @classmethod
    def validate_base_columns(cls, df):

        missing = [
            column
            for column in cls.BASE_COLUMNS
            if column not in df.columns
        ]

        if missing:

            raise ValueError(
                "\nMissing required reservoir columns:\n"
                f"{missing}"
            )

        print(
            "✓ All required reservoir columns are present."
        )

    # =========================================================
    # Correlation input validation
    # =========================================================

    @classmethod
    def validate_pvt_inputs(cls, df):

        missing = [
            column
            for column in cls.PVT_INPUT_COLUMNS
            if column not in df.columns
        ]

        if missing:

            raise ValueError(
                "\nMissing PVT correlation inputs:\n"
                f"{missing}"
            )

        print(
            "✓ All PVT correlation inputs are present."
        )

    # =========================================================
    # Missing values
    # =========================================================

    @staticmethod
    def validate_missing_values(df):

        missing = df.isnull().sum()

        missing = missing[missing > 0]

        if len(missing) > 0:

            print("\n⚠ Missing Values:")

            print(missing)

        else:

            print(
                "✓ No missing values found."
            )

    # =========================================================
    # Negative values
    # =========================================================

    @staticmethod
    def validate_negative_values(df):

        numerical_columns = [
            "Days",
            "Np_STB",
            "Gp_Mscf",
            "Wp_STB",
            "Pressure_psia",
            "Temperature_F",
            "Oil_API",
            "Gas_Gravity",
            "Water_Salinity_wt_pct",
            "Bo_rb_per_STB",
            "Bg_rb_per_scf",
            "Bw_rb_per_STB",
            "Rs_scf_per_STB",
        ]

        for column in numerical_columns:

            if column not in df.columns:
                continue

            if (df[column] < 0).any():

                raise ValueError(
                    f"Negative values detected in "
                    f"'{column}'."
                )

        print(
            "✓ No negative values detected."
        )

    # =========================================================
    # PVT property validation
    # =========================================================

    @classmethod
    def validate_pvt_properties(cls, df):

        available = [
            column
            for column in cls.PVT_COLUMNS
            if column in df.columns
        ]

        if not available:

            print(
                "ℹ No PVT properties supplied. "
                "Correlation calculation may be required."
            )

            return

        print("\nAvailable PVT properties:")

        for column in available:

            missing_count = df[column].isna().sum()

            if missing_count > 0:

                print(
                    f"  ⚠ {column}: "
                    f"{missing_count} missing values"
                )

            else:

                print(
                    f"  ✓ {column}"
                )