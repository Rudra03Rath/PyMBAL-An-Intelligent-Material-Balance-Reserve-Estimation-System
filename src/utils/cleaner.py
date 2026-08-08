"""
PyMBAL - Data Cleaner

Performs basic preprocessing of reservoir data.
"""

import pandas as pd


class DataCleaner:

    @staticmethod
    def clean(df):

        df = df.copy()

        print("\nCleaning Dataset...")
        print("-" * 40)

        # -----------------------------------------------------
        # Date
        # -----------------------------------------------------

        try:

            df["Date"] = pd.to_datetime(
                df["Date"]
            )

            print(
                "✓ Date column converted."
            )

        except Exception as exc:

            raise ValueError(
                "Unable to convert Date column."
            ) from exc

        # -----------------------------------------------------
        # Sort
        # -----------------------------------------------------

        df = (
            df
            .sort_values("Date")
            .reset_index(drop=True)
        )

        print(
            "✓ Dataset sorted chronologically."
        )

        # -----------------------------------------------------
        # Duplicate rows
        # -----------------------------------------------------

        duplicates = df.duplicated().sum()

        if duplicates > 0:

            df = (
                df
                .drop_duplicates()
                .reset_index(drop=True)
            )

            print(
                f"✓ Removed {duplicates} duplicate rows."
            )

        else:

            print(
                "✓ No duplicate rows found."
            )

        # -----------------------------------------------------
        # Pressure trend
        # -----------------------------------------------------

        if "Pressure_psia" in df.columns:

            print(
                "\nChecking pressure trend..."
            )

            pressure_increases = (
                df["Pressure_psia"]
                .diff()
                > 0
            )

            count = pressure_increases.sum()

            if count > 0:

                print(
                    f"⚠ Pressure increased at "
                    f"{count} record(s)."
                )

                print(
                    "  This is a warning, not an error."
                )

            else:

                print(
                    "✓ Pressure trend looks reasonable."
                )

        # -----------------------------------------------------
        # Cumulative production
        # -----------------------------------------------------

        cumulative_columns = [
            "Np_STB",
            "Gp_Mscf",
            "Wp_STB",
        ]

        for column in cumulative_columns:

            if column not in df.columns:
                continue

            decreases = (
                df[column]
                .diff()
                < 0
            )

            count = decreases.sum()

            if count > 0:

                print(
                    f"⚠ {column} decreases at "
                    f"{count} record(s)."
                )

            else:

                print(
                    f"✓ {column} verified."
                )

        return df