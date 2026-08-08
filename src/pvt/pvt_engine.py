"""
PyMBAL - PVT Engine

Central coordinator for all PVT calculations.

Supported modes
---------------
laboratory:
    Use laboratory PVT data supplied by the user.

correlation:
    Calculate PVT properties using empirical correlations.

hybrid:
    Use laboratory/user-supplied properties where available
    and calculate missing properties using correlations.

The actual correlations are implemented in:
    oil.py
    gas.py
    water.py

Laboratory interpolation is implemented in:
    interpolation.py
"""

import pandas as pd

from .oil import standing_pvt
from .gas import gas_pvt
from .water import water_pvt
from .interpolation import interpolate_pvt_dataframe


class PVTEngine:
    """
    Main PVT calculation engine for PyMBAL.
    """

    # ---------------------------------------------------------
    # Required / supported PVT properties
    # ---------------------------------------------------------

    PVT_PROPERTIES = [
        "Bo_rb_per_STB",
        "Bg_rb_per_scf",
        "Bw_rb_per_STB",
        "Rs_scf_per_STB",
    ]

    # ---------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------

    def __init__(
        self,
        mode="laboratory",
        oil_correlation="standing",
        gas_correlation="hall_yarborough",
        water_correlation="mccain",
        temperature=None,
        api=None,
        gas_gravity=None,
        salinity=None,
        allow_extrapolation=False,
    ):
        """
        Parameters
        ----------
        mode : str
            PVT calculation mode.

        oil_correlation : str
            Oil correlation to use.

        gas_correlation : str
            Gas correlation to use.

        water_correlation : str
            Water correlation to use.

        temperature : float
            Reservoir temperature, °F.

        api : float
            Oil API gravity.

        gas_gravity : float
            Gas specific gravity.

        salinity : float
            Water salinity, wt%.

        allow_extrapolation : bool
            Whether laboratory PVT interpolation may
            extrapolate beyond the available pressure range.
        """

        # -----------------------------------------------------
        # Normalize mode
        # -----------------------------------------------------

        self.mode = mode.lower().strip()

        allowed_modes = [
            "laboratory",
            "correlation",
            "hybrid",
        ]

        if self.mode not in allowed_modes:
            raise ValueError(
                f"Invalid PVT mode: '{mode}'. "
                f"Choose from {allowed_modes}."
            )

        # -----------------------------------------------------
        # Correlation selections
        # -----------------------------------------------------

        self.oil_correlation = (
            oil_correlation.lower().strip()
        )

        self.gas_correlation = (
            gas_correlation.lower().strip()
        )

        self.water_correlation = (
            water_correlation.lower().strip()
        )

        # -----------------------------------------------------
        # Reservoir-fluid inputs
        # -----------------------------------------------------

        self.temperature = temperature
        self.api = api
        self.gas_gravity = gas_gravity
        self.salinity = salinity

        self.allow_extrapolation = (
            allow_extrapolation
        )

    # =========================================================
    # PUBLIC API
    # =========================================================

    def calculate(
        self,
        df,
        laboratory_pvt_df=None,
    ):
        """
        Calculate/process PVT properties.

        Parameters
        ----------
        df : pandas.DataFrame
            Reservoir pressure history.

        laboratory_pvt_df : pandas.DataFrame, optional
            Laboratory PVT dataset.

            Required for laboratory interpolation when
            the PVT data are stored separately from the
            production history.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing PVT properties.
        """

        self._validate_dataframe(df)

        df = df.copy()

        print("\nPVT ENGINE")
        print("=" * 50)
        print(f"Mode: {self.mode.upper()}")

        # -----------------------------------------------------
        # Laboratory mode
        # -----------------------------------------------------

        if self.mode == "laboratory":

            df = self._process_laboratory(
                df,
                laboratory_pvt_df
            )

        # -----------------------------------------------------
        # Correlation mode
        # -----------------------------------------------------

        elif self.mode == "correlation":

            df = self._process_correlations(df)

        # -----------------------------------------------------
        # Hybrid mode
        # -----------------------------------------------------

        elif self.mode == "hybrid":

            df = self._process_hybrid(
                df,
                laboratory_pvt_df
            )

        # -----------------------------------------------------
        # Final validation
        # -----------------------------------------------------

        self._validate_pvt_properties(df)

        print("\n✓ PVT calculation completed.")

        return df

    # =========================================================
    # LABORATORY MODE
    # =========================================================

    def _process_laboratory(
        self,
        df,
        laboratory_pvt_df=None,
    ):
        """
        Process laboratory PVT data.

        Two situations are supported:

        1. PVT properties already exist in df.
        2. Separate laboratory PVT table is supplied.
        """

        print("\nUsing laboratory PVT data.")

        # -----------------------------------------------------
        # Case 1:
        # PVT properties already exist in production dataset
        # -----------------------------------------------------

        if self._has_all_pvt_properties(df):

            print(
                "✓ PVT properties already present "
                "in reservoir dataset."
            )

            return df

        # -----------------------------------------------------
        # Case 2:
        # Separate laboratory PVT table
        # -----------------------------------------------------

        if laboratory_pvt_df is None:

            missing = self._missing_pvt_properties(df)

            raise ValueError(
                "\nLaboratory PVT mode requires either:\n"
                "1. PVT properties in the reservoir dataset, or\n"
                "2. a separate laboratory_pvt_df.\n\n"
                f"Missing properties: {missing}"
            )

        # -----------------------------------------------------
        # Interpolate laboratory PVT
        # -----------------------------------------------------

        if "Pressure_psia" not in df.columns:

            raise ValueError(
                "Reservoir dataset must contain "
                "'Pressure_psia'."
            )

        interpolated = interpolate_pvt_dataframe(
            target_pressure=df["Pressure_psia"].to_numpy(),
            laboratory_df=laboratory_pvt_df,
            pressure_column="Pressure_psia",
            allow_extrapolation=self.allow_extrapolation,
        )

        # -----------------------------------------------------
        # Add interpolated properties
        # -----------------------------------------------------

        for column in self.PVT_PROPERTIES:

            if column in interpolated.columns:

                df[column] = interpolated[column].to_numpy()

        print(
            "✓ Laboratory PVT properties "
            "interpolated successfully."
        )

        return df

    # =========================================================
    # CORRELATION MODE
    # =========================================================

    def _process_correlations(self, df):
        """
        Calculate PVT properties from correlations.
        """

        print("\nUsing empirical PVT correlations.")

        self._validate_correlation_inputs()

        pressure = df["Pressure_psia"].to_numpy()

        # -----------------------------------------------------
        # Oil PVT
        # -----------------------------------------------------

        if self.oil_correlation == "standing":

            oil_result = standing_pvt(
                pressure=pressure,
                temperature=self.temperature,
                api=self.api,
                gas_gravity=self.gas_gravity,
            )

            df["Rs_scf_per_STB"] = (
                oil_result["Rs_scf_per_STB"]
            )

            df["Bo_rb_per_STB"] = (
                oil_result["Bo_rb_per_STB"]
            )

            df["Pb_psia"] = (
                oil_result["Pb_psia"]
            )

        else:

            raise NotImplementedError(
                f"Oil correlation "
                f"'{self.oil_correlation}' "
                "is not implemented yet."
            )

        # -----------------------------------------------------
        # Gas PVT
        # -----------------------------------------------------

        if self.gas_correlation == "hall_yarborough":

            gas_result = gas_pvt(
                pressure=pressure,
                temperature=self.temperature,
                gas_gravity=self.gas_gravity,
            )

            df["Ppc_psia"] = (
                gas_result["Ppc_psia"]
            )

            df["Tpc_R"] = (
                gas_result["Tpc_R"]
            )

            df["Ppr"] = (
                gas_result["Ppr"]
            )

            df["Tpr"] = (
                gas_result["Tpr"]
            )

            df["Z"] = (
                gas_result["Z"]
            )

            df["Bg_rb_per_scf"] = (
                gas_result["Bg_rb_per_scf"]
            )

        else:

            raise NotImplementedError(
                f"Gas correlation "
                f"'{self.gas_correlation}' "
                "is not implemented yet."
            )

        # -----------------------------------------------------
        # Water PVT
        # -----------------------------------------------------

        if self.water_correlation == "mccain":

            if self.salinity is None:

                raise ValueError(
                    "Salinity is required for "
                    "McCain water-property calculations."
                )

            water_result = water_pvt(
                pressure=pressure,
                temperature=self.temperature,
                salinity=self.salinity,
            )

            df["Bw_rb_per_STB"] = (
                water_result["Bw_rb_per_STB"]
            )

            df["mu_w_atm_cp"] = (
                water_result["mu_w_atm_cp"]
            )

            df["mu_w_cp"] = (
                water_result["mu_w_cp"]
            )

        else:

            raise NotImplementedError(
                f"Water correlation "
                f"'{self.water_correlation}' "
                "is not implemented yet."
            )

        print(
            "✓ Oil PVT calculated."
        )

        print(
            "✓ Gas PVT calculated."
        )

        print(
            "✓ Water PVT calculated."
        )

        return df

    # =========================================================
    # HYBRID MODE
    # =========================================================

    def _process_hybrid(
        self,
        df,
        laboratory_pvt_df=None,
    ):
        """
        Hybrid PVT calculation.

        Existing PVT properties are preserved.

        Missing properties are calculated using
        correlations.
        """

        print(
            "\nUsing hybrid PVT mode."
        )

        df = df.copy()

        missing = self._missing_pvt_properties(df)

        # -----------------------------------------------------
        # Nothing missing
        # -----------------------------------------------------

        if not missing:

            print(
                "✓ All PVT properties already available."
            )

            return df

        print("\nMissing properties:")

        for property_name in missing:

            print(
                f"  → {property_name}"
            )

        # -----------------------------------------------------
        # If laboratory table is available, use it first
        # -----------------------------------------------------

        if laboratory_pvt_df is not None:

            interpolated = interpolate_pvt_dataframe(
                target_pressure=df["Pressure_psia"].to_numpy(),
                laboratory_df=laboratory_pvt_df,
                pressure_column="Pressure_psia",
                allow_extrapolation=self.allow_extrapolation,
            )

            for property_name in missing:

                if property_name in interpolated.columns:

                    values = interpolated[
                        property_name
                    ].to_numpy()

                    df[property_name] = values

            print(
                "✓ Available laboratory PVT "
                "properties applied."
            )

        # -----------------------------------------------------
        # Recalculate missing properties
        # -----------------------------------------------------

        remaining = self._missing_pvt_properties(df)

        if remaining:

            print(
                "\nCalculating remaining properties "
                "using correlations..."
            )

            # We calculate correlation outputs into a
            # temporary dataframe, then only use values
            # that are actually missing.

            correlation_df = self._process_correlations(
                df.copy()
            )

            for property_name in remaining:

                if property_name in correlation_df.columns:

                    df[property_name] = (
                        correlation_df[property_name]
                    )

        return df

    # =========================================================
    # VALIDATION
    # =========================================================

    @staticmethod
    def _validate_dataframe(df):

        if not isinstance(
            df,
            pd.DataFrame
        ):
            raise TypeError(
                "PVTEngine requires a pandas DataFrame."
            )

        if df.empty:

            raise ValueError(
                "Cannot perform PVT calculations "
                "on an empty DataFrame."
            )

        if "Pressure_psia" not in df.columns:

            raise ValueError(
                "DataFrame must contain "
                "'Pressure_psia'."
            )

        if df["Pressure_psia"].isna().any():

            raise ValueError(
                "Pressure_psia contains missing values."
            )

        if (
            df["Pressure_psia"] <= 0
        ).any():

            raise ValueError(
                "Pressure_psia contains "
                "non-positive values."
            )

    # ---------------------------------------------------------

    def _validate_correlation_inputs(self):

        if self.temperature is None:

            raise ValueError(
                "Temperature is required when using "
                "PVT correlations."
            )

        if self.gas_gravity is None:

            raise ValueError(
                "Gas gravity is required when using "
                "PVT correlations."
            )

        if self.api is None:

            raise ValueError(
                "Oil API gravity is required when using "
                "oil PVT correlations."
            )

    # ---------------------------------------------------------

    @classmethod
    def _missing_pvt_properties(
        cls,
        df
    ):

        return [
            property_name
            for property_name in cls.PVT_PROPERTIES
            if property_name not in df.columns
        ]

    # ---------------------------------------------------------

    @classmethod
    def _has_all_pvt_properties(
        cls,
        df
    ):

        return len(
            cls._missing_pvt_properties(df)
        ) == 0

    # ---------------------------------------------------------

    @classmethod
    def _validate_pvt_properties(
        cls,
        df
    ):
        """
        Validate calculated/supplied PVT properties.
        """

        for property_name in cls.PVT_PROPERTIES:

            if property_name not in df.columns:

                raise ValueError(
                    f"Required PVT property "
                    f"'{property_name}' is missing."
                )

            if df[property_name].isna().any():

                raise ValueError(
                    f"{property_name} contains "
                    "missing values."
                )

            if (
                df[property_name] <= 0
            ).any():

                raise ValueError(
                    f"{property_name} contains "
                    "non-positive values."
                )

        print(
            "✓ PVT properties passed final validation."
        )