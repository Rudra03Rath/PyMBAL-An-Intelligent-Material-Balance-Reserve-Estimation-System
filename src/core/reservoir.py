"""
PyMBAL - Reservoir Model

Handles reservoir-level static properties and
initial reservoir conditions required by the
material balance engine.
"""

import json
from pathlib import Path
from dataclasses import dataclass

from src.pvt import PVTEngine


@dataclass
class ReservoirState:
    """
    Represents the initial reservoir state.

    All pressure-dependent PVT properties correspond
    to the initial reservoir pressure Pi.
    """

    # --------------------------------------------------------
    # Reservoir properties
    # --------------------------------------------------------

    initial_pressure_psia: float
    initial_temperature_F: float

    initial_water_saturation: float

    formation_compressibility_psi_inv: float
    water_compressibility_psi_inv: float

    gas_cap_ratio: float | None

    # --------------------------------------------------------
    # Fluid properties
    # --------------------------------------------------------

    oil_api: float
    gas_gravity: float
    water_salinity_wt_pct: float

    # --------------------------------------------------------
    # Initial PVT properties
    # --------------------------------------------------------

    Boi_rb_per_STB: float
    Bgi_rb_per_scf: float
    Rsi_scf_per_STB: float
    Bwi_rb_per_STB: float


class ReservoirModel:
    """
    Loads reservoir configuration and builds
    the initial reservoir state.

    Initial PVT is supplied externally by the
    PVTEngine so that the reservoir state uses
    the same PVT workflow as the material-balance
    dataset.
    """

    def __init__(self, config_path):

        self.config_path = Path(config_path)

        if not self.config_path.exists():

            raise FileNotFoundError(
                f"Reservoir configuration file not found:\n"
                f"{self.config_path}"
            )

        self.config = self._load_config()

        self._validate_config()

    # ========================================================
    # Load JSON
    # ========================================================

    def _load_config(self):

        try:

            with open(
                self.config_path,
                "r",
                encoding="utf-8"
            ) as file:

                return json.load(file)

        except json.JSONDecodeError as exc:

            raise ValueError(
                "Invalid JSON format in reservoir "
                "configuration file."
            ) from exc

    # ========================================================
    # Validate Configuration
    # ========================================================

    def _validate_config(self):

        if "reservoir" not in self.config:

            raise ValueError(
                "Missing 'reservoir' section "
                "in configuration."
            )

        if "fluid" not in self.config:

            raise ValueError(
                "Missing 'fluid' section "
                "in configuration."
            )

        reservoir = self.config["reservoir"]
        fluid = self.config["fluid"]

        required_reservoir = [
            "initial_pressure_psia",
            "initial_temperature_F",
            "initial_water_saturation",
            "formation_compressibility_psi_inv",
            "water_compressibility_psi_inv",
        ]

        required_fluid = [
            "oil_api",
            "gas_gravity",
            "water_salinity_wt_pct",
        ]

        for key in required_reservoir:

            if key not in reservoir:

                raise ValueError(
                    f"Missing reservoir parameter: {key}"
                )

        for key in required_fluid:

            if key not in fluid:

                raise ValueError(
                    f"Missing fluid parameter: {key}"
                )

        # ----------------------------------------------------
        # Physical validation
        # ----------------------------------------------------

        Pi = reservoir[
            "initial_pressure_psia"
        ]

        T = reservoir[
            "initial_temperature_F"
        ]

        Swi = reservoir[
            "initial_water_saturation"
        ]

        cf = reservoir[
            "formation_compressibility_psi_inv"
        ]

        cw = reservoir[
            "water_compressibility_psi_inv"
        ]

        m = reservoir.get(
            "gas_cap_ratio",
            None
        )

        api = fluid[
            "oil_api"
        ]

        gas_gravity = fluid[
            "gas_gravity"
        ]

        salinity = fluid[
            "water_salinity_wt_pct"
        ]

        if Pi <= 0:

            raise ValueError(
                "Initial pressure must be greater than zero."
            )

        if T <= -459.67:

            raise ValueError(
                "Temperature must be above absolute zero."
            )

        if not 0 < Swi < 1:

            raise ValueError(
                "Initial water saturation must be "
                "between 0 and 1."
            )

        if cf < 0:

            raise ValueError(
                "Formation compressibility cannot be negative."
            )

        if cw < 0:

            raise ValueError(
                "Water compressibility cannot be negative."
            )

        if m is not None and m < 0:

            raise ValueError(
                "Gas-cap ratio cannot be negative."
            )

        if api <= 0:

            raise ValueError(
                "Oil API gravity must be positive."
            )

        if gas_gravity <= 0:

            raise ValueError(
                "Gas gravity must be positive."
            )

        if salinity < 0:

            raise ValueError(
                "Water salinity cannot be negative."
            )

    # ========================================================
    # Legacy / Standalone Correlation PVT
    # ========================================================

    def calculate_initial_pvt(self):

        """
        Calculate initial PVT directly from correlations.

        This method is retained for standalone reservoir
        testing and backward compatibility.

        The main integrated application should obtain
        initial PVT from PVTEngine.get_initial_pvt().
        """

        reservoir = self.config[
            "reservoir"
        ]

        fluid = self.config[
            "fluid"
        ]

        Pi = reservoir[
            "initial_pressure_psia"
        ]

        temperature = reservoir[
            "initial_temperature_F"
        ]

        api = fluid[
            "oil_api"
        ]

        gas_gravity = fluid[
            "gas_gravity"
        ]

        salinity = fluid[
            "water_salinity_wt_pct"
        ]

        # ----------------------------------------------------
        # Use PVT engine in correlation mode
        # ----------------------------------------------------

        pvt_engine = PVTEngine(
            mode="correlation",
            temperature=temperature,
            api=api,
            gas_gravity=gas_gravity,
            salinity=salinity,
        )

        # ----------------------------------------------------
        # One-row dataframe at Pi
        # ----------------------------------------------------

        import pandas as pd

        initial_df = pd.DataFrame({
            "Pressure_psia": [
                Pi
            ]
        })

        result = pvt_engine.calculate(
            initial_df
        )

        row = result.iloc[0]

        return {

            "Boi_rb_per_STB":
                float(
                    row[
                        "Bo_rb_per_STB"
                    ]
                ),

            "Bgi_rb_per_scf":
                float(
                    row[
                        "Bg_rb_per_scf"
                    ]
                ),

            "Rsi_scf_per_STB":
                float(
                    row[
                        "Rs_scf_per_STB"
                    ]
                ),

            "Bwi_rb_per_STB":
                float(
                    row[
                        "Bw_rb_per_STB"
                    ]
                ),
        }

    # ========================================================
    # Build Reservoir State
    # ========================================================

    def get_initial_state(
        self,
        initial_pvt
    ):
        """
        Build the initial ReservoirState.

        Parameters
        ----------
        initial_pvt : dict
            Initial PVT dictionary returned by PVTEngine.

            Expected keys:

                Bo_rb_per_STB
                Bg_rb_per_scf
                Rs_scf_per_STB
                Bw_rb_per_STB

            The method converts these into the
            initial-state names:

                Boi
                Bgi
                Rsi
                Bwi

        Returns
        -------
        ReservoirState
        """

        if not isinstance(
            initial_pvt,
            dict
        ):

            raise TypeError(
                "initial_pvt must be a dictionary."
            )

        reservoir = self.config[
            "reservoir"
        ]

        fluid = self.config[
            "fluid"
        ]

        # ----------------------------------------------------
        # PVT engine property names
        # ----------------------------------------------------

        required_pvt = [

            "Bo_rb_per_STB",

            "Bg_rb_per_scf",

            "Rs_scf_per_STB",

            "Bw_rb_per_STB",
        ]

        missing = [

            key

            for key in required_pvt

            if key not in initial_pvt
        ]

        if missing:

            raise ValueError(
                "Missing initial PVT properties: "
                f"{missing}"
            )

        # ----------------------------------------------------
        # Extract initial PVT
        #
        # IMPORTANT:
        #
        # PVTEngine uses:
        #
        #   Bo
        #   Bg
        #   Rs
        #   Bw
        #
        # ReservoirState uses:
        #
        #   Boi
        #   Bgi
        #   Rsi
        #   Bwi
        #
        # The mapping happens here.
        # ----------------------------------------------------

        Boi = float(
            initial_pvt[
                "Bo_rb_per_STB"
            ]
        )

        Bgi = float(
            initial_pvt[
                "Bg_rb_per_scf"
            ]
        )

        Rsi = float(
            initial_pvt[
                "Rs_scf_per_STB"
            ]
        )

        Bwi = float(
            initial_pvt[
                "Bw_rb_per_STB"
            ]
        )

        # ----------------------------------------------------
        # Physical validation of initial PVT
        # ----------------------------------------------------

        initial_pvt_values = {
            "Boi_rb_per_STB": Boi,
            "Bgi_rb_per_scf": Bgi,
            "Rsi_scf_per_STB": Rsi,
            "Bwi_rb_per_STB": Bwi,
        }

        for property_name, value in (
            initial_pvt_values.items()
        ):

            if value <= 0:

                raise ValueError(
                    f"Initial PVT property "
                    f"'{property_name}' must be "
                    "greater than zero."
                )

        # ----------------------------------------------------
        # Create reservoir state
        # ----------------------------------------------------

        state = ReservoirState(

            # ------------------------------------------------
            # Reservoir
            # ------------------------------------------------

            initial_pressure_psia=
                reservoir[
                    "initial_pressure_psia"
                ],

            initial_temperature_F=
                reservoir[
                    "initial_temperature_F"
                ],

            initial_water_saturation=
                reservoir[
                    "initial_water_saturation"
                ],

            formation_compressibility_psi_inv=
                reservoir[
                    "formation_compressibility_psi_inv"
                ],

            water_compressibility_psi_inv=
                reservoir[
                    "water_compressibility_psi_inv"
                ],

            gas_cap_ratio=
                reservoir.get(
                    "gas_cap_ratio",
                    None
                ),

            # ------------------------------------------------
            # Fluid
            # ------------------------------------------------

            oil_api=
                fluid[
                    "oil_api"
                ],

            gas_gravity=
                fluid[
                    "gas_gravity"
                ],

            water_salinity_wt_pct=
                fluid[
                    "water_salinity_wt_pct"
                ],

            # ------------------------------------------------
            # Initial PVT
            # ------------------------------------------------

            Boi_rb_per_STB=
                Boi,

            Bgi_rb_per_scf=
                Bgi,

            Rsi_scf_per_STB=
                Rsi,

            Bwi_rb_per_STB=
                Bwi,
        )

        return state

    # ========================================================
    # Display Reservoir State
    # ========================================================

    @staticmethod
    def display_state(
        state
    ):

        print("\n")
        print("=" * 60)
        print("INITIAL RESERVOIR STATE")
        print("=" * 60)

        print("\nReservoir")
        print("-" * 40)

        print(
            f"Initial Pressure       : "
            f"{state.initial_pressure_psia:.2f} psia"
        )

        print(
            f"Temperature            : "
            f"{state.initial_temperature_F:.2f} °F"
        )

        print(
            f"Initial Water Sat.     : "
            f"{state.initial_water_saturation:.4f}"
        )

        print(
            f"Formation Compress.    : "
            f"{state.formation_compressibility_psi_inv:.3e} psi⁻¹"
        )

        print(
            f"Water Compressibility  : "
            f"{state.water_compressibility_psi_inv:.3e} psi⁻¹"
        )

        print(
            f"Gas Cap Ratio (m)      : "
            f"{state.gas_cap_ratio:.4f}"
        )

        print("\nFluid")
        print("-" * 40)

        print(
            f"Oil API                : "
            f"{state.oil_api:.2f}"
        )

        print(
            f"Gas Gravity            : "
            f"{state.gas_gravity:.3f}"
        )

        print(
            f"Water Salinity         : "
            f"{state.water_salinity_wt_pct:.2f} wt%"
        )

        print("\nInitial PVT")
        print("-" * 40)

        print(
            f"Boi                   : "
            f"{state.Boi_rb_per_STB:.5f} rb/STB"
        )

        print(
            f"Bgi                   : "
            f"{state.Bgi_rb_per_scf:.7f} rb/scf"
        )

        print(
            f"Rsi                   : "
            f"{state.Rsi_scf_per_STB:.3f} scf/STB"
        )

        print(
            f"Bwi                   : "
            f"{state.Bwi_rb_per_STB:.5f} rb/STB"
        )