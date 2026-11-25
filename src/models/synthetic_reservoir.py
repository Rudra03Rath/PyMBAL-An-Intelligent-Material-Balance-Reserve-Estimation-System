"""
PyMBAL - Synthetic Reservoir Generator

Generates controlled synthetic reservoir datasets for validating
material-balance and water-influx calculations.

Supported reservoir cases
-------------------------
1. Closed oil reservoir
2. Oil reservoir with gas cap
3. Oil reservoir with Schilthuis water influx
4. Oil reservoir with Fetkovich water influx
5. Oil reservoir with Van Everdingen-Hurst water influx

The synthetic datasets are constructed from known reservoir
parameters so that the material-balance solution has a known
ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.influx.schilthuis import SchilthuisAquifer
from src.influx.fetkovich import FetkovichAquifer
from src.influx.van_everdingen_hurst import (
    VanEverdingenHurstAquifer,
)


# =====================================================================
# RESERVOIR CONFIGURATION
# =====================================================================

@dataclass
class SyntheticReservoirConfig:

    # ---------------------------------------------------------------
    # Reservoir
    # ---------------------------------------------------------------

    true_ooip_stb: float = 10_000_000.0

    initial_pressure_psia: float = 4000.0

    temperature_f: float = 200.0

    initial_water_saturation: float = 0.25

    formation_compressibility_psi_inv: float = 6.0e-6

    water_compressibility_psi_inv: float = 3.0e-6

    gas_cap_ratio: float = 0.0

    # ---------------------------------------------------------------
    # Fluid
    # ---------------------------------------------------------------

    oil_api: float = 35.0

    gas_gravity: float = 0.80

    water_salinity_wt_percent: float = 5.0

    # ---------------------------------------------------------------
    # Initial PVT
    # ---------------------------------------------------------------

    boi_rb_per_stb: float = 1.42

    bgi_rb_per_scf: float = 0.00082

    rsi_scf_per_stb: float = 780.0

    bwi_rb_per_stb: float = 1.02

    # ---------------------------------------------------------------
    # Simulation
    # ---------------------------------------------------------------

    number_of_points: int = 30

    final_pressure_psia: float = 2500.0

    start_date: str = "2020-01-01"

    # ---------------------------------------------------------------
    # PVT pressure trends
    # ---------------------------------------------------------------

    bo_pressure_slope: float = 0.00005

    bg_pressure_slope: float = 5.0e-8

    rs_pressure_slope: float = 0.30

    bw_pressure_slope: float = 0.00001

    # ---------------------------------------------------------------
    # Schilthuis
    # ---------------------------------------------------------------

    schilthuis_aquifer_pressure_psia: float = 4000.0

    schilthuis_influx_constant_rb_per_day_psi: float = 2.0

    # ---------------------------------------------------------------
    # Fetkovich
    # ---------------------------------------------------------------

    fetkovich_initial_aquifer_pressure_psia: float = 4000.0

    fetkovich_initial_water_volume_rb: float = 5_000_000.0

    fetkovich_total_compressibility_psi_inv: float = 2.0e-5

    fetkovich_productivity_index_rb_per_day_psi: float = 100.0

    # ---------------------------------------------------------------
    # Van Everdingen-Hurst
    # ---------------------------------------------------------------

    veh_aquifer_permeability_md: float = 100.0

    veh_aquifer_porosity: float = 0.20

    veh_aquifer_thickness_ft: float = 100.0

    veh_total_compressibility_psi_inv: float = 2.0e-5

    veh_water_viscosity_cp: float = 0.50

    veh_reservoir_radius_ft: float = 1000.0

    veh_encroachment_angle_deg: float = 360.0


# =====================================================================
# SYNTHETIC RESERVOIR
# =====================================================================

class SyntheticReservoir:

    """
    Controlled synthetic reservoir generator.

    The generator first constructs the reservoir expansion term E
    from known PVT properties and reservoir parameters.

    Production is then generated so that:

        F = N_true * E + We

    Therefore, when the generated dataset is passed through the
    material-balance engine with the correct aquifer model, the
    known OOIP should be recovered.
    """

    def __init__(
        self,
        config: SyntheticReservoirConfig | None = None,
    ):

        self.config = (
            config
            if config is not None
            else SyntheticReservoirConfig()
        )

        self._validate_config()

    # =================================================================
    # VALIDATION
    # =================================================================

    def _validate_config(self):

        c = self.config

        if c.true_ooip_stb <= 0:
            raise ValueError(
                "true_ooip_stb must be positive."
            )

        if c.initial_pressure_psia <= 0:
            raise ValueError(
                "initial_pressure_psia must be positive."
            )

        if c.final_pressure_psia <= 0:
            raise ValueError(
                "final_pressure_psia must be positive."
            )

        if (
            c.final_pressure_psia
            > c.initial_pressure_psia
        ):
            raise ValueError(
                "final_pressure_psia must be "
                "less than or equal to initial_pressure_psia."
            )

        if c.number_of_points < 2:
            raise ValueError(
                "number_of_points must be at least 2."
            )

        if not 0 <= c.gas_cap_ratio:
            raise ValueError(
                "gas_cap_ratio cannot be negative."
            )

        if c.boi_rb_per_stb <= 0:
            raise ValueError(
                "boi_rb_per_stb must be positive."
            )

        if c.bgi_rb_per_scf <= 0:
            raise ValueError(
                "bgi_rb_per_scf must be positive."
            )

        if c.rsi_scf_per_stb < 0:
            raise ValueError(
                "rsi_scf_per_stb cannot be negative."
            )

        if c.bwi_rb_per_stb <= 0:
            raise ValueError(
                "bwi_rb_per_stb must be positive."
            )

    # =================================================================
    # PRESSURE HISTORY
    # =================================================================

    def _generate_pressure_history(self):

        c = self.config

        pressure = np.linspace(
            c.initial_pressure_psia,
            c.final_pressure_psia,
            c.number_of_points,
        )

        dates = pd.date_range(
            start=c.start_date,
            periods=c.number_of_points,
            freq="30D",
        )

        # -------------------------------------------------------------
        # Elapsed time in days
        #
        # This is now explicitly part of the synthetic dataset so
        # that the validator, app.py, and all aquifer models use the
        # same time history.
        # -------------------------------------------------------------

        days = (
            dates - dates[0]
        ).days.astype(float)

        return dates, days, pressure

    # =================================================================
    # PVT
    # =================================================================

    def _generate_pvt(self, pressure):

        c = self.config

        delta_p = (
            c.initial_pressure_psia
            - pressure
        )

        # -------------------------------------------------------------
        # Oil formation volume factor
        # -------------------------------------------------------------

        bo = (
            c.boi_rb_per_stb
            - c.bo_pressure_slope * delta_p
        )

        # -------------------------------------------------------------
        # Gas formation volume factor
        # -------------------------------------------------------------

        bg = (
            c.bgi_rb_per_scf
            + c.bg_pressure_slope * delta_p
        )

        # -------------------------------------------------------------
        # Solution gas-oil ratio
        # -------------------------------------------------------------

        rs = (
            c.rsi_scf_per_stb
            - c.rs_pressure_slope * delta_p
        )

        # -------------------------------------------------------------
        # Water formation volume factor
        # -------------------------------------------------------------

        bw = (
            c.bwi_rb_per_stb
            - c.bw_pressure_slope * delta_p
        )

        # -------------------------------------------------------------
        # Physical lower bounds
        # -------------------------------------------------------------

        bo = np.maximum(
            bo,
            0.01,
        )

        bg = np.maximum(
            bg,
            1.0e-8,
        )

        rs = np.maximum(
            rs,
            0.0,
        )

        bw = np.maximum(
            bw,
            0.01,
        )

        return bo, bg, rs, bw

    # =================================================================
    # EXPANSION TERMS
    # =================================================================

    def _calculate_expansion_terms(
        self,
        bo,
        bg,
        rs,
        bw,
    ):

        c = self.config

        boi = c.boi_rb_per_stb
        bgi = c.bgi_rb_per_scf
        rsi = c.rsi_scf_per_stb
        bwi = c.bwi_rb_per_stb

        # -------------------------------------------------------------
        # Oil expansion
        #
        # Eo = (Bo - Boi) + (Rsi - Rs)Bg
        # -------------------------------------------------------------

        eo = (
            bo - boi
            + (rsi - rs) * bg
        )

        # -------------------------------------------------------------
        # Gas-cap expansion
        #
        # Eg = Bg - Bgi
        #
        # mEg is added to total expansion.
        # -------------------------------------------------------------

        eg = (
            boi
            * (
                bg / bgi - 1.0
            )
        )

        meg = (
            c.gas_cap_ratio * eg
        )

        # -------------------------------------------------------------
        # Connate water + rock expansion
        #
        # Efw = Boi * Swi * (cw + cf) * Δp
        # -------------------------------------------------------------

        delta_p = (
            c.initial_pressure_psia
            - np.asarray(
                self.pressure_history
            )
        )

        # -------------------------------------------------------------
        # Base formation + connate-water expansion
        #
        # This quantity is independent of m.
        # The coupled calibration engine receives this history.
        # -------------------------------------------------------------

        efw_base = (
            boi
            * (
                (
                    c.formation_compressibility_psi_inv
                    + (
                        c.water_compressibility_psi_inv
                        * c.initial_water_saturation
                    )
                )
                / (
                    1.0
                    - c.initial_water_saturation
                )
            )
            * delta_p
        )

        # Full Efw contribution for the actual synthetic reservoir.
        #
        # Efw_total = (1 + m) * Efw_base
        # -------------------------------------------------------------

        efw = (
            (1.0 + c.gas_cap_ratio)
            * efw_base
        )

        etotal = (
            eo
            + meg
            + efw
        )

        return eo, eg, meg, efw_base, efw, etotal

    # =================================================================
    # SCHILTHUIS WATER INFLUX
    # =================================================================

    def _generate_schilthuis_influx(
        self,
        pressure,
        days,
    ):

        c = self.config

        aquifer = SchilthuisAquifer(
            aquifer_pressure_psia=(
                c.schilthuis_aquifer_pressure_psia
            ),
            influx_constant_rb_per_day_psi=(
                c.schilthuis_influx_constant_rb_per_day_psi
            ),
        )

        result = aquifer.calculate_influx(
            pressure_psia=pressure,
            time_days=days,
        )

        return result

    # =================================================================
    # FETKOVICH WATER INFLUX
    # =================================================================

    def _generate_fetkovich_influx(
        self,
        pressure,
        days,
    ):

        c = self.config

        aquifer = FetkovichAquifer(
            initial_aquifer_pressure_psia=(
                c.fetkovich_initial_aquifer_pressure_psia
            ),
            initial_water_volume_rb=(
                c.fetkovich_initial_water_volume_rb
            ),
            total_compressibility_psi_inv=(
                c.fetkovich_total_compressibility_psi_inv
            ),
            productivity_index_rb_per_day_psi=(
                c.fetkovich_productivity_index_rb_per_day_psi
            ),
        )

        result = aquifer.calculate_influx(
            pressure_psia=pressure,
            time_days=days,
        )

        return result

    # =================================================================
    # VAN EVERDINGEN-HURST WATER INFLUX
    # =================================================================

    def _generate_van_everdingen_hurst_influx(
        self,
        pressure,
        days,
    ):

        c = self.config

        aquifer = VanEverdingenHurstAquifer(

            aquifer_permeability_md=(
                c.veh_aquifer_permeability_md
            ),

            aquifer_porosity=(
                c.veh_aquifer_porosity
            ),

            aquifer_thickness_ft=(
                c.veh_aquifer_thickness_ft
            ),

            total_compressibility_psi_inv=(
                c.veh_total_compressibility_psi_inv
            ),

            water_viscosity_cp=(
                c.veh_water_viscosity_cp
            ),

            reservoir_radius_ft=(
                c.veh_reservoir_radius_ft
            ),

            encroachment_angle_deg=(
                c.veh_encroachment_angle_deg
            ),
        )

        result = aquifer.calculate_influx(
            pressure_psia=pressure,
            time_days=days,
        )

        return result

    # =================================================================
    # GENERATE BASE RESERVOIR
    # =================================================================

    def generate(
        self,
        water_influx_model: str = "none",
    ):

        """
        Generate a synthetic reservoir dataset.

        Parameters
        ----------
        water_influx_model :
            "none"
            "schilthuis"
            "fetkovich"
            "van_everdingen_hurst"

        Returns
        -------
        pandas.DataFrame
        """

        model_name = (
            water_influx_model
            .strip()
            .lower()
        )

        aliases = {

            "closed":
                "none",

            "closed_oil":
                "none",

            "schilthuis":
                "schilthuis",

            "fetkovich":
                "fetkovich",

            "van_everdingen_hurst":
                "van_everdingen_hurst",

            "van-everdingen-hurst":
                "van_everdingen_hurst",

            "veh":
                "van_everdingen_hurst",
        }

        model_name = aliases.get(
            model_name,
            model_name,
        )

        valid_models = {

            "none",
            "schilthuis",
            "fetkovich",
            "van_everdingen_hurst",
        }

        if model_name not in valid_models:

            raise ValueError(
                "Unsupported water influx model: "
                f"{water_influx_model}"
            )

        # -------------------------------------------------------------
        # Pressure, dates and elapsed time
        # -------------------------------------------------------------

        (
            dates,
            days,
            pressure,
        ) = self._generate_pressure_history()

        self.pressure_history = pressure

        # -------------------------------------------------------------
        # PVT
        # -------------------------------------------------------------

        bo, bg, rs, bw = (
            self._generate_pvt(
                pressure
            )
        )

        # -------------------------------------------------------------
        # Expansion
        # -------------------------------------------------------------

        (
            eo,
            eg,
            meg,
            efw_base,
            efw,
            etotal,
        ) = self._calculate_expansion_terms(
            bo,
            bg,
            rs,
            bw,
        )

        # -------------------------------------------------------------
        # Water influx
        # -------------------------------------------------------------

        if model_name == "none":

            we = np.zeros(
                len(pressure)
            )

            influx_rate = np.zeros(
                len(pressure)
            )

            aquifer_pressure = np.full(
                len(pressure),
                np.nan,
            )

            delta_we = np.zeros(
                len(pressure)
            )

        elif model_name == "schilthuis":

            influx_result = (
                self._generate_schilthuis_influx(
                    pressure,
                    days,
                )
            )

            we = influx_result[
                "We_rb"
            ].to_numpy()

            influx_rate = (
                influx_result[
                    "influx_rate_rb_per_day"
                ].to_numpy()
            )

            aquifer_pressure = np.full(
                len(pressure),
                self.config
                .schilthuis_aquifer_pressure_psia,
            )

            delta_we = (
                influx_result[
                    "delta_We_rb"
                ].to_numpy()
            )

        elif model_name == "fetkovich":

            influx_result = (
                self._generate_fetkovich_influx(
                    pressure,
                    days,
                )
            )

            we = influx_result[
                "We_rb"
            ].to_numpy()

            influx_rate = (
                influx_result[
                    "influx_rate_rb_per_day"
                ].to_numpy()
            )

            aquifer_pressure = (
                influx_result[
                    "aquifer_pressure_psia"
                ].to_numpy()
            )

            delta_we = (
                influx_result[
                    "delta_We_rb"
                ].to_numpy()
            )

        else:

            influx_result = (
                self._generate_van_everdingen_hurst_influx(
                    pressure,
                    days,
                )
            )

            we = influx_result[
                "We_rb"
            ].to_numpy()

            influx_rate = (
                influx_result[
                    "influx_rate_rb_per_day"
                ].to_numpy()
            )

            delta_we = (
                influx_result[
                    "delta_We_rb"
                ].to_numpy()
            )

            # V-E-H is an infinite-acting aquifer
            # and therefore does not have a finite
            # aquifer-pressure state variable.

            aquifer_pressure = np.full(
                len(pressure),
                np.nan,
            )

        # -------------------------------------------------------------
        # Material balance
        #
        # F = N E + We
        # -------------------------------------------------------------

        f = (
            self.config.true_ooip_stb
            * etotal
            + we
        )

        # -------------------------------------------------------------
        # Production history
        # -------------------------------------------------------------

        producing_gor = np.full(
            len(pressure),
            self.config.rsi_scf_per_stb,
            dtype=float,
        )

        producing_gor[0] = 0.0

        withdrawal_factor = (
            bo
            + (
                producing_gor
                - rs
            ) * bg
        )

        withdrawal_factor[0] = 1.0

        np_stb = np.zeros(
            len(pressure)
        )

        np_stb[1:] = (
            f[1:]
            / withdrawal_factor[1:]
        )

        gp_mscf = (
            np_stb
            * producing_gor
            / 1000.0
        )

        wp_stb = np.zeros(
            len(pressure)
        )

        # -------------------------------------------------------------
        # Reservoir pressure difference
        # -------------------------------------------------------------

        pressure_difference = (
            self.config.initial_pressure_psia
            - pressure
        )

        # -------------------------------------------------------------
        # Reservoir type
        # -------------------------------------------------------------

        if model_name == "none":

            if self.config.gas_cap_ratio > 0:

                reservoir_type = (
                    "Oil reservoir with gas cap"
                )

            else:

                reservoir_type = (
                    "Closed oil reservoir"
                )

        else:

            if self.config.gas_cap_ratio > 0:

                reservoir_type = (
                    "Oil reservoir with gas cap "
                    "and water influx"
                )

            else:

                reservoir_type = (
                    "Oil reservoir with water influx"
                )

        # -------------------------------------------------------------
        # Dataframe
        # -------------------------------------------------------------

        data = pd.DataFrame({

            "Date":
                dates,

            # ---------------------------------------------------------
            # IMPORTANT:
            # Common elapsed-time column required by the validator
            # and app.py.
            # ---------------------------------------------------------

            "Days":
                days,

            "Pressure_psia":
                pressure,

            "Np_STB":
                np_stb,

            "Gp_Mscf":
                gp_mscf,

            "Wp_STB":
                wp_stb,

            "Bo_rb_per_STB":
                bo,

            "Bg_rb_per_scf":
                bg,

            "Rs_scf_per_STB":
                rs,

            "Bw_rb_per_STB":
                bw,

            "Eo_rb_per_STB":
                eo,

            "Eg_rb_per_STB":
                eg,

            "mEg_rb_per_STB":
                meg,

            "Efw_base_rb_per_STB":
                efw_base,

            "Efw_rb_per_STB":
                efw,

            "Etotal_rb_per_STB":
                etotal,

            "F_rb":
                f,

            "We_rb":
                we,

            "delta_We_rb":
                delta_we,

            "influx_rate_rb_per_day":
                influx_rate,

            "Aquifer_Pressure_psia":
                aquifer_pressure,

            "Pressure_Difference_psia":
                pressure_difference,

            "Rp_scf_per_STB":
                producing_gor,

            "Reservoir_Type":
                reservoir_type,

            "Water_Influx_Model":
                model_name,

            # Ground truth for synthetic-data validation only.
            # Calibration code must never consume these columns.
            "True_OOIP_STB":
                self.config.true_ooip_stb,

            "True_Gas_Cap_Ratio_m":
                self.config.gas_cap_ratio,

            "True_Aquifer_Model":
                model_name,
        })

        # -------------------------------------------------------------
        # Final synthetic-dataset schema validation
        #
        # These columns form the common interface consumed by
        # DataValidator, the material-balance engine, and app.py.
        # -------------------------------------------------------------

        required_columns = [
            "Date",
            "Days",
            "Pressure_psia",
            "Np_STB",
            "Gp_Mscf",
            "Wp_STB",
            "Bo_rb_per_STB",
            "Bg_rb_per_scf",
            "Rs_scf_per_STB",
            "Bw_rb_per_STB",
            "Eo_rb_per_STB",
            "Eg_rb_per_STB",
            "mEg_rb_per_STB",
            "Efw_base_rb_per_STB",
            "Efw_rb_per_STB",
            "Etotal_rb_per_STB",
            "F_rb",
            "We_rb",
            "delta_We_rb",
            "influx_rate_rb_per_day",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:
            raise ValueError(
                "Synthetic dataset generation failed. "
                f"Missing required columns: {missing_columns}"
            )

        # Days must start at zero and be non-decreasing.
        days_values = data["Days"].to_numpy(dtype=float)

        if not np.isclose(days_values[0], 0.0):
            raise ValueError(
                "Synthetic dataset Days must start at 0."
            )

        if np.any(np.diff(days_values) < 0):
            raise ValueError(
                "Synthetic dataset Days must be monotonically increasing."
            )

        return data

    # =================================================================
    # CONVENIENCE METHODS
    # =================================================================

    def generate_closed_oil(
        self,
    ):

        return self.generate(
            water_influx_model="none"
        )

    def generate_gas_cap(
        self,
    ):

        return self.generate(
            water_influx_model="none"
        )

    def generate_schilthuis(
        self,
    ):

        return self.generate(
            water_influx_model="schilthuis"
        )

    def generate_fetkovich(
        self,
    ):

        return self.generate(
            water_influx_model="fetkovich"
        )

    def generate_van_everdingen_hurst(
        self,
    ):

        return self.generate(
            water_influx_model=
                "van_everdingen_hurst"
        )

    # =================================================================
    # SAVE
    # =================================================================

    def save_dataset(
        self,
        dataframe: pd.DataFrame,
        filepath: str | Path,
    ):

        filepath = Path(
            filepath
        )

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe.to_csv(
            filepath,
            index=False,
        )

        return filepath


# =====================================================================
# DEFAULT GENERATION EXAMPLE
# =====================================================================

if __name__ == "__main__":

    generator = SyntheticReservoir()

    df = (
        generator
        .generate_van_everdingen_hurst()
    )

    output_path = Path(
        "data/sample_data/"
        "synthetic_van_everdingen_hurst/"
        "synthetic_van_everdingen_hurst_reservoir.csv"
    )

    generator.save_dataset(
        df,
        output_path,
    )

    print(
        "=" * 70
    )

    print(
        "PyMBAL - VAN EVERDINGEN-HURST "
        "SYNTHETIC RESERVOIR"
    )

    print(
        "=" * 70
    )

    print(
        f"True OOIP              : "
        f"{generator.config.true_ooip_stb:,.0f} STB"
    )

    print(
        f"True OOIP              : "
        f"{generator.config.true_ooip_stb / 1e6:.4f} MMSTB"
    )

    print(
        f"Initial Pressure       : "
        f"{generator.config.initial_pressure_psia:.2f} psia"
    )

    print(
        f"Final Pressure         : "
        f"{generator.config.final_pressure_psia:.2f} psia"
    )

    print(
        f"Aquifer permeability   : "
        f"{generator.config.veh_aquifer_permeability_md:.2f} md"
    )

    print(
        f"Aquifer porosity       : "
        f"{generator.config.veh_aquifer_porosity:.4f}"
    )

    print(
        f"Aquifer thickness      : "
        f"{generator.config.veh_aquifer_thickness_ft:.2f} ft"
    )

    print(
        f"Aquifer ct             : "
        f"{generator.config.veh_total_compressibility_psi_inv:.6e} psi⁻¹"
    )

    print(
        f"Water viscosity        : "
        f"{generator.config.veh_water_viscosity_cp:.4f} cp"
    )

    print(
        f"Reservoir radius       : "
        f"{generator.config.veh_reservoir_radius_ft:.2f} ft"
    )

    print(
        f"Encroachment angle     : "
        f"{generator.config.veh_encroachment_angle_deg:.2f}°"
    )

    print(
        f"\n✓ Generated "
        f"{len(df)} data points."
    )

    print(
        f"\nMaximum cumulative We : "
        f"{df['We_rb'].max():,.6f} rb"
    )

    print(
        "\nFirst 10 rows:"
    )

    print(
        df[
            [
                "Date",
                "Days",
                "Pressure_psia",
                "Np_STB",
                "Gp_Mscf",
                "Wp_STB",
                "Bo_rb_per_STB",
                "Bg_rb_per_scf",
                "Rs_scf_per_STB",
                "Bw_rb_per_STB",
                "Eo_rb_per_STB",
                "Eg_rb_per_STB",
                "mEg_rb_per_STB",
                "Efw_rb_per_STB",
                "Etotal_rb_per_STB",
                "We_rb",
                "F_rb",
                "influx_rate_rb_per_day",
            ]
        ].head(10)
    )

    print(
        f"\n✓ Synthetic dataset saved:"
    )

    print(
        output_path
    )