"""
PyMBAL - VAN EVERDINGEN-HURST SYNTHETIC RESERVOIR VALIDATION

Validates a synthetic oil reservoir generated using the
Van Everdingen-Hurst transient aquifer model.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.synthetic_reservoir import (
    SyntheticReservoir,
    SyntheticReservoirConfig,
)


# =====================================================================
# CONFIGURATION
# =====================================================================

TRUE_OOIP = 10_000_000.0

INITIAL_PRESSURE = 4000.0

FINAL_PRESSURE = 2500.0

M = 0.0

N_POINTS = 30

OUTPUT_PATH = Path(
    "data/sample_data/"
    "synthetic_van_everdingen_hurst/"
    "synthetic_van_everdingen_hurst_reservoir.csv"
)


# =====================================================================
# MAIN TEST
# =====================================================================

def main():

    print("=" * 70)
    print(
        "PyMBAL - VAN EVERDINGEN-HURST "
        "SYNTHETIC RESERVOIR VALIDATION"
    )
    print("=" * 70)

    # =================================================================
    # 1. GROUND TRUTH
    # =================================================================

    print("\n")
    print("## GROUND TRUTH")

    print(
        f"True OOIP              : "
        f"{TRUE_OOIP:,.0f} STB"
    )

    print(
        f"True OOIP              : "
        f"{TRUE_OOIP / 1e6:.4f} MMSTB"
    )

    print(
        f"Initial Pressure       : "
        f"{INITIAL_PRESSURE:.2f} psia"
    )

    print(
        f"Final Pressure         : "
        f"{FINAL_PRESSURE:.2f} psia"
    )

    print(
        f"Gas Cap Ratio (m)      : "
        f"{M:.4f}"
    )

    # =================================================================
    # 2. CREATE CONFIGURATION
    # =================================================================

    config = SyntheticReservoirConfig(

        true_ooip_stb=TRUE_OOIP,

        initial_pressure_psia=
            INITIAL_PRESSURE,

        final_pressure_psia=
            FINAL_PRESSURE,

        gas_cap_ratio=M,

        number_of_points=N_POINTS,

        # -------------------------------------------------------------
        # V-E-H parameters
        # -------------------------------------------------------------

        veh_aquifer_permeability_md=100.0,

        veh_aquifer_porosity=0.20,

        veh_aquifer_thickness_ft=100.0,

        veh_total_compressibility_psi_inv=
            2.0e-5,

        veh_water_viscosity_cp=0.50,

        veh_reservoir_radius_ft=1000.0,

        veh_encroachment_angle_deg=360.0,
    )

    generator = SyntheticReservoir(
        config
    )

    print(
        f"\nAquifer permeability  : "
        f"{config.veh_aquifer_permeability_md:.2f} md"
    )

    print(
        f"Aquifer porosity      : "
        f"{config.veh_aquifer_porosity:.4f}"
    )

    print(
        f"Aquifer thickness     : "
        f"{config.veh_aquifer_thickness_ft:.2f} ft"
    )

    print(
        f"Aquifer ct            : "
        f"{config.veh_total_compressibility_psi_inv:.6e} psi⁻¹"
    )

    print(
        f"Water viscosity       : "
        f"{config.veh_water_viscosity_cp:.4f} cp"
    )

    print(
        f"Reservoir radius      : "
        f"{config.veh_reservoir_radius_ft:.2f} ft"
    )

    print(
        f"Encroachment angle    : "
        f"{config.veh_encroachment_angle_deg:.2f}°"
    )

    # =================================================================
    # 3. GENERATE DATASET
    # =================================================================

    print("\n")
    print(
        "## GENERATING SYNTHETIC DATASET"
    )

    df = generator.generate(
        water_influx_model=
            "van_everdingen_hurst"
    )

    assert isinstance(
        df,
        pd.DataFrame
    )

    assert len(df) == N_POINTS

    print(
        f"✓ Generated {len(df)} data points."
    )

    # =================================================================
    # 4. BASIC COLUMN CHECK
    # =================================================================

    required_columns = [

        "Date",

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

        "F_rb",

        "We_rb",

        "delta_We_rb",

        "influx_rate_rb_per_day",

        "Rp_scf_per_STB",

    ]

    missing_columns = [

        column

        for column in required_columns

        if column not in df.columns
    ]

    assert not missing_columns, (
        "Missing columns: "
        f"{missing_columns}"
    )

    print(
        "✓ Required synthetic-reservoir "
        "columns present."
    )

    # =================================================================
    # 5. WATER INFLUX MUST BE ACTIVE
    # =================================================================

    print("\n")
    print(
        "## WATER INFLUX CHECK"
    )

    max_we = df[
        "We_rb"
    ].max()

    print(
        f"Maximum cumulative We : "
        f"{max_we:.6f} rb"
    )

    assert max_we > 0.0

    print(
        "✓ V-E-H water influx is active."
    )

    # =================================================================
    # 6. WATER INFLUX MONOTONICITY
    # =================================================================

    we = df[
        "We_rb"
    ].to_numpy()

    delta_we = df[
        "delta_We_rb"
    ].to_numpy()

    influx_rate = df[
        "influx_rate_rb_per_day"
    ].to_numpy()

    assert np.all(
        np.diff(we) >= -1e-8
    )

    assert np.all(
        delta_we >= -1e-8
    )

    assert np.all(
        influx_rate >= -1e-8
    )

    print(
        f"Initial We : "
        f"{we[0]:.6f} rb"
    )

    print(
        f"Final We   : "
        f"{we[-1]:.6f} rb"
    )

    print(
        "✓ Cumulative water-influx "
        "behavior PASSED."
    )

    # =================================================================
    # 7. PRESSURE HISTORY
    # =================================================================

    pressure = df[
        "Pressure_psia"
    ].to_numpy()

    assert np.isclose(
        pressure[0],
        INITIAL_PRESSURE,
    )

    assert np.isclose(
        pressure[-1],
        FINAL_PRESSURE,
    )

    assert np.all(
        np.diff(pressure) <= 0
    )

    print(
        f"\nInitial pressure : "
        f"{pressure[0]:.6f} psia"
    )

    print(
        f"Final pressure   : "
        f"{pressure[-1]:.6f} psia"
    )

    print(
        "✓ Pressure-history check PASSED."
    )

    # =================================================================
    # 8. F = N × E + We
    # =================================================================

    print("\n")
    print(
        "## MATERIAL BALANCE CONSISTENCY"
    )

    expansion = df[
        "Etotal_rb_per_STB"
    ].to_numpy()

    f = df[
        "F_rb"
    ].to_numpy()

    expected_f = (
        TRUE_OOIP
        * expansion
        + we
    )

    max_error = np.max(
        np.abs(
            f - expected_f
        )
    )

    print(
        f"Maximum |F - (N×E + We)| : "
        f"{max_error:.6e} rb"
    )

    assert max_error < 1e-6

    print(
        "✓ F = N_true × E + We "
        "check PASSED."
    )

    # =================================================================
    # 9. F - We = N × E
    # =================================================================

    f_minus_we = (
        f - we
    )

    expected_f_minus_we = (
        TRUE_OOIP
        * expansion
    )

    max_error = np.max(
        np.abs(
            f_minus_we
            - expected_f_minus_we
        )
    )

    print(
        f"Maximum |(F-We) - N×E| : "
        f"{max_error:.6e} rb"
    )

    assert max_error < 1e-6

    print(
        "✓ F - We = N_true × E "
        "check PASSED."
    )

    # =================================================================
    # 10. PRODUCTION HISTORY CONSISTENCY
    # =================================================================

    print("\n")
    print(
        "## PRODUCTION HISTORY CONSISTENCY"
    )

    np_stb = df[
        "Np_STB"
    ].to_numpy()

    gp_mscf = df[
        "Gp_Mscf"
    ].to_numpy()

    wp_stb = df[
        "Wp_STB"
    ].to_numpy()

    bo = df[
        "Bo_rb_per_STB"
    ].to_numpy()

    bg = df[
        "Bg_rb_per_scf"
    ].to_numpy()

    rs = df[
        "Rs_scf_per_STB"
    ].to_numpy()

    rp = df[
        "Rp_scf_per_STB"
    ].to_numpy()

    # Production underground withdrawal:
    #
    # F = Np [Bo + (Rp - Rs)Bg] + Wp Bw
    #
    # Synthetic reservoir uses Wp = 0.

    production_f = (
        np_stb
        * (
            bo
            + (
                rp - rs
            )
            * bg
        )
        + wp_stb
        * df[
            "Bw_rb_per_STB"
        ].to_numpy()
    )

    production_f[0] = 0.0

    max_production_f_error = np.max(
        np.abs(
            production_f
            - f
        )
    )

    print(
        f"Maximum production F error : "
        f"{max_production_f_error:.6e} rb"
    )

    assert max_production_f_error < 1e-6

    print(
        "✓ Production → F consistency PASSED."
    )

    # =================================================================
    # 11. APPARENT OOIP
    # =================================================================

    print("\n")
    print(
        "## APPARENT OOIP CHECK"
    )

    valid = (
        expansion > 0
    )

    apparent_n = (
        (
            f[valid]
            - we[valid]
        )
        / expansion[valid]
    )

    max_n_error = np.max(
        np.abs(
            apparent_n
            - TRUE_OOIP
        )
    )

    print(
        f"True OOIP                : "
        f"{TRUE_OOIP:,.2f} STB"
    )

    print(
        f"Maximum apparent N error : "
        f"{max_n_error:.6f} STB"
    )

    assert max_n_error < 1e-4

    print(
        "✓ Apparent OOIP check PASSED."
    )

    # =================================================================
    # 12. INITIAL CONDITIONS
    # =================================================================

    print("\n")
    print(
        "## INITIAL-CONDITION CHECK"
    )

    assert np.isclose(
        df[
            "We_rb"
        ].iloc[0],
        0.0,
    )

    assert np.isclose(
        df[
            "F_rb"
        ].iloc[0],
        0.0,
    )

    assert np.isclose(
        df[
            "Np_STB"
        ].iloc[0],
        0.0,
    )

    assert np.isclose(
        df[
            "Gp_Mscf"
        ].iloc[0],
        0.0,
    )

    print(
        f"Initial pressure : "
        f"{pressure[0]:.6f} psia"
    )

    print(
        f"Initial We       : "
        f"{we[0]:.6f} rb"
    )

    print(
        f"Initial F        : "
        f"{f[0]:.6f} rb"
    )

    print(
        "✓ Initial-condition "
        "check PASSED."
    )

    # =================================================================
    # 13. V-E-H PARAMETER CHECK
    # =================================================================

    print("\n")
    print(
        "## V-E-H PARAMETER CHECK"
    )

    model_parameters = generator.config

    assert (
        model_parameters
        .veh_aquifer_permeability_md
        == 100.0
    )

    assert (
        model_parameters
        .veh_aquifer_porosity
        == 0.20
    )

    assert (
        model_parameters
        .veh_aquifer_thickness_ft
        == 100.0
    )

    assert (
        model_parameters
        .veh_total_compressibility_psi_inv
        == 2.0e-5
    )

    assert (
        model_parameters
        .veh_water_viscosity_cp
        == 0.50
    )

    assert (
        model_parameters
        .veh_reservoir_radius_ft
        == 1000.0
    )

    print(
        "✓ V-E-H parameter configuration "
        "check PASSED."
    )

    # =================================================================
    # 14. SAVE CSV
    # =================================================================

    print("\n")
    print(
        "## SAVING SYNTHETIC DATASET"
    )

    saved_path = generator.save_dataset(
        df,
        OUTPUT_PATH,
    )

    assert saved_path.exists()

    print(
        "✓ Synthetic dataset saved:"
    )

    print(
        saved_path
    )

    # =================================================================
    # 15. RELOAD CHECK
    # =================================================================

    reloaded = pd.read_csv(
        saved_path
    )

    assert len(
        reloaded
    ) == N_POINTS

    assert np.isclose(
        reloaded[
            "We_rb"
        ].iloc[-1],
        we[-1],
    )

    assert np.isclose(
        reloaded[
            "F_rb"
        ].iloc[-1],
        f[-1],
    )

    print(
        "✓ Synthetic dataset reload "
        "check PASSED."
    )

    # =================================================================
    # 16. DISPLAY DATA
    # =================================================================

    print("\n")
    print(
        "## SYNTHETIC DATASET"
    )

    print(
        df[
            [
                "Date",
                "Pressure_psia",
                "Np_STB",
                "Gp_Mscf",
                "We_rb",
                "F_rb",
                "Etotal_rb_per_STB",
                "influx_rate_rb_per_day",
            ]
        ].head(10)
    )

    # =================================================================
    # FINAL
    # =================================================================

    print("\n")
    print("=" * 70)

    print(
        "✓ ALL VAN EVERDINGEN-HURST "
        "SYNTHETIC RESERVOIR CHECKS PASSED"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()