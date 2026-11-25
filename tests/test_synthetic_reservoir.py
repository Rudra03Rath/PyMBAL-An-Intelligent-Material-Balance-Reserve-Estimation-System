"""
PyMBAL - Synthetic Reservoir Generator Test

Validates that the synthetic reservoir generator itself
satisfies the intended material-balance relationships:

    F = N_true * E

and

    F = Np [Bo + (Rp - Rs) Bg] + Wp Bw

before the generated dataset is passed into PyMBAL.
"""

import numpy as np

from src.models.synthetic_reservoir import (
    SyntheticReservoir
)


def main():

    print("=" * 70)
    print(
        "PyMBAL - SYNTHETIC RESERVOIR VALIDATION"
    )
    print("=" * 70)

    # ========================================================
    # 1. CREATE SYNTHETIC RESERVOIR
    # ========================================================

    reservoir = SyntheticReservoir(
        N_true=10_000_000.0,

        initial_pressure_psia=4000.0,

        temperature_F=200.0,

        initial_water_saturation=0.25,

        formation_compressibility_psi_inv=6.0e-6,

        water_compressibility_psi_inv=3.0e-6,

        oil_api=35.0,

        gas_gravity=0.80,

        salinity_wt_pct=5.0,

        gas_cap_ratio=0.0,
    )

    reservoir.display_ground_truth()

    # ========================================================
    # 2. GENERATE DATASET
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "GENERATING SYNTHETIC DATASET"
    )
    print("=" * 70)

    df = reservoir.generate_dataset(
        final_pressure_psia=2500.0,
        n_points=30,
    )

    print(
        f"\n✓ Generated {len(df)} data points."
    )

    # ========================================================
    # 3. CHECK F = N_TRUE × E
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 1: MATERIAL BALANCE CONSISTENCY"
    )
    print("=" * 70)

    expected_F = (
        reservoir.N_true
        *
        df[
            "Etotal_rb_per_STB"
        ]
    )

    actual_F = (
        df[
            "F_rb"
        ]
    )

    balance_error = (
        actual_F
        -
        expected_F
    )

    max_error = np.max(
        np.abs(
            balance_error
        )
    )

    print(
        f"Maximum |F - N×E| : "
        f"{max_error:.6e} rb"
    )

    if not np.allclose(
        actual_F.to_numpy(),
        expected_F.to_numpy(),
        rtol=1e-10,
        atol=1e-8,
    ):

        raise AssertionError(
            "Synthetic reservoir failed "
            "the F = N_true × E consistency check."
        )

    print(
        "✓ F = N_true × E check PASSED."
    )

    # ========================================================
    # 4. RECONSTRUCT F FROM PRODUCTION
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 2: PRODUCTION → UNDERGROUND WITHDRAWAL"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # The first row represents the initial reservoir state:
    #
    #     Np = 0
    #     Gp = 0
    #     Wp = 0
    #
    # Therefore:
    #
    #     Rp = Gp / Np
    #
    # is undefined for that row.
    #
    # Exclude the initial reference row from this check.
    # --------------------------------------------------------

    production_rows = (
        df[
            "Np_STB"
        ]
        >
        0
    )

    test_df = (
        df.loc[
            production_rows
        ]
        .copy()
    )

    if test_df.empty:

        raise AssertionError(
            "No production rows were generated."
        )

    # --------------------------------------------------------
    # Calculate producing GOR
    #
    # Gp is stored in Mscf.
    #
    # Therefore:
    #
    #     Gp_scf = Gp_Mscf × 1000
    #
    # and:
    #
    #     Rp = Gp_scf / Np
    # --------------------------------------------------------

    Rp = (

        test_df[
            "Gp_Mscf"
        ]

        *
        1000.0

        /

        test_df[
            "Np_STB"
        ]
    )

    # --------------------------------------------------------
    # Reconstruct underground withdrawal:
    #
    # F =
    # Np [Bo + (Rp - Rs) Bg]
    # + Wp Bw
    # --------------------------------------------------------

    reconstructed_F = (

        test_df[
            "Np_STB"
        ]

        *

        (

            test_df[
                "Bo_rb_per_STB"
            ]

            +

            (

                Rp

                -

                test_df[
                    "Rs_scf_per_STB"
                ]
            )

            *

            test_df[
                "Bg_rb_per_scf"
            ]
        )

        +

        test_df[
            "Wp_STB"
        ]

        *

        test_df[
            "Bw_rb_per_STB"
        ]
    )

    production_error = (

        reconstructed_F

        -

        test_df[
            "F_rb"
        ]
    )

    max_production_error = np.max(
        np.abs(
            production_error
        )
    )

    print(
        f"Maximum production F error : "
        f"{max_production_error:.6e} rb"
    )

    if not np.allclose(
        reconstructed_F.to_numpy(),
        test_df[
            "F_rb"
        ].to_numpy(),
        rtol=1e-10,
        atol=1e-8,
    ):

        raise AssertionError(
            "Production history does not "
            "reproduce the required F."
        )

    print(
        "✓ Production → F check PASSED."
    )

    # ========================================================
    # 5. CHECK APPARENT OOIP
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 3: APPARENT OOIP"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Ignore the initial state where:
    #
    #     E = 0
    #     F = 0
    #
    # because F / E is undefined there.
    # --------------------------------------------------------

    valid = (

        df[
            "Etotal_rb_per_STB"
        ]

        >

        1e-12
    )

    N_apparent = (

        df.loc[
            valid,
            "F_rb"
        ]

        /

        df.loc[
            valid,
            "Etotal_rb_per_STB"
        ]
    )

    N_error = (

        N_apparent

        -

        reservoir.N_true
    )

    max_N_error = np.max(
        np.abs(
            N_error
        )
    )

    print(
        f"True OOIP                : "
        f"{reservoir.N_true:,.2f} STB"
    )

    print(
        f"Maximum apparent N error : "
        f"{max_N_error:,.6f} STB"
    )

    if not np.allclose(
        N_apparent.to_numpy(),
        reservoir.N_true,
        rtol=1e-10,
        atol=1e-6,
    ):

        raise AssertionError(
            "Synthetic reservoir does not "
            "recover the specified true OOIP."
        )

    print(
        "✓ Apparent OOIP check PASSED."
    )

    # ========================================================
    # 6. DISPLAY SAMPLE DATA
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "SAMPLE SYNTHETIC DATA"
    )
    print("=" * 70)

    columns = [

        "Date",

        "Pressure_psia",

        "Np_STB",

        "Gp_Mscf",

        "Wp_STB",

        "Bo_rb_per_STB",

        "Bg_rb_per_scf",

        "Rs_scf_per_STB",

        "F_rb",

        "Etotal_rb_per_STB",
    ]

    print(
        df[
            columns
        ].head(10)
    )

    # ========================================================
    # 7. SAVE DATASET
    # ========================================================

    output_path = (
        "data/sample_data/"
        "synthetic_closed_oil_reservoir.csv"
    )

    reservoir.save_dataset(
        df,
        output_path
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "✓ ALL SYNTHETIC RESERVOIR CHECKS PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":

    main()