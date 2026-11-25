"""
PyMBAL - Gas-Cap Synthetic Reservoir Test

Validates the synthetic gas-cap reservoir generator.

Ground truth:

    N_true = 10 MMSTB
    m_true = 0.25

The synthetic reservoir must satisfy:

    E = Eo + mEg + Efw

and:

    F = N_true * E

The test also independently reconstructs F from
the generated production history.
"""

import numpy as np

from src.models.synthetic_reservoir import (
    SyntheticReservoir
)


def main():

    print("=" * 70)
    print(
        "PyMBAL - GAS-CAP SYNTHETIC RESERVOIR VALIDATION"
    )
    print("=" * 70)

    # ========================================================
    # 1. CREATE GAS-CAP SYNTHETIC RESERVOIR
    # ========================================================

    N_true = 10_000_000.0
    m_true = 0.25

    reservoir = SyntheticReservoir(

        N_true=N_true,

        initial_pressure_psia=4000.0,

        temperature_F=200.0,

        initial_water_saturation=0.25,

        formation_compressibility_psi_inv=6.0e-6,

        water_compressibility_psi_inv=3.0e-6,

        oil_api=35.0,

        gas_gravity=0.80,

        salinity_wt_pct=5.0,

        gas_cap_ratio=m_true,
    )

    reservoir.display_ground_truth()

    # ========================================================
    # 2. GENERATE DATASET
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "GENERATING GAS-CAP SYNTHETIC DATASET"
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
    # 3. CHECK GAS-CAP EXPANSION
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 1: GAS-CAP EXPANSION"
    )
    print("=" * 70)

    expected_mEg = (

        m_true

        *

        df[
            "Eg_rb_per_STB"
        ]
    )

    actual_mEg = (
        df[
            "mEg_rb_per_STB"
        ]
    )

    mEg_error = (
        actual_mEg
        -
        expected_mEg
    )

    max_mEg_error = np.max(
        np.abs(
            mEg_error
        )
    )

    print(
        f"Gas-cap ratio m       : "
        f"{m_true:.4f}"
    )

    print(
        f"Maximum |mEg - m×Eg| : "
        f"{max_mEg_error:.6e} rb/STB"
    )

    if not np.allclose(

        actual_mEg.to_numpy(),

        expected_mEg.to_numpy(),

        rtol=1e-10,

        atol=1e-12,
    ):

        raise AssertionError(
            "Gas-cap expansion term does not "
            "satisfy mEg = m × Eg."
        )

    print(
        "✓ Gas-cap expansion check PASSED."
    )

    # ========================================================
    # 4. CHECK TOTAL EXPANSION
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 2: TOTAL EXPANSION"
    )
    print("=" * 70)

    expected_E = (

        df[
            "Eo_rb_per_STB"
        ]

        +

        df[
            "mEg_rb_per_STB"
        ]

        +

        df[
            "Efw_rb_per_STB"
        ]
    )

    actual_E = (
        df[
            "Etotal_rb_per_STB"
        ]
    )

    expansion_error = (
        actual_E
        -
        expected_E
    )

    max_expansion_error = np.max(
        np.abs(
            expansion_error
        )
    )

    print(
        "Expected:"
    )

    print(
        "    E = Eo + mEg + Efw"
    )

    print(
        f"Maximum expansion error : "
        f"{max_expansion_error:.6e} rb/STB"
    )

    if not np.allclose(

        actual_E.to_numpy(),

        expected_E.to_numpy(),

        rtol=1e-10,

        atol=1e-12,
    ):

        raise AssertionError(
            "Total expansion does not satisfy "
            "E = Eo + mEg + Efw."
        )

    print(
        "✓ Total expansion check PASSED."
    )

    # ========================================================
    # 5. CHECK F = N_TRUE × E
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 3: MATERIAL BALANCE CONSISTENCY"
    )
    print("=" * 70)

    expected_F = (

        N_true

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

    max_balance_error = np.max(
        np.abs(
            balance_error
        )
    )

    print(
        f"True OOIP              : "
        f"{N_true:,.2f} STB"
    )

    print(
        f"Maximum |F - N×E|     : "
        f"{max_balance_error:.6e} rb"
    )

    if not np.allclose(

        actual_F.to_numpy(),

        expected_F.to_numpy(),

        rtol=1e-10,

        atol=1e-8,
    ):

        raise AssertionError(
            "Gas-cap synthetic reservoir failed "
            "the F = N_true × E consistency check."
        )

    print(
        "✓ F = N_true × E check PASSED."
    )

    # ========================================================
    # 6. RECONSTRUCT F FROM PRODUCTION
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 4: PRODUCTION → UNDERGROUND WITHDRAWAL"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Ignore initial reference row.
    #
    # At initial conditions:
    #
    #     Np = 0
    #     Gp = 0
    #
    # Therefore Rp = Gp/Np is undefined.
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
    # Producing GOR
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
    # Reconstruct:
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
            "Gas-cap production history does not "
            "reproduce the required F."
        )

    print(
        "✓ Production → F check PASSED."
    )

    # ========================================================
    # 7. CHECK APPARENT OOIP
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 5: APPARENT OOIP"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Ignore initial point where E = 0.
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

        N_true
    )

    max_N_error = np.max(
        np.abs(
            N_error
        )
    )

    print(
        f"True OOIP                : "
        f"{N_true:,.2f} STB"
    )

    print(
        f"Maximum apparent N error : "
        f"{max_N_error:,.6f} STB"
    )

    if not np.allclose(

        N_apparent.to_numpy(),

        N_true,

        rtol=1e-10,

        atol=1e-6,
    ):

        raise AssertionError(
            "Gas-cap synthetic reservoir does not "
            "recover the specified true OOIP."
        )

    print(
        "✓ Apparent OOIP check PASSED."
    )

    # ========================================================
    # 8. VERIFY GAS-CAP ACTUALLY CONTRIBUTES
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 6: GAS-CAP CONTRIBUTION IS NON-ZERO"
    )
    print("=" * 70)

    # Ignore initial state.
    non_initial_mEg = (
        df.loc[
            valid,
            "mEg_rb_per_STB"
        ]
    )

    max_mEg = np.max(
        np.abs(
            non_initial_mEg.to_numpy()
        )
    )

    print(
        f"Maximum mEg contribution : "
        f"{max_mEg:.6e} rb/STB"
    )

    if max_mEg <= 0:

        raise AssertionError(
            "Gas-cap ratio is non-zero, but the "
            "gas-cap expansion contribution is zero."
        )

    print(
        "✓ Gas-cap contribution is active."
    )

    # ========================================================
    # 9. DISPLAY SAMPLE DATA
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "SAMPLE GAS-CAP SYNTHETIC DATA"
    )
    print("=" * 70)

    columns = [

        "Date",

        "Pressure_psia",

        "Np_STB",

        "Gp_Mscf",

        "Bo_rb_per_STB",

        "Bg_rb_per_scf",

        "Rs_scf_per_STB",

        "Eo_rb_per_STB",

        "Eg_rb_per_STB",

        "mEg_rb_per_STB",

        "Efw_rb_per_STB",

        "Etotal_rb_per_STB",

        "F_rb",
    ]

    print(
        df[
            columns
        ].head(10)
    )

    # ========================================================
    # 10. SAVE DATASET
    # ========================================================

    output_path = (
        "data/sample_data/"
        "synthetic_gas_cap_reservoir.csv"
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
        "✓ ALL GAS-CAP SYNTHETIC RESERVOIR "
        "CHECKS PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":

    main()