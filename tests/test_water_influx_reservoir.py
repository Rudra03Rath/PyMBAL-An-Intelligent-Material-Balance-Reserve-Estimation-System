"""
PyMBAL - Water Influx Synthetic Reservoir Validation

Validates the synthetic reservoir with Schilthuis water influx.

Ground truth:

    N_true = 10 MMSTB
    m = 0
    We > 0

The synthetic reservoir must satisfy:

    F = N_true * E + We

Therefore:

    F - We = N_true * E

The test verifies:

    1. Water-influx dataset generation
    2. Schilthuis influx consistency
    3. Positive cumulative water influx
    4. F = N_true*E + We
    5. F - We = N_true*E
    6. Apparent OOIP recovery
    7. Monotonic cumulative water influx
    8. Production history consistency
    9. Dataset saving
"""


from pathlib import Path

import numpy as np

from src.models.synthetic_reservoir import (
    SyntheticReservoir
)

from src.influx.schilthuis import (
    SchilthuisAquifer
)


# ============================================================
# NUMERICAL CHECK HELPER
# ============================================================

def check_close(
    actual,
    expected,
    message,
    rtol=1e-10,
    atol=1e-8,
):
    """
    Compare numerical values or arrays.
    """

    if not np.allclose(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
    ):

        raise AssertionError(
            message
        )


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 70)
    print(
        "PyMBAL - WATER INFLUX SYNTHETIC RESERVOIR VALIDATION"
    )
    print("=" * 70)

    # ========================================================
    # GROUND-TRUTH PARAMETERS
    # ========================================================

    N_true = 10_000_000.0

    m = 0.0

    Pe = 4000.0

    C = 2.0

    final_pressure = 2500.0

    n_points = 30

    print("\n")
    print(
        "GROUND TRUTH"
    )
    print("-" * 60)

    print(
        f"True OOIP              : "
        f"{N_true:,.0f} STB"
    )

    print(
        f"True OOIP              : "
        f"{N_true / 1e6:.4f} MMSTB"
    )

    print(
        f"Gas Cap Ratio (m)      : "
        f"{m:.4f}"
    )

    print(
        f"Aquifer Pressure      : "
        f"{Pe:.2f} psia"
    )

    print(
        f"Influx Constant        : "
        f"{C:.6f} rb/(day·psi)"
    )

    print(
        f"Final Reservoir Pressure: "
        f"{final_pressure:.2f} psia"
    )

    # ========================================================
    # CREATE SYNTHETIC RESERVOIR
    # ========================================================

    reservoir = SyntheticReservoir(

        N_true=N_true,

        gas_cap_ratio=m,
    )

    reservoir.display_ground_truth(

        water_influx=True,

        aquifer_pressure_psia=Pe,

        influx_constant_rb_per_day_psi=C,
    )

    # ========================================================
    # GENERATE DATASET
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "GENERATING WATER-INFLUX SYNTHETIC DATASET"
    )
    print("=" * 70)

    df = reservoir.generate_dataset(

        final_pressure_psia=
            final_pressure,

        n_points=
            n_points,

        water_influx=True,

        aquifer_pressure_psia=
            Pe,

        influx_constant_rb_per_day_psi=
            C,
    )

    print(
        f"\n✓ Generated {len(df)} data points."
    )

    # ========================================================
    # CHECK 1:
    # WATER INFLUX IS ACTIVE
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 1: WATER INFLUX IS ACTIVE"
    )
    print("=" * 70)

    We = (
        df[
            "We_rb"
        ]
        .to_numpy()
    )

    max_We = np.max(
        We
    )

    print(
        f"Maximum cumulative We : "
        f"{max_We:.6f} rb"
    )

    if max_We <= 0:

        raise AssertionError(
            "Water influx is not active."
        )

    print(
        "✓ Water influx is active."
    )

    # ========================================================
    # CHECK 2:
    # SCHILTHUIS MODEL CONSISTENCY
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 2: SCHILTHUIS INFLUX CONSISTENCY"
    )
    print("=" * 70)

    aquifer = SchilthuisAquifer(

        aquifer_pressure_psia=Pe,

        influx_constant_rb_per_day_psi=C,
    )

    expected_influx = (
        aquifer.calculate_influx(

            pressure_psia=
                df[
                    "Pressure_psia"
                ],

            time_days=
                df[
                    "Days"
                ],
        )
    )

    actual_We = (
        df[
            "We_rb"
        ]
        .to_numpy()
    )

    expected_We = (
        expected_influx[
            "We_rb"
        ]
        .to_numpy()
    )

    max_We_error = np.max(
        np.abs(
            actual_We
            -
            expected_We
        )
    )

    print(
        f"Maximum |We_generated - We_Schilthuis| : "
        f"{max_We_error:.6e} rb"
    )

    check_close(

        actual_We,

        expected_We,

        "Synthetic water influx does not "
        "match the Schilthuis model.",
    )

    print(
        "✓ Schilthuis water-influx check PASSED."
    )

    # ========================================================
    # CHECK 3:
    # POSITIVE / MONOTONIC CUMULATIVE INFLUX
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 3: CUMULATIVE WATER INFLUX BEHAVIOR"
    )
    print("=" * 70)

    if not np.all(
        We >= 0
    ):

        raise AssertionError(
            "Negative cumulative water influx detected."
        )

    if not np.all(
        np.diff(We) >= -1e-10
    ):

        raise AssertionError(
            "Cumulative water influx is not monotonic."
        )

    print(
        "Initial We : "
        f"{We[0]:.6f} rb"
    )

    print(
        "Final We   : "
        f"{We[-1]:.6f} rb"
    )

    print(
        "✓ Cumulative water-influx behavior PASSED."
    )

    # ========================================================
    # CHECK 4:
    # F = N_TRUE * E + We
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 4: MATERIAL BALANCE WITH WATER INFLUX"
    )
    print("=" * 70)

    expected_F = (

        N_true

        *

        df[
            "Etotal_rb_per_STB"
        ].to_numpy()

        +

        df[
            "We_rb"
        ].to_numpy()
    )

    actual_F = (
        df[
            "F_rb"
        ].to_numpy()
    )

    max_F_error = np.max(
        np.abs(
            actual_F
            -
            expected_F
        )
    )

    print(
        f"Maximum |F - (N×E + We)| : "
        f"{max_F_error:.6e} rb"
    )

    check_close(

        actual_F,

        expected_F,

        "Synthetic F does not satisfy "
        "F = N×E + We.",
    )

    print(
        "✓ F = N_true × E + We check PASSED."
    )

    # ========================================================
    # CHECK 5:
    # F - We = N_TRUE * E
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 5: CORRECTED MATERIAL BALANCE"
    )
    print("=" * 70)

    corrected_F = (

        df[
            "F_rb"
        ].to_numpy()

        -

        df[
            "We_rb"
        ].to_numpy()
    )

    expected_corrected_F = (

        N_true

        *

        df[
            "Etotal_rb_per_STB"
        ].to_numpy()
    )

    max_corrected_error = np.max(
        np.abs(
            corrected_F
            -
            expected_corrected_F
        )
    )

    print(
        f"Maximum |(F - We) - N×E| : "
        f"{max_corrected_error:.6e} rb"
    )

    check_close(

        corrected_F,

        expected_corrected_F,

        "Corrected material balance "
        "F - We = N×E failed.",
    )

    print(
        "✓ F - We = N_true × E check PASSED."
    )

    # ========================================================
    # CHECK 6:
    # APPARENT OOIP RECOVERY
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 6: APPARENT OOIP RECOVERY"
    )
    print("=" * 70)

    E = (
        df[
            "Etotal_rb_per_STB"
        ].to_numpy()
    )

    # --------------------------------------------------------
    # Ignore initial point where E = 0.
    # --------------------------------------------------------

    valid = (
        E > 0
    )

    apparent_N = (

        corrected_F[valid]

        /

        E[valid]
    )

    max_N_error = np.max(
        np.abs(
            apparent_N
            -
            N_true
        )
    )

    print(
        f"True OOIP              : "
        f"{N_true:,.2f} STB"
    )

    print(
        f"Maximum apparent N error : "
        f"{max_N_error:.6f} STB"
    )

    check_close(

        apparent_N,

        np.full_like(
            apparent_N,
            N_true,
        ),

        "Apparent OOIP does not recover "
        "the known N_true.",
    )

    print(
        "✓ Apparent OOIP check PASSED."
    )

    # ========================================================
    # CHECK 7:
    # PRODUCTION HISTORY CONSISTENCY
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 7: PRODUCTION HISTORY CONSISTENCY"
    )
    print("=" * 70)

    Rp = 800.0

    production_F = (

        df[
            "Np_STB"
        ].to_numpy()

        *

        (

            df[
                "Bo_rb_per_STB"
            ].to_numpy()

            +

            (

                Rp

                -

                df[
                    "Rs_scf_per_STB"
                ].to_numpy()
            )

            *

            df[
                "Bg_rb_per_scf"
            ].to_numpy()
        )
    )

    max_production_error = np.max(
        np.abs(
            production_F
            -
            df[
                "F_rb"
            ].to_numpy()
        )
    )

    print(
        f"Maximum production F error : "
        f"{max_production_error:.6e} rb"
    )

    check_close(

        production_F,

        df[
            "F_rb"
        ].to_numpy(),

        "Production history does not "
        "reproduce the required F.",
    )

    # --------------------------------------------------------
    # Verify gas production
    # --------------------------------------------------------

    expected_Gp_Mscf = (

        df[
            "Np_STB"
        ].to_numpy()

        *

        Rp

        /

        1000.0
    )

    max_Gp_error = np.max(
        np.abs(
            df[
                "Gp_Mscf"
            ].to_numpy()

            -

            expected_Gp_Mscf
        )
    )

    print(
        f"Maximum Gp error          : "
        f"{max_Gp_error:.6e} Mscf"
    )

    check_close(

        df[
            "Gp_Mscf"
        ].to_numpy(),

        expected_Gp_Mscf,

        "Gas production history is inconsistent.",
    )

    print(
        "✓ Production history consistency PASSED."
    )

    # ========================================================
    # CHECK 8:
    # INITIAL CONDITIONS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 8: INITIAL CONDITIONS"
    )
    print("=" * 70)

    first_row = df.iloc[0]

    if abs(
        first_row["Pressure_psia"]
        -
        reservoir.initial_pressure_psia
    ) > 1e-10:

        raise AssertionError(
            "Initial pressure is incorrect."
        )

    if abs(
        first_row["We_rb"]
    ) > 1e-10:

        raise AssertionError(
            "Initial cumulative water influx "
            "must be zero."
        )

    if abs(
        first_row["F_rb"]
    ) > 1e-10:

        raise AssertionError(
            "Initial F must be zero."
        )

    if abs(
        first_row["Np_STB"]
    ) > 1e-10:

        raise AssertionError(
            "Initial Np must be zero."
        )

    print(
        "Initial pressure : "
        f"{first_row['Pressure_psia']:.2f} psia"
    )

    print(
        "Initial We        : "
        f"{first_row['We_rb']:.6f} rb"
    )

    print(
        "Initial F         : "
        f"{first_row['F_rb']:.6f} rb"
    )

    print(
        "✓ Initial-condition check PASSED."
    )

    # ========================================================
    # CHECK 9:
    # SAVE DATASET
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 9: SAVE SYNTHETIC DATASET"
    )
    print("=" * 70)

    output_path = Path(
        "data/sample_data/"
        "synthetic_water_influx_reservoir.csv"
    )

    reservoir.save_dataset(
        df,
        output_path,
    )

    if not output_path.exists():

        raise AssertionError(
            "Synthetic water-influx dataset "
            "was not saved."
        )

    print(
        "✓ Synthetic dataset save check PASSED."
    )

    # ========================================================
    # DISPLAY SAMPLE DATA
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "SYNTHETIC WATER-INFLUX DATASET"
    )
    print("=" * 70)

    display_columns = [

        "Date",

        "Pressure_psia",

        "Np_STB",

        "Gp_Mscf",

        "We_rb",

        "F_rb",

        "Etotal_rb_per_STB",

        "F_minus_We_rb",
    ]

    print(
        df[
            display_columns
        ].head(10)
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "✓ ALL WATER-INFLUX SYNTHETIC RESERVOIR "
        "CHECKS PASSED"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()