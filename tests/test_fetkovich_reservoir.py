"""
PyMBAL - Fetkovich Synthetic Reservoir Validation

Validates the complete synthetic-reservoir workflow for a finite
Fetkovich aquifer.

Ground truth:
    OOIP = 10 MMSTB
    m   = 0
    Pai = 4000 psia
    Wi  = 5,000,000 rb
    ct  = 20e-6 psi^-1
    J   = 100 rb/(day*psi)

The test verifies:

    1. Synthetic dataset generation
    2. Fetkovich water influx
    3. F = N*E + We
    4. F - We = N*E
    5. Production history consistency
    6. Aquifer pressure decline
    7. Influx-rate behavior
    8. Apparent OOIP recovery
    9. Ground-truth OOIP recovery
   10. Dataset save
"""


from pathlib import Path

import numpy as np

from src.models.synthetic_reservoir import (
    SyntheticReservoir,
)


# ============================================================
# HELPER
# ============================================================

def check_close(
    actual,
    expected,
    message,
    rtol=1e-10,
    atol=1e-8,
):
    """
    Raise AssertionError if two arrays/scalars are not
    numerically equal within tolerance.
    """

    if not np.allclose(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
    ):
        raise AssertionError(message)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "PyMBAL - FETKOVICH SYNTHETIC RESERVOIR VALIDATION"
    )
    print("=" * 70)

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    N_true = 10_000_000.0

    initial_pressure = 4000.0

    final_pressure = 2500.0

    gas_cap_ratio = 0.0

    aquifer_pressure = 4000.0

    initial_water_volume = 5_000_000.0

    total_compressibility = 20.0e-6

    productivity_index = 100.0

    n_points = 30

    print("\n")
    print("=" * 60)
    print("GROUND TRUTH")
    print("=" * 60)

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
        f"{gas_cap_ratio:.4f}"
    )

    print(
        f"Aquifer Pressure      : "
        f"{aquifer_pressure:.2f} psia"
    )

    print(
        f"Initial Aquifer Volume: "
        f"{initial_water_volume:,.0f} rb"
    )

    print(
        f"Aquifer ct            : "
        f"{total_compressibility:.6e} psi⁻¹"
    )

    print(
        f"Productivity Index     : "
        f"{productivity_index:.4f} "
        f"rb/(day·psi)"
    )

    print(
        f"Final Reservoir Pressure: "
        f"{final_pressure:.2f} psia"
    )

    # ========================================================
    # CONSTRUCT SYNTHETIC RESERVOIR
    # ========================================================

    reservoir = SyntheticReservoir(

        N_true=N_true,

        initial_pressure_psia=
            initial_pressure,

        gas_cap_ratio=
            gas_cap_ratio,
    )

    reservoir.display_ground_truth(

        water_influx=True,

        aquifer_model=
            "Fetkovich finite aquifer",

        aquifer_pressure_psia=
            aquifer_pressure,

        initial_water_volume_rb=
            initial_water_volume,

        total_compressibility_psi_inv=
            total_compressibility,

        productivity_index_rb_per_day_psi=
            productivity_index,
    )

    # ========================================================
    # GENERATE DATASET
    # ========================================================

    print("\n")
    print("=" * 60)
    print("GENERATING FETKOVICH SYNTHETIC DATASET")
    print("=" * 60)

    df = reservoir.generate_dataset(

        final_pressure_psia=
            final_pressure,

        n_points=
            n_points,

        water_influx_model=
            "fetkovich",

        aquifer_pressure_psia=
            aquifer_pressure,

        initial_water_volume_rb=
            initial_water_volume,

        total_compressibility_psi_inv=
            total_compressibility,

        productivity_index_rb_per_day_psi=
            productivity_index,
    )

    print(
        f"\n✓ Generated {len(df)} data points."
    )

    if len(df) != n_points:

        raise AssertionError(
            "Generated dataset contains "
            "incorrect number of data points."
        )

    # ========================================================
    # CHECK 1:
    # WATER INFLUX IS ACTIVE
    # ========================================================

    print("\n")
    print("=" * 60)
    print("CHECK 1: WATER INFLUX ACTIVITY")
    print("=" * 60)

    We = df[
        "We_rb"
    ].to_numpy()

    max_We = np.max(
        We
    )

    print(
        f"Maximum cumulative We : "
        f"{max_We:.6f} rb"
    )

    if max_We <= 0:

        raise AssertionError(
            "Fetkovich water influx is not active."
        )

    print(
        "✓ Water influx is active."
    )

    # ========================================================
    # CHECK 2:
    # F = N × E + We
    # ========================================================

    print("\n")
    print("=" * 60)
    print(
        "CHECK 2: MATERIAL BALANCE CONSISTENCY"
    )
    print("=" * 60)

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

        "F = N×E + We consistency check failed.",
    )

    print(
        "✓ F = N_true × E + We check PASSED."
    )

    # ========================================================
    # CHECK 3:
    # F - We = N × E
    # ========================================================

    print("\n")
    print("=" * 60)
    print(
        "CHECK 3: WATER-INFLUX-CORRECTED MBE"
    )
    print("=" * 60)

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
        f"Maximum |(F-We) - N×E| : "
        f"{max_corrected_error:.6e} rb"
    )

    check_close(

        corrected_F,

        expected_corrected_F,

        "F - We = N×E check failed.",
    )

    print(
        "✓ F - We = N_true × E check PASSED."
    )

    # ========================================================
    # CHECK 4:
    # PRODUCTION HISTORY CONSISTENCY
    # ========================================================

    print("\n")
    print("=" * 60)
    print(
        "CHECK 4: PRODUCTION HISTORY CONSISTENCY"
    )
    print("=" * 60)

    Rp = 800.0

    production_coefficient = (

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

    calculated_F = (

        df[
            "Np_STB"
        ].to_numpy()

        *

        production_coefficient
    )

    actual_F = (
        df[
            "F_rb"
        ].to_numpy()
    )

    max_production_F_error = np.max(
        np.abs(
            calculated_F
            -
            actual_F
        )
    )

    print(
        f"Maximum production F error : "
        f"{max_production_F_error:.6e} rb"
    )

    check_close(

        calculated_F,

        actual_F,

        "Production history does not reproduce F.",
    )

    # --------------------------------------------------------
    # Gas production consistency
    # --------------------------------------------------------

    expected_Gp = (

        df[
            "Np_STB"
        ].to_numpy()

        *

        Rp

        /

        1000.0
    )

    actual_Gp = (
        df[
            "Gp_Mscf"
        ].to_numpy()
    )

    max_Gp_error = np.max(
        np.abs(
            actual_Gp
            -
            expected_Gp
        )
    )

    print(
        f"Maximum Gp error          : "
        f"{max_Gp_error:.6e} Mscf"
    )

    check_close(

        actual_Gp,

        expected_Gp,

        "Gas production history is inconsistent.",
    )

    print(
        "✓ Production history consistency PASSED."
    )

    # ========================================================
    # CHECK 5:
    # AQUIFER PRESSURE DECLINE
    # ========================================================

    print("\n")
    print("=" * 60)
    print(
        "CHECK 5: AQUIFER PRESSURE DECLINE"
    )
    print("=" * 60)

    Pa = (
        df[
            "aquifer_pressure_psia"
        ].to_numpy()
    )

    print(
        f"Initial aquifer pressure : "
        f"{Pa[0]:.6f} psia"
    )

    print(
        f"Final aquifer pressure   : "
        f"{Pa[-1]:.6f} psia"
    )

    if not np.all(
        np.diff(Pa) <= 1e-10
    ):

        raise AssertionError(
            "Aquifer pressure is not monotonically "
            "decreasing."
        )

    if not (
        Pa[-1] < Pa[0]
    ):

        raise AssertionError(
            "Aquifer pressure did not decline."
        )

    print(
        "✓ Aquifer pressure decline check PASSED."
    )

    # ========================================================
    # CHECK 6:
    # INFLUX RATE BEHAVIOR
    # ========================================================

    print("\n")
    print("=" * 60)
    print(
        "CHECK 6: INFLUX RATE BEHAVIOR"
    )
    print("=" * 60)

    rates = (
        df[
            "influx_rate_rb_per_day"
        ].to_numpy()
    )

    print(
        "First five influx rates:"
    )

    print(
        rates[:5]
    )

    if np.any(
        rates < -1e-10
    ):

        raise AssertionError(
            "Negative water-influx rate detected."
        )

    if rates[-1] <= 0:

        raise AssertionError(
            "Final Fetkovich influx rate "
            "should be positive."
        )

    print(
        "✓ Influx rates are physically valid."
    )

    # ========================================================
    # CHECK 7:
    # CUMULATIVE WE MONOTONICITY
    # ========================================================

    print("\n")
    print("=" * 60)
    print(
        "CHECK 7: CUMULATIVE WATER INFLUX"
    )
    print("=" * 60)

    if not np.all(
        We >= -1e-10
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
        f"Initial We : "
        f"{We[0]:.6f} rb"
    )

    print(
        f"Final We   : "
        f"{We[-1]:.6f} rb"
    )

    print(
        "✓ Cumulative water-influx behavior PASSED."
    )

    # ========================================================
    # CHECK 8:
    # FETKOVICH STORAGE CONSERVATION
    # ========================================================

    print("\n")
    print("=" * 60)
    print(
        "CHECK 8: FETKOVICH STORAGE CONSERVATION"
    )
    print("=" * 60)

    Ca = (

        initial_water_volume

        *

        total_compressibility
    )

    expected_Pa = (

        aquifer_pressure

        -

        We
        /
        Ca
    )

    storage_error = np.max(
        np.abs(
            Pa
            -
            expected_Pa
        )
    )

    print(
        f"Aquifer storage capacity : "
        f"{Ca:.6f} rb/psi"
    )

    print(
        f"Maximum storage error     : "
        f"{storage_error:.6e} psi"
    )

    check_close(

        Pa,

        expected_Pa,

        "Fetkovich aquifer storage conservation failed.",
    )

    print(
        "✓ Fetkovich storage conservation PASSED."
    )

    # ========================================================
    # CHECK 9:
    # APPARENT OOIP
    # ========================================================

    print("\n")
    print("=" * 60)
    print(
        "CHECK 9: APPARENT OOIP"
    )
    print("=" * 60)

    E = (
        df[
            "Etotal_rb_per_STB"
        ].to_numpy()
    )

    apparent_N = (

        corrected_F[1:]

        /

        E[1:]
    )

    max_N_error = np.max(
        np.abs(
            apparent_N
            -
            N_true
        )
    )

    print(
        f"True OOIP                : "
        f"{N_true:,.2f} STB"
    )

    print(
        f"Maximum apparent N error : "
        f"{max_N_error:.6f} STB"
    )

    check_close(

        apparent_N,

        np.full(
            len(apparent_N),
            N_true,
        ),

        "Apparent OOIP does not recover "
        "the known OOIP.",
        rtol=1e-9,
        atol=1e-3,
    )

    print(
        "✓ Apparent OOIP check PASSED."
    )

    # ========================================================
    # CHECK 10:
    # INITIAL CONDITIONS
    # ========================================================

    print("\n")
    print("=" * 60)
    print(
        "CHECK 10: INITIAL CONDITIONS"
    )
    print("=" * 60)

    print(
        f"Initial reservoir pressure : "
        f"{df['Pressure_psia'].iloc[0]:.6f} psia"
    )

    print(
        f"Initial aquifer pressure  : "
        f"{Pa[0]:.6f} psia"
    )

    print(
        f"Initial We                : "
        f"{We[0]:.6f} rb"
    )

    print(
        f"Initial F                 : "
        f"{df['F_rb'].iloc[0]:.6f} rb"
    )

    if not np.isclose(
        df[
            "Pressure_psia"
        ].iloc[0],
        initial_pressure,
        atol=1e-10,
    ):

        raise AssertionError(
            "Initial reservoir pressure is incorrect."
        )

    if not np.isclose(
        Pa[0],
        aquifer_pressure,
        atol=1e-10,
    ):

        raise AssertionError(
            "Initial aquifer pressure is incorrect."
        )

    if not np.isclose(
        We[0],
        0.0,
        atol=1e-10,
    ):

        raise AssertionError(
            "Initial water influx should be zero."
        )

    if not np.isclose(
        df[
            "F_rb"
        ].iloc[0],
        0.0,
        atol=1e-10,
    ):

        raise AssertionError(
            "Initial underground withdrawal "
            "should be zero."
        )

    print(
        "✓ Initial-condition check PASSED."
    )

    # ========================================================
    # DISPLAY DATA
    # ========================================================

    print("\n")
    print("=" * 60)
    print(
        "SAMPLE SYNTHETIC DATA"
    )
    print("=" * 60)

    print(
        df[
            [
                "Date",
                "Pressure_psia",
                "Np_STB",
                "Gp_Mscf",
                "We_rb",
                "aquifer_pressure_psia",
                "influx_rate_rb_per_day",
                "F_rb",
                "Etotal_rb_per_STB",
                "F_minus_We_rb",
            ]
        ].head(10)
    )

    # ========================================================
    # SAVE DATASET
    # ========================================================

    output_path = Path(
        "data/sample_data/"
        "synthetic_fetkovich_reservoir.csv"
    )

    reservoir.save_dataset(
        df,
        output_path,
    )

    if not output_path.exists():

        raise AssertionError(
            "Synthetic Fetkovich dataset "
            "was not saved."
        )

    print(
        "✓ Synthetic dataset save check PASSED."
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "✓ ALL FETKOVICH SYNTHETIC RESERVOIR CHECKS PASSED"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()