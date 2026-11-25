"""
PyMBAL - VAN EVERDINGEN-HURST AQUIFER VALIDATION

Validation tests for the infinite-acting radial
Van Everdingen-Hurst aquifer model.
"""

import numpy as np
import pandas as pd

from src.influx.van_everdingen_hurst import (
    VanEverdingenHurstAquifer,
)


# ============================================================
# TEST PARAMETERS
# ============================================================

K_MD = 100.0
PHI = 0.20
H_FT = 100.0
CT_PSI_INV = 2.0e-5
MU_W_CP = 0.5
R_E_FT = 1000.0
ANGLE_DEG = 360.0


# ============================================================
# MODEL FACTORY
# ============================================================

def build_model():

    return VanEverdingenHurstAquifer(

        aquifer_permeability_md=K_MD,

        aquifer_porosity=PHI,

        aquifer_thickness_ft=H_FT,

        total_compressibility_psi_inv=CT_PSI_INV,

        water_viscosity_cp=MU_W_CP,

        reservoir_radius_ft=R_E_FT,

        encroachment_angle_deg=ANGLE_DEG,
    )


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 70)
    print(
        "PyMBAL - VAN EVERDINGEN-HURST AQUIFER VALIDATION"
    )
    print("=" * 70)

    # ========================================================
    # 1. MODEL CONSTRUCTION
    # ========================================================

    print("\n")
    print("## Test Parameters")

    print(
        f"Aquifer permeability : "
        f"{K_MD:.2f} md"
    )

    print(
        f"Aquifer porosity     : "
        f"{PHI:.4f}"
    )

    print(
        f"Aquifer thickness    : "
        f"{H_FT:.2f} ft"
    )

    print(
        f"Total compressibility: "
        f"{CT_PSI_INV:.6e} psi⁻¹"
    )

    print(
        f"Water viscosity      : "
        f"{MU_W_CP:.4f} cp"
    )

    print(
        f"Reservoir radius     : "
        f"{R_E_FT:.2f} ft"
    )

    print(
        f"Encroachment angle   : "
        f"{ANGLE_DEG:.2f}°"
    )

    model = build_model()

    assert isinstance(
        model,
        VanEverdingenHurstAquifer,
    )

    assert (
        model.aquifer_constant_rb_per_psi
        > 0
    )

    print(
        f"\nAquifer constant B   : "
        f"{model.aquifer_constant_rb_per_psi:.6f} rb/psi"
    )

    print(
        "✓ Model construction check PASSED."
    )

    # ========================================================
    # 2. PARAMETER VALIDATION
    # ========================================================

    print("\n## Parameter Validation")

    invalid_cases = [

        (
            "negative permeability",
            dict(
                aquifer_permeability_md=-1.0
            ),
        ),

        (
            "zero porosity",
            dict(
                aquifer_porosity=0.0
            ),
        ),

        (
            "invalid porosity > 1",
            dict(
                aquifer_porosity=1.1
            ),
        ),

        (
            "negative thickness",
            dict(
                aquifer_thickness_ft=-10.0
            ),
        ),

        (
            "negative compressibility",
            dict(
                total_compressibility_psi_inv=-1e-5
            ),
        ),

        (
            "zero water viscosity",
            dict(
                water_viscosity_cp=0.0
            ),
        ),

        (
            "negative radius",
            dict(
                reservoir_radius_ft=-100.0
            ),
        ),

        (
            "invalid encroachment angle",
            dict(
                encroachment_angle_deg=400.0
            ),
        ),
    ]

    for name, overrides in invalid_cases:

        params = dict(

            aquifer_permeability_md=K_MD,

            aquifer_porosity=PHI,

            aquifer_thickness_ft=H_FT,

            total_compressibility_psi_inv=
                CT_PSI_INV,

            water_viscosity_cp=
                MU_W_CP,

            reservoir_radius_ft=
                R_E_FT,

            encroachment_angle_deg=
                ANGLE_DEG,
        )

        params.update(
            overrides
        )

        try:

            VanEverdingenHurstAquifer(
                **params
            )

        except ValueError:

            print(
                f"✓ {name} rejected."
            )

        else:

            raise AssertionError(
                f"Invalid parameter was "
                f"accepted: {name}"
            )

    print(
        "✓ Parameter validation checks PASSED."
    )

    # ========================================================
    # 3. DIMENSIONLESS TIME
    # ========================================================

    print("\n## Dimensionless Time")

    time_days = np.array(
        [
            0.0,
            1.0,
            10.0,
            100.0,
            1000.0,
        ]
    )

    tD = model.dimensionless_time(
        time_days
    )

    print(
        "Time days:"
    )

    print(
        time_days
    )

    print(
        "Dimensionless time:"
    )

    print(
        tD
    )

    assert np.isclose(
        tD[0],
        0.0,
    )

    assert np.all(
        np.diff(tD) > 0
    )

    assert np.all(
        tD >= 0
    )

    print(
        "✓ Dimensionless-time calculation PASSED."
    )

    # ========================================================
    # 4. DIMENSIONLESS INFLUX FUNCTION
    # ========================================================

    print(
        "\n## Dimensionless Cumulative Influx Q_D"
    )

    test_tD = np.array(
        [
            0.0,
            0.001,
            0.01,
            0.05,
            0.10,
            0.50,
            1.0,
            2.0,
            5.0,
            10.0,
            100.0,
            1000.0,
        ]
    )

    qD = model.dimensionless_influx(
        test_tD
    )

    print(
        "tD:"
    )

    print(
        test_tD
    )

    print(
        "QD:"
    )

    print(
        qD
    )

    # Initial condition
    assert np.isclose(
        qD[0],
        0.0,
    )

    # QD must be non-negative
    assert np.all(
        qD >= 0
    )

    # Cumulative influence function
    # must be non-decreasing.
    assert np.all(
        np.diff(qD) >= -1e-12
    )

    # Early-time asymptotic behavior
    expected_early = (
        2.0
        * np.sqrt(
            0.001 / np.pi
        )
    )

    calculated_early = (
        model.dimensionless_influx(
            0.001
        )
    )

    assert np.isclose(
        calculated_early,
        expected_early,
        rtol=1e-10,
        atol=1e-12,
    )

    print(
        "✓ Q_D non-negativity check PASSED."
    )

    print(
        "✓ Q_D monotonicity check PASSED."
    )

    print(
        "✓ Early-time Q_D behavior PASSED."
    )

    # ========================================================
    # 5. ZERO PRESSURE-DIFFERENCE CHECK
    # ========================================================

    print(
        "\n## Zero Pressure-Difference Check"
    )

    pressure = np.array(
        [
            4000.0,
            4000.0,
            4000.0,
            4000.0,
            4000.0,
        ]
    )

    time = np.array(
        [
            0.0,
            10.0,
            20.0,
            30.0,
            40.0,
        ]
    )

    result = model.calculate_influx(
        pressure_psia=pressure,
        time_days=time,
    )

    max_we = np.max(
        np.abs(
            result[
                "We_rb"
            ]
        )
    )

    max_rate = np.max(
        np.abs(
            result[
                "influx_rate_rb_per_day"
            ]
        )
    )

    print(
        f"Maximum cumulative We : "
        f"{max_we:.6e} rb"
    )

    print(
        f"Maximum influx rate    : "
        f"{max_rate:.6e} rb/day"
    )

    assert max_we < 1e-10

    assert max_rate < 1e-10

    print(
        "✓ Zero pressure-difference "
        "check PASSED."
    )

    # ========================================================
    # 6. SINGLE PRESSURE-DROP CHECK
    # ========================================================

    print(
        "\n## Single Pressure-Drop Check"
    )

    pressure = np.array(
        [
            4000.0,
            3900.0,
            3900.0,
            3900.0,
            3900.0,
        ]
    )

    time = np.array(
        [
            0.0,
            10.0,
            20.0,
            40.0,
            80.0,
        ]
    )

    result = model.calculate_influx(
        pressure_psia=pressure,
        time_days=time,
    )

    print(
        result[
            [
                "time_days",
                "pressure_psia",
                "dimensionless_time",
                "We_rb",
            ]
        ]
    )

    # No influx at the instant of the
    # pressure change.
    assert np.isclose(
        result[
            "We_rb"
        ].iloc[1],
        0.0,
        atol=1e-10,
    )

    # Influx must develop after the
    # pressure disturbance.
    assert (
        result[
            "We_rb"
        ].iloc[-1]
        > 0
    )

    # Cumulative influx must increase.
    assert np.all(
        np.diff(
            result[
                "We_rb"
            ]
        ) >= -1e-10
    )

    print(
        f"Final We : "
        f"{result['We_rb'].iloc[-1]:.6f} rb"
    )

    print(
        "✓ Single pressure-drop "
        "response PASSED."
    )

    # ========================================================
    # 7. PRESSURE-DRAWNDOWN BEHAVIOR
    # ========================================================

    print(
        "\n## Increasing Drawdown Check"
    )

    pressure = np.array(
        [
            4000.0,
            3950.0,
            3850.0,
            3700.0,
            3500.0,
        ]
    )

    time = np.array(
        [
            0.0,
            10.0,
            20.0,
            30.0,
            40.0,
        ]
    )

    result = model.calculate_influx(
        pressure_psia=pressure,
        time_days=time,
    )

    print(
        "Pressure:"
    )

    print(
        pressure
    )

    print(
        "Cumulative We:"
    )

    print(
        result[
            "We_rb"
        ].to_numpy()
    )

    assert (
        result[
            "We_rb"
        ].iloc[-1]
        > result[
            "We_rb"
        ].iloc[1]
    )

    assert np.all(
        np.diff(
            result[
                "We_rb"
            ]
        ) >= -1e-10
    )

    print(
        "✓ Increasing-drawdown "
        "behavior PASSED."
    )

    # ========================================================
    # 8. SUPERPOSITION CHECK
    # ========================================================

    print(
        "\n## Pressure-History Superposition Check"
    )

    pressure = np.array(
        [
            4000.0,
            3900.0,
            3800.0,
            3700.0,
        ]
    )

    time = np.array(
        [
            0.0,
            10.0,
            20.0,
            40.0,
        ]
    )

    result_multi = model.calculate_influx(
        pressure_psia=pressure,
        time_days=time,
    )

    # Compare against a history where
    # only the first pressure drop occurs.
    pressure_single = np.array(
        [
            4000.0,
            3900.0,
            3900.0,
            3900.0,
        ]
    )

    result_single = model.calculate_influx(
        pressure_psia=pressure_single,
        time_days=time,
    )

    final_multi = (
        result_multi[
            "We_rb"
        ].iloc[-1]
    )

    final_single = (
        result_single[
            "We_rb"
        ].iloc[-1]
    )

    print(
        f"Final We - multiple drops : "
        f"{final_multi:.6f} rb"
    )

    print(
        f"Final We - single drop    : "
        f"{final_single:.6f} rb"
    )

    assert (
        final_multi
        > final_single
    )

    print(
        "✓ Pressure-history "
        "superposition check PASSED."
    )

    # ========================================================
    # 9. INPUT VALIDATION
    # ========================================================

    print(
        "\n## Input Validation"
    )

    # Mismatched lengths
    try:

        model.calculate_influx(

            pressure_psia=[
                4000.0,
                3900.0,
            ],

            time_days=[
                0.0,
            ],
        )

    except ValueError:

        print(
            "✓ Mismatched input lengths rejected."
        )

    else:

        raise AssertionError(
            "Mismatched input lengths "
            "were accepted."
        )

    # Non-monotonic time
    try:

        model.calculate_influx(

            pressure_psia=[
                4000.0,
                3900.0,
                3800.0,
            ],

            time_days=[
                0.0,
                20.0,
                10.0,
            ],
        )

    except ValueError:

        print(
            "✓ Non-monotonic time history rejected."
        )

    else:

        raise AssertionError(
            "Non-monotonic time history "
            "was accepted."
        )

    # Negative pressure
    try:

        model.calculate_influx(

            pressure_psia=[
                4000.0,
                -100.0,
            ],

            time_days=[
                0.0,
                10.0,
            ],
        )

    except ValueError:

        print(
            "✓ Negative pressure rejected."
        )

    else:

        raise AssertionError(
            "Negative pressure was accepted."
        )

    # NaN pressure
    try:

        model.calculate_influx(

            pressure_psia=[
                4000.0,
                np.nan,
            ],

            time_days=[
                0.0,
                10.0,
            ],
        )

    except ValueError:

        print(
            "✓ NaN pressure rejected."
        )

    else:

        raise AssertionError(
            "NaN pressure was accepted."
        )

    # Time not starting at zero
    try:

        model.calculate_influx(

            pressure_psia=[
                4000.0,
                3900.0,
            ],

            time_days=[
                5.0,
                10.0,
            ],
        )

    except ValueError:

        print(
            "✓ Non-zero initial time rejected."
        )

    else:

        raise AssertionError(
            "Non-zero initial time "
            "was accepted."
        )

    print(
        "✓ Input validation checks PASSED."
    )

    # ========================================================
    # 10. RESULT STRUCTURE
    # ========================================================

    print(
        "\n## Result Structure Check"
    )

    required_columns = [

        "time_days",

        "pressure_psia",

        "dimensionless_time",

        "pressure_change_psia",

        "influx_rate_rb_per_day",

        "delta_We_rb",

        "We_rb",
    ]

    for column in required_columns:

        assert (
            column in result_multi.columns
        )

    assert len(
        result_multi
    ) == len(
        pressure
    )

    print(
        "✓ Result dataframe structure PASSED."
    )

    # ========================================================
    # 11. NO NEGATIVE INFLUX
    # ========================================================

    print(
        "\n## Physical Validity Check"
    )

    assert np.all(
        result_multi[
            "We_rb"
        ].to_numpy()
        >= -1e-10
    )

    assert np.all(
        result_multi[
            "delta_We_rb"
        ].to_numpy()
        >= -1e-10
    )

    assert np.all(
        result_multi[
            "influx_rate_rb_per_day"
        ].to_numpy()
        >= -1e-10
    )

    print(
        "✓ No-negative-influx check PASSED."
    )

    # ========================================================
    # 12. DISPLAY FINAL RESULT
    # ========================================================

    print("\n")
    print("=" * 70)

    print(
        "VAN EVERDINGEN-HURST RESULT"
    )

    print("=" * 70)

    print(
        result_multi
    )

    print("\n")
    print(
        "Model parameters:"
    )

    for key, value in (
        model.get_parameters().items()
    ):

        print(
            f"{key:40s}: {value}"
        )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n")
    print("=" * 70)

    print(
        "✓ ALL VAN EVERDINGEN-HURST "
        "CHECKS PASSED"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()