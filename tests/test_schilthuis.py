"""
PyMBAL - Schilthuis Aquifer Validation

Validates the Schilthuis steady-state water-influx model
independently of the material-balance engine.

Model:

    q_w = C (Pe - P)

and:

    We(t) = integral[q_w dt]

For discrete data:

    dWe = q_w * dt

    We = cumulative sum(dWe)

The tests verify:

    1. Model construction
    2. Zero pressure difference -> zero influx
    3. Correct constant pressure-difference calculation
    4. Correct cumulative integration
    5. Increasing influx with increasing pressure drawdown
    6. No negative influx
    7. Monotonic cumulative influx
    8. Input validation
"""


import numpy as np

from src.influx.schilthuis import (
    SchilthuisAquifer
)


def check_close(
    actual,
    expected,
    message,
    rtol=1e-10,
    atol=1e-10,
):
    """
    Compare two numerical values/arrays.
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


def main():

    print("=" * 70)
    print(
        "PyMBAL - SCHILTHUIS AQUIFER VALIDATION"
    )
    print("=" * 70)

    # ========================================================
    # MODEL PARAMETERS
    # ========================================================

    Pe = 4000.0

    C = 2.0

    print("\n")
    print(
        "Test Parameters"
    )
    print("-" * 50)

    print(
        f"Aquifer pressure Pe : "
        f"{Pe:.2f} psia"
    )

    print(
        f"Influx constant C   : "
        f"{C:.6f} rb/(day·psi)"
    )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    aquifer = SchilthuisAquifer(

        aquifer_pressure_psia=Pe,

        influx_constant_rb_per_day_psi=C,
    )

    print("\n")
    print(
        f"Model: {aquifer.get_name()}"
    )

    # ========================================================
    # CHECK 1: MODEL CONSTRUCTION
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 1: MODEL CONSTRUCTION"
    )
    print("=" * 70)

    if aquifer.get_name() != (
        "Schilthuis Steady-State"
    ):

        raise AssertionError(
            "Incorrect Schilthuis model name."
        )

    if aquifer.aquifer_pressure_psia != Pe:

        raise AssertionError(
            "Aquifer pressure was not stored correctly."
        )

    if (
        aquifer.influx_constant_rb_per_day_psi
        != C
    ):

        raise AssertionError(
            "Influx constant was not stored correctly."
        )

    print(
        "✓ Model construction check PASSED."
    )

    # ========================================================
    # CHECK 2: ZERO PRESSURE DIFFERENCE
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 2: ZERO PRESSURE DIFFERENCE"
    )
    print("=" * 70)

    time = np.array(
        [0.0, 10.0, 20.0, 30.0]
    )

    pressure = np.array(
        [4000.0, 4000.0, 4000.0, 4000.0]
    )

    result = aquifer.calculate_influx(
        pressure_psia=pressure,
        time_days=time,
    )

    expected_rate = np.zeros(
        len(time)
    )

    expected_We = np.zeros(
        len(time)
    )

    check_close(
        result[
            "influx_rate_rb_per_day"
        ].to_numpy(),

        expected_rate,

        "Zero pressure difference produced "
        "non-zero influx rate.",
    )

    check_close(
        result[
            "We_rb"
        ].to_numpy(),

        expected_We,

        "Zero pressure difference produced "
        "non-zero cumulative influx.",
    )

    print(
        "Maximum influx rate : "
        f"{np.max(np.abs(result['influx_rate_rb_per_day'])):.6e} rb/day"
    )

    print(
        "Maximum cumulative We : "
        f"{np.max(np.abs(result['We_rb'])):.6e} rb"
    )

    print(
        "✓ Zero pressure-difference check PASSED."
    )

    # ========================================================
    # CHECK 3: CONSTANT PRESSURE DIFFERENCE
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 3: CONSTANT PRESSURE DIFFERENCE"
    )
    print("=" * 70)

    pressure = np.array(
        [3900.0, 3900.0, 3900.0, 3900.0]
    )

    time = np.array(
        [0.0, 10.0, 20.0, 30.0]
    )

    result = aquifer.calculate_influx(
        pressure_psia=pressure,
        time_days=time,
    )

    # --------------------------------------------------------
    # Pressure difference:
    #
    #     Pe - P = 4000 - 3900
    #             = 100 psi
    #
    # q = C * 100
    #   = 2 * 100
    #   = 200 rb/day
    # --------------------------------------------------------

    expected_rate = np.array(
        [200.0, 200.0, 200.0, 200.0]
    )

    # --------------------------------------------------------
    # First point has dt = 0.
    #
    # Therefore:
    #
    # We = [0, 2000, 4000, 6000]
    # --------------------------------------------------------

    expected_We = np.array(
        [
            0.0,
            2000.0,
            4000.0,
            6000.0,
        ]
    )

    check_close(
        result[
            "influx_rate_rb_per_day"
        ].to_numpy(),

        expected_rate,

        "Constant pressure-difference influx rate "
        "is incorrect.",
    )

    check_close(
        result[
            "We_rb"
        ].to_numpy(),

        expected_We,

        "Cumulative Schilthuis influx is incorrect.",
    )

    print(
        "Expected influx rate : 200 rb/day"
    )

    print(
        "Calculated influx rate : "
        f"{result['influx_rate_rb_per_day'].iloc[-1]:.2f} rb/day"
    )

    print(
        "Expected final We : 6000 rb"
    )

    print(
        "Calculated final We : "
        f"{result['We_rb'].iloc[-1]:.2f} rb"
    )

    print(
        "✓ Constant pressure-difference check PASSED."
    )

    # ========================================================
    # CHECK 4: VARIABLE PRESSURE HISTORY
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 4: VARIABLE PRESSURE HISTORY"
    )
    print("=" * 70)

    time = np.array(
        [0.0, 10.0, 20.0, 30.0]
    )

    pressure = np.array(
        [
            4000.0,
            3950.0,
            3900.0,
            3850.0,
        ]
    )

    result = aquifer.calculate_influx(
        pressure_psia=pressure,
        time_days=time,
    )

    # --------------------------------------------------------
    # Expected rates:
    #
    # P = 4000:
    # q = 2*(4000-4000) = 0
    #
    # P = 3950:
    # q = 2*50 = 100
    #
    # P = 3900:
    # q = 2*100 = 200
    #
    # P = 3850:
    # q = 2*150 = 300
    # --------------------------------------------------------

    expected_rate = np.array(
        [
            0.0,
            100.0,
            200.0,
            300.0,
        ]
    )

    # --------------------------------------------------------
    # Incremental influx:
    #
    # Point 0:
    # dt = 0
    # dWe = 0
    #
    # Point 1:
    # dWe = 100 * 10 = 1000
    #
    # Point 2:
    # dWe = 200 * 10 = 2000
    #
    # Point 3:
    # dWe = 300 * 10 = 3000
    #
    # Therefore:
    #
    # We = [0, 1000, 3000, 6000]
    # --------------------------------------------------------

    expected_delta_We = np.array(
        [
            0.0,
            1000.0,
            2000.0,
            3000.0,
        ]
    )

    expected_We = np.array(
        [
            0.0,
            1000.0,
            3000.0,
            6000.0,
        ]
    )

    check_close(
        result[
            "influx_rate_rb_per_day"
        ].to_numpy(),

        expected_rate,

        "Variable-pressure influx rate is incorrect.",
    )

    check_close(
        result[
            "delta_We_rb"
        ].to_numpy(),

        expected_delta_We,

        "Incremental water influx is incorrect.",
    )

    check_close(
        result[
            "We_rb"
        ].to_numpy(),

        expected_We,

        "Cumulative water influx is incorrect.",
    )

    print(
        "Expected final We : 6000 rb"
    )

    print(
        "Calculated final We : "
        f"{result['We_rb'].iloc[-1]:.2f} rb"
    )

    print(
        "✓ Variable-pressure integration check PASSED."
    )

    # ========================================================
    # CHECK 5: INCREASING DRAWDOWN
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 5: INCREASING DRAWDOWN"
    )
    print("=" * 70)

    pressure = np.array(
        [
            4000.0,
            3950.0,
            3900.0,
            3850.0,
        ]
    )

    time = np.array(
        [
            0.0,
            10.0,
            20.0,
            30.0,
        ]
    )

    result = aquifer.calculate_influx(
        pressure_psia=pressure,
        time_days=time,
    )

    rates = (
        result[
            "influx_rate_rb_per_day"
        ].to_numpy()
    )

    # --------------------------------------------------------
    # The pressure drawdown increases:
    #
    # 0 < 50 < 100 < 150 psi
    #
    # Therefore the influx rate must increase:
    #
    # 0 < 100 < 200 < 300 rb/day
    # --------------------------------------------------------

    if not np.all(
        np.diff(rates) >= 0
    ):

        raise AssertionError(
            "Influx rate did not increase with "
            "increasing pressure drawdown."
        )

    print(
        "Influx rates:"
    )

    print(
        rates
    )

    print(
        "✓ Increasing-drawdown check PASSED."
    )

    # ========================================================
    # CHECK 6: CUMULATIVE INFLUX MONOTONICITY
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 6: CUMULATIVE INFLUX MONOTONICITY"
    )
    print("=" * 70)

    We = (
        result[
            "We_rb"
        ].to_numpy()
    )

    if not np.all(
        np.diff(We) >= -1e-12
    ):

        raise AssertionError(
            "Cumulative water influx decreased "
            "during the simulation."
        )

    print(
        "Cumulative We:"
    )

    print(
        We
    )

    print(
        "✓ Cumulative influx monotonicity check PASSED."
    )

    # ========================================================
    # CHECK 7: NO NEGATIVE INFLUX
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 7: NO NEGATIVE INFLUX"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Reservoir pressure is above aquifer pressure.
    #
    # Therefore Pe - P < 0.
    #
    # The model must clamp the influx rate to zero.
    # --------------------------------------------------------

    pressure = np.array(
        [
            4100.0,
            4200.0,
            4300.0,
        ]
    )

    time = np.array(
        [
            0.0,
            10.0,
            20.0,
        ]
    )

    result = aquifer.calculate_influx(
        pressure_psia=pressure,
        time_days=time,
    )

    rates = (
        result[
            "influx_rate_rb_per_day"
        ].to_numpy()
    )

    We = (
        result[
            "We_rb"
        ].to_numpy()
    )

    if not np.all(
        rates >= 0
    ):

        raise AssertionError(
            "Negative water influx rate was generated."
        )

    if not np.all(
        We >= 0
    ):

        raise AssertionError(
            "Negative cumulative water influx was generated."
        )

    check_close(
        rates,
        np.zeros(3),
        "Influx should be zero when P > Pe.",
    )

    check_close(
        We,
        np.zeros(3),
        "Cumulative influx should remain zero "
        "when P > Pe.",
    )

    print(
        "✓ No-negative-influx check PASSED."
    )

    # ========================================================
    # CHECK 8: INPUT VALIDATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CHECK 8: INPUT VALIDATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Invalid aquifer pressure
    # --------------------------------------------------------

    try:

        SchilthuisAquifer(
            aquifer_pressure_psia=0.0,
            influx_constant_rb_per_day_psi=C,
        )

        raise AssertionError(
            "Zero aquifer pressure was accepted."
        )

    except ValueError:

        print(
            "✓ Invalid aquifer pressure rejected."
        )

    # --------------------------------------------------------
    # Invalid influx constant
    # --------------------------------------------------------

    try:

        SchilthuisAquifer(
            aquifer_pressure_psia=Pe,
            influx_constant_rb_per_day_psi=0.0,
        )

        raise AssertionError(
            "Zero influx constant was accepted."
        )

    except ValueError:

        print(
            "✓ Invalid influx constant rejected."
        )

    # --------------------------------------------------------
    # Mismatched pressure/time lengths
    # --------------------------------------------------------

    try:

        aquifer.calculate_influx(

            pressure_psia=[
                4000.0,
                3900.0,
            ],

            time_days=[
                0.0,
                10.0,
                20.0,
            ],
        )

        raise AssertionError(
            "Mismatched pressure/time lengths "
            "were accepted."
        )

    except ValueError:

        print(
            "✓ Mismatched input lengths rejected."
        )

    # --------------------------------------------------------
    # Non-monotonic time
    # --------------------------------------------------------

    try:

        aquifer.calculate_influx(

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

        raise AssertionError(
            "Non-monotonic time history was accepted."
        )

    except ValueError:

        print(
            "✓ Non-monotonic time history rejected."
        )

    print(
        "✓ Input validation checks PASSED."
    )

    # ========================================================
    # DISPLAY FINAL RESULT
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "SAMPLE SCHILTHUIS RESULT"
    )
    print("=" * 70)

    print(
        result
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "✓ ALL SCHILTHUIS AQUIFER CHECKS PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":

    main()