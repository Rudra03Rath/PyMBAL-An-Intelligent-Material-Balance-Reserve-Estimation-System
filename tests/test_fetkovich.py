"""
PyMBAL - Fetkovich Aquifer Validation

Independent unit tests for the finite-aquifer Fetkovich model.

Tests:
    1. Model construction
    2. Parameter validation
    3. Zero pressure-difference behavior
    4. Constant reservoir-pressure response
    5. Aquifer pressure decline
    6. Influx-rate decline
    7. Monotonic cumulative water influx
    8. Aquifer storage conservation
    9. Variable-pressure history
    10. Invalid input handling
"""


import numpy as np

from src.influx.fetkovich import (
    FetkovichAquifer
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
    Numerical comparison helper.
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
        "PyMBAL - FETKOVICH AQUIFER VALIDATION"
    )
    print("=" * 70)

    # ========================================================
    # TEST PARAMETERS
    # ========================================================

    Pa_i = 4000.0

    Wi = 5_000_000.0

    ct = 20.0e-6

    J = 100.0

    print("\n")
    print("Test Parameters")
    print("-" * 60)

    print(
        f"Initial aquifer pressure : "
        f"{Pa_i:.2f} psia"
    )

    print(
        f"Initial aquifer volume   : "
        f"{Wi:,.0f} rb"
    )

    print(
        f"Total compressibility    : "
        f"{ct:.6e} psi⁻¹"
    )

    print(
        f"Productivity index       : "
        f"{J:.4f} rb/(day·psi)"
    )

    # ========================================================
    # CHECK 1:
    # MODEL CONSTRUCTION
    # ========================================================

    print("\n")
    print("=" * 60)
    print(
        "CHECK 1: MODEL CONSTRUCTION"
    )
    print("=" * 60)

    model = FetkovichAquifer(

        initial_aquifer_pressure_psia=Pa_i,

        initial_water_volume_rb=Wi,

        total_compressibility_psi_inv=ct,

        productivity_index_rb_per_day_psi=J,
    )

    if model.get_name() != (
        "Fetkovich Finite Aquifer"
    ):

        raise AssertionError(
            "Incorrect Fetkovich model name."
        )

    expected_Ca = (
        Wi * ct
    )

    check_close(

        model.storage_capacity_rb_per_psi,

        expected_Ca,

        "Incorrect aquifer storage capacity.",
    )

    print(
        f"Aquifer storage capacity : "
        f"{model.storage_capacity_rb_per_psi:.6f} rb/psi"
    )

    print(
        "✓ Model construction check PASSED."
    )

    # ========================================================
    # CHECK 2:
    # PARAMETER VALIDATION
    # ========================================================

    print("\n")
    print("=" * 60)
    print(
        "CHECK 2: PARAMETER VALIDATION"
    )
    print("=" * 60)

    invalid_cases = [

        (
            dict(
                initial_aquifer_pressure_psia=0.0,
                initial_water_volume_rb=Wi,
                total_compressibility_psi_inv=ct,
                productivity_index_rb_per_day_psi=J,
            ),
            "invalid aquifer pressure",
        ),

        (
            dict(
                initial_aquifer_pressure_psia=Pa_i,
                initial_water_volume_rb=0.0,
                total_compressibility_psi_inv=ct,
                productivity_index_rb_per_day_psi=J,
            ),
            "invalid aquifer volume",
        ),

        (
            dict(
                initial_aquifer_pressure_psia=Pa_i,
                initial_water_volume_rb=Wi,
                total_compressibility_psi_inv=0.0,
                productivity_index_rb_per_day_psi=J,
            ),
            "invalid compressibility",
        ),

        (
            dict(
                initial_aquifer_pressure_psia=Pa_i,
                initial_water_volume_rb=Wi,
                total_compressibility_psi_inv=ct,
                productivity_index_rb_per_day_psi=0.0,
            ),
            "invalid productivity index",
        ),
    ]

    for parameters, description in invalid_cases:

        try:

            FetkovichAquifer(
                **parameters
            )

        except ValueError:

            print(
                f"✓ {description} rejected."
            )

        else:

            raise AssertionError(
                f"Model accepted {description}."
            )

    print(
        "✓ Parameter validation check PASSED."
    )

    # ========================================================
    # CHECK 3:
    # ZERO PRESSURE DIFFERENCE
    # ========================================================

    print("\n")
    print("=" * 60)
    print(
        "CHECK 3: ZERO PRESSURE-DIFFERENCE"
    )
    print("=" * 60)

    time = np.array(
        [0, 10, 20, 30, 40],
        dtype=float,
    )

    pressure = np.full(
        len(time),
        Pa_i,
    )

    result = model.calculate_influx(

        pressure_psia=pressure,

        time_days=time,
    )

    max_rate = np.max(
        np.abs(
            result[
                "influx_rate_rb_per_day"
            ]
        )
    )

    max_We = np.max(
        np.abs(
            result[
                "We_rb"
            ]
        )
    )

    print(
        f"Maximum influx rate : "
        f"{max_rate:.6e} rb/day"
    )

    print(
        f"Maximum cumulative We : "
        f"{max_We:.6e} rb"
    )

    check_close(

        result[
            "influx_rate_rb_per_day"
        ].to_numpy(),

        np.zeros(
            len(time)
        ),

        "Zero pressure difference "
        "should produce zero influx rate.",
    )

    check_close(

        result[
            "We_rb"
        ].to_numpy(),

        np.zeros(
            len(time)
        ),

        "Zero pressure difference "
        "should produce zero cumulative influx.",
    )

    print(
        "✓ Zero pressure-difference check PASSED."
    )

    # ========================================================
    # CHECK 4:
    # CONSTANT RESERVOIR PRESSURE BELOW AQUIFER
    # ========================================================

    print("\n")
    print("=" * 60)
    print(
        "CHECK 4: CONSTANT RESERVOIR PRESSURE"
    )
    print("=" * 60)

    reservoir_pressure = 3500.0

    time = np.array(
        [0, 1, 2, 3, 4, 5],
        dtype=float,
    )

    pressure = np.full(
        len(time),
        reservoir_pressure,
    )

    result = model.calculate_influx(

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

    print(
        "Influx rates:"
    )

    print(
        rates
    )

    print(
        "Cumulative We:"
    )

    print(
        We
    )

    if rates[1] <= 0:

        raise AssertionError(
            "Positive pressure difference "
            "should produce positive influx."
        )

    if We[-1] <= 0:

        raise AssertionError(
            "Cumulative water influx "
            "should be positive."
        )

    print(
        "✓ Constant-pressure influx check PASSED."
    )

    # ========================================================
    # CHECK 5:
    # AQUIFER PRESSURE MUST DECLINE
    # ========================================================

    print("\n")
    print("=" * 60)
    print(
        "CHECK 5: AQUIFER PRESSURE DECLINE"
    )
    print("=" * 60)

    aquifer_pressure = (
        result[
            "aquifer_pressure_psia"
        ].to_numpy()
    )

    print(
        "Aquifer pressures:"
    )

    print(
        aquifer_pressure
    )

    if not np.all(
        np.diff(
            aquifer_pressure
        ) <= 1e-10
    ):

        raise AssertionError(
            "Aquifer pressure should not "
            "increase while supplying water."
        )

    if not (
        aquifer_pressure[-1]
        <
        aquifer_pressure[0]
    ):

        raise AssertionError(
            "Aquifer pressure did not decline."
        )

    print(
        "✓ Aquifer pressure decline check PASSED."
    )

    # ========================================================
    # CHECK 6:
    # INFLUX RATE DECLINES
    # ========================================================

    print("\n")
    print("=" * 60)
    print(
        "CHECK 6: INFLUX-RATE DECLINE"
    )
    print("=" * 60)

    active_rates = rates[1:]

    print(
        "Active influx rates:"
    )

    print(
        active_rates
    )

    if not np.all(
        np.diff(
            active_rates
        ) <= 1e-10
    ):

        raise AssertionError(
            "Fetkovich influx rate should "
            "decline as aquifer pressure declines "
            "under constant reservoir pressure."
        )

    print(
        "✓ Influx-rate decline check PASSED."
    )

    # ========================================================
    # CHECK 7:
    # CUMULATIVE INFLUX MONOTONICITY
    # ========================================================

    print("\n")
    print("=" * 60)
    print(
        "CHECK 7: CUMULATIVE INFLUX MONOTONICITY"
    )
    print("=" * 60)

    if not np.all(
        We >= -1e-10
    ):

        raise AssertionError(
            "Negative cumulative water influx "
            "detected."
        )

    if not np.all(
        np.diff(
            We
        ) >= -1e-10
    ):

        raise AssertionError(
            "Cumulative water influx "
            "is not monotonic."
        )

    print(
        "✓ Cumulative influx monotonicity "
        "check PASSED."
    )

    # ========================================================
    # CHECK 8:
    # AQUifer STORAGE CONSERVATION
    # ========================================================

    print("\n")
    print("=" * 60)
    print(
        "CHECK 8: AQUIFER STORAGE CONSERVATION"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # From:
    #
    # We = Ca * (Pa_i - Pa)
    #
    # Therefore:
    #
    # Pa = Pa_i - We/Ca
    # --------------------------------------------------------

    expected_aquifer_pressure = (

        Pa_i

        -

        We
        /
        model.storage_capacity_rb_per_psi
    )

    actual_aquifer_pressure = (
        result[
            "aquifer_pressure_psia"
        ].to_numpy()
    )

    max_pressure_error = np.max(
        np.abs(
            actual_aquifer_pressure
            -
            expected_aquifer_pressure
        )
    )

    print(
        f"Maximum aquifer storage error : "
        f"{max_pressure_error:.6e} psi"
    )

    check_close(

        actual_aquifer_pressure,

        expected_aquifer_pressure,

        "Aquifer storage conservation failed.",
    )

    print(
        "✓ Aquifer storage conservation "
        "check PASSED."
    )

    # ========================================================
    # CHECK 9:
    # VARIABLE PRESSURE HISTORY
    # ========================================================

    print("\n")
    print("=" * 60)
    print(
        "CHECK 9: VARIABLE PRESSURE HISTORY"
    )
    print("=" * 60)

    time = np.array(
        [0, 5, 10, 20, 30, 45],
        dtype=float,
    )

    pressure = np.array(
        [
            4000.0,
            3900.0,
            3700.0,
            3400.0,
            3200.0,
            3000.0,
        ],
        dtype=float,
    )

    variable_result = (
        model.calculate_influx(

            pressure_psia=pressure,

            time_days=time,
        )
    )

    variable_We = (
        variable_result[
            "We_rb"
        ].to_numpy()
    )

    variable_rates = (
        variable_result[
            "influx_rate_rb_per_day"
        ].to_numpy()
    )

    variable_Pa = (
        variable_result[
            "aquifer_pressure_psia"
        ].to_numpy()
    )

    print(
        variable_result
    )

    if not np.all(
        variable_We >= -1e-10
    ):

        raise AssertionError(
            "Variable-pressure case produced "
            "negative cumulative influx."
        )

    if not np.all(
        np.diff(
            variable_We
        ) >= -1e-10
    ):

        raise AssertionError(
            "Variable-pressure cumulative "
            "influx is not monotonic."
        )

    if not np.all(
        variable_rates >= -1e-10
    ):

        raise AssertionError(
            "Variable-pressure case produced "
            "negative influx rates."
        )

    if not np.all(
        np.diff(
            variable_Pa
        ) <= 1e-10
    ):

        raise AssertionError(
            "Aquifer pressure increased "
            "during water withdrawal."
        )

    print(
        "✓ Variable-pressure history check PASSED."
    )

    # ========================================================
    # CHECK 10:
    # INPUT VALIDATION
    # ========================================================

    print("\n")
    print("=" * 60)
    print(
        "CHECK 10: INPUT VALIDATION"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Mismatched lengths
    # --------------------------------------------------------

    try:

        model.calculate_influx(

            pressure_psia=[
                4000,
                3900,
            ],

            time_days=[
                0,
            ],
        )

    except ValueError:

        print(
            "✓ Mismatched input lengths rejected."
        )

    else:

        raise AssertionError(
            "Mismatched input lengths were accepted."
        )

    # --------------------------------------------------------
    # Non-monotonic time
    # --------------------------------------------------------

    try:

        model.calculate_influx(

            pressure_psia=[
                4000,
                3900,
                3800,
            ],

            time_days=[
                0,
                10,
                5,
            ],
        )

    except ValueError:

        print(
            "✓ Non-monotonic time history rejected."
        )

    else:

        raise AssertionError(
            "Non-monotonic time history was accepted."
        )

    # --------------------------------------------------------
    # Negative pressure
    # --------------------------------------------------------

    try:

        model.calculate_influx(

            pressure_psia=[
                4000,
                -100,
            ],

            time_days=[
                0,
                10,
            ],
        )

    except ValueError:

        print(
            "✓ Invalid reservoir pressure rejected."
        )

    else:

        raise AssertionError(
            "Invalid reservoir pressure was accepted."
        )

    # --------------------------------------------------------
    # NaN pressure
    # --------------------------------------------------------

    try:

        model.calculate_influx(

            pressure_psia=[
                4000,
                np.nan,
            ],

            time_days=[
                0,
                10,
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

    print(
        "✓ Input validation checks PASSED."
    )

    # ========================================================
    # DISPLAY PARAMETERS
    # ========================================================

    print("\n")
    print("=" * 60)
    print(
        "MODEL PARAMETERS"
    )
    print("=" * 60)

    parameters = (
        model.get_parameters()
    )

    for key, value in parameters.items():

        print(
            f"{key:40s}: {value}"
        )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "✓ ALL FETKOVICH AQUIFER CHECKS PASSED"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()