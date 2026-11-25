import numpy as np
import pandas as pd

from src.core.gas_cap import (
    GasCapModel,
    calculate_gas_cap_contribution,
    calculate_gas_cap_expansion,
    calculate_gas_cap_terms,
)


BOI = 1.42
BGI = 0.00082


def make_dataset():

    return pd.DataFrame(
        {
            "Pressure_psia": [
                4000.0,
                3900.0,
                3800.0,
                3700.0,
            ],
            "Bg_rb_per_scf": [
                0.000820,
                0.000850,
                0.000880,
                0.000910,
            ],
        }
    )


def expect_raises(
    func,
    exception_type,
):

    try:
        func()

    except exception_type:
        return

    raise AssertionError(
        f"Expected "
        f"{exception_type.__name__}."
    )


def test_basic_expansion():

    df = make_dataset()

    result = calculate_gas_cap_expansion(
        df,
        Boi=BOI,
        Bgi=BGI,
    )

    expected = (
        BOI
        *
        (
            df["Bg_rb_per_scf"]
            / BGI
            - 1.0
        )
    )

    assert result.name == (
        "Eg_rb_per_STB"
    )

    assert np.allclose(
        result,
        expected,
    )

    # At initial pressure:
    # Bg = Bgi -> Eg = 0
    assert np.isclose(
        result.iloc[0],
        0.0,
    )

    print(
        "✓ Basic gas-cap expansion "
        "calculation PASSED."
    )


def test_known_value():

    df = pd.DataFrame(
        {
            "Bg_rb_per_scf": [
                0.00100
            ]
        }
    )

    result = calculate_gas_cap_expansion(
        df,
        Boi=1.50,
        Bgi=0.00075,
    )

    expected = (
        1.50
        *
        (
            0.00100
            / 0.00075
            - 1.0
        )
    )

    assert np.isclose(
        result.iloc[0],
        expected,
    )

    print(
        "✓ Known-value calculation PASSED."
    )


def test_zero_gas_cap():

    df = make_dataset()

    result = (
        calculate_gas_cap_contribution(
            df,
            Boi=BOI,
            Bgi=BGI,
            m=0.0,
        )
    )

    assert np.allclose(
        result,
        0.0,
    )

    print(
        "✓ m = 0 gas-cap case PASSED."
    )


def test_m_proportionality():

    df = make_dataset()

    Eg = calculate_gas_cap_expansion(
        df,
        Boi=BOI,
        Bgi=BGI,
    )

    m1 = calculate_gas_cap_contribution(
        df,
        Boi=BOI,
        Bgi=BGI,
        m=1.0,
    )

    m2 = calculate_gas_cap_contribution(
        df,
        Boi=BOI,
        Bgi=BGI,
        m=2.0,
    )

    assert np.allclose(
        m1,
        Eg,
    )

    assert np.allclose(
        m2,
        2.0 * Eg,
    )

    print(
        "✓ Gas-cap ratio "
        "proportionality PASSED."
    )


def test_dataframe_terms():

    df = make_dataset()

    result = calculate_gas_cap_terms(
        df,
        Boi=BOI,
        Bgi=BGI,
        m=1.5,
    )

    assert (
        "Eg_rb_per_STB"
        in result.columns
    )

    assert (
        "mEg_rb_per_STB"
        in result.columns
    )

    assert np.allclose(
        result["mEg_rb_per_STB"],
        1.5
        *
        result["Eg_rb_per_STB"],
    )

    # Original data must remain.
    assert (
        "Pressure_psia"
        in result.columns
    )

    assert (
        "Bg_rb_per_scf"
        in result.columns
    )

    print(
        "✓ Gas-cap dataframe "
        "output PASSED."
    )


def test_model_class():

    df = make_dataset()

    model = GasCapModel(
        Boi=BOI,
        Bgi=BGI,
        m=1.25,
    )

    result = model.calculate(
        df
    )

    direct = calculate_gas_cap_terms(
        df,
        Boi=BOI,
        Bgi=BGI,
        m=1.25,
    )

    assert np.allclose(
        result["Eg_rb_per_STB"],
        direct["Eg_rb_per_STB"],
    )

    assert np.allclose(
        result["mEg_rb_per_STB"],
        direct["mEg_rb_per_STB"],
    )

    print(
        "✓ GasCapModel wrapper PASSED."
    )


def test_monotonicity():

    df = make_dataset()

    result = calculate_gas_cap_expansion(
        df,
        Boi=BOI,
        Bgi=BGI,
    )

    assert np.all(
        np.diff(result)
        >= 0.0
    )

    print(
        "✓ Gas-cap expansion "
        "monotonicity PASSED."
    )


def test_invalid_inputs():

    df = make_dataset()

    expect_raises(
        lambda:
        calculate_gas_cap_expansion(
            df,
            Boi=0.0,
            Bgi=BGI,
        ),
        ValueError,
    )

    expect_raises(
        lambda:
        calculate_gas_cap_expansion(
            df,
            Boi=BOI,
            Bgi=0.0,
        ),
        ValueError,
    )

    expect_raises(
        lambda:
        calculate_gas_cap_contribution(
            df,
            Boi=BOI,
            Bgi=BGI,
            m=-0.1,
        ),
        ValueError,
    )

    bad_bg = df.copy()

    bad_bg.loc[
        1,
        "Bg_rb_per_scf"
    ] = 0.0

    expect_raises(
        lambda:
        calculate_gas_cap_expansion(
            bad_bg,
            Boi=BOI,
            Bgi=BGI,
        ),
        ValueError,
    )

    missing_bg = df.drop(
        columns=[
            "Bg_rb_per_scf"
        ]
    )

    expect_raises(
        lambda:
        calculate_gas_cap_expansion(
            missing_bg,
            Boi=BOI,
            Bgi=BGI,
        ),
        ValueError,
    )

    print(
        "✓ Invalid-input "
        "validation PASSED."
    )


def test_initial_condition():

    df = pd.DataFrame(
        {
            "Bg_rb_per_scf": [
                BGI,
                BGI * 1.10,
                BGI * 1.20,
            ]
        }
    )

    result = calculate_gas_cap_expansion(
        df,
        Boi=BOI,
        Bgi=BGI,
    )

    assert np.isclose(
        result.iloc[0],
        0.0,
    )

    assert np.isclose(
        result.iloc[1],
        BOI * 0.10,
    )

    assert np.isclose(
        result.iloc[2],
        BOI * 0.20,
    )

    print(
        "✓ Initial-condition "
        "consistency PASSED."
    )


def main():

    print("=" * 70)
    print(
        "PyMBAL - GAS CAP "
        "EXPANSION VALIDATION"
    )
    print("=" * 70)

    print()
    print("FORMULATION")
    print("-" * 70)
    print(
        "Eg  = Boi * (Bg/Bgi - 1)"
    )
    print(
        "mEg = m * Eg"
    )

    print()
    print("=" * 70)
    print(
        "GAS-CAP EXPANSION TESTS"
    )
    print("=" * 70)

    test_basic_expansion()
    test_known_value()
    test_zero_gas_cap()
    test_m_proportionality()
    test_dataframe_terms()
    test_model_class()
    test_monotonicity()
    test_invalid_inputs()
    test_initial_condition()

    print()
    print("=" * 70)
    print(
        "✓ ALL GAS-CAP EXPANSION "
        "CHECKS PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()