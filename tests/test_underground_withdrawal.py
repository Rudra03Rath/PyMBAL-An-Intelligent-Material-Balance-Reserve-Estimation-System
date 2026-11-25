import pandas as pd

from src.core.underground_withdrawal import (
    calculate_producing_gor,
    calculate_oil_withdrawal,
    calculate_gas_withdrawal,
    calculate_water_withdrawal,
    calculate_underground_withdrawal,
    calculate_underground_withdrawal_direct,
)


# ============================================================
# Synthetic test data
# ============================================================

df = pd.DataFrame({

    "Np_STB": [
        0,
        10000,
        25000,
        45000,
        70000,
    ],

    "Gp_Mscf": [
        0,
        7000,
        18000,
        34000,
        55000,
    ],

    "Wp_STB": [
        0,
        1000,
        2500,
        4500,
        7000,
    ],

    "Bo_rb_per_STB": [
        1.35,
        1.34,
        1.33,
        1.32,
        1.31,
    ],

    "Bg_rb_per_scf": [
        0.00070,
        0.00072,
        0.00075,
        0.00078,
        0.00081,
    ],

    "Bw_rb_per_STB": [
        1.02,
        1.021,
        1.022,
        1.023,
        1.024,
    ],

    "Rs_scf_per_STB": [
        700,
        680,
        660,
        640,
        620,
    ],
})


# ============================================================
# Producing GOR
# ============================================================

rp = calculate_producing_gor(
    df
)

print("\nProducing GOR")
print("-" * 50)

print(rp)


# ============================================================
# Individual withdrawal terms
# ============================================================

Fo = calculate_oil_withdrawal(
    df
)

Fg = calculate_gas_withdrawal(
    df
)

Fw = calculate_water_withdrawal(
    df
)

print("\nIndividual Withdrawal Terms")
print("-" * 50)

print(
    pd.DataFrame({
        "Fo_rb": Fo,
        "Fg_rb": Fg,
        "Fw_rb": Fw,
    })
)


# ============================================================
# Total underground withdrawal
# ============================================================

result = calculate_underground_withdrawal(
    df
)

print("\nComplete Underground Withdrawal")
print("-" * 50)

print(
    result[
        [
            "Np_STB",
            "Gp_Mscf",
            "Wp_STB",
            "Rp_scf_per_STB",
            "Fo_rb",
            "Fg_rb",
            "Fw_rb",
            "F_rb",
        ]
    ]
)


# ============================================================
# Independent formula verification
# ============================================================

F_direct = (
    calculate_underground_withdrawal_direct(
        df
    )
)

difference = (
    result["F_rb"]
    - F_direct
).abs()

print("\nFormula Verification")
print("-" * 50)

print(
    f"Maximum difference = "
    f"{difference.max():.12e} rb"
)


if difference.max() < 1e-6:

    print(
        "✓ Decomposed and direct "
        "formulations agree."
    )

else:

    print(
        "✗ Verification failed."
    )