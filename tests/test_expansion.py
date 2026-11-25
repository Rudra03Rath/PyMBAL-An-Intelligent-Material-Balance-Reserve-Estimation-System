import numpy as np
import pandas as pd

from src.core.expansion import (
    calculate_oil_expansion,
    calculate_gas_cap_expansion,
    calculate_water_rock_expansion,
    calculate_total_expansion,
)


# ============================================================
# Synthetic PVT / pressure history
# ============================================================

df = pd.DataFrame({

    "Pressure_psia": [
        4000,
        3800,
        3600,
        3400,
        3200,
    ],

    "Bo_rb_per_STB": [
        1.35,
        1.36,
        1.37,
        1.38,
        1.39,
    ],

    "Bg_rb_per_scf": [
        0.00070,
        0.00072,
        0.00074,
        0.00076,
        0.00078,
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
# Initial reservoir conditions
# ============================================================

Pi = 4000.0

Boi = 1.35

Bgi = 0.00070

Rsi = 700.0

Swi = 0.25

cf = 6e-6

cw = 3e-6

# First test: no initial gas cap
m = 0.0


# ============================================================
# Oil expansion
# ============================================================

Eo = calculate_oil_expansion(
    df,
    Boi=Boi,
    Rsi=Rsi
)

print("\nOil Expansion")
print("-" * 60)

print(Eo)


# ============================================================
# Gas-cap expansion
# ============================================================

Eg = calculate_gas_cap_expansion(
    df,
    Boi=Boi,
    Bgi=Bgi
)

print("\nGas-Cap Expansion")
print("-" * 60)

print(Eg)


# ============================================================
# Formation + water expansion
# ============================================================

Efw = calculate_water_rock_expansion(
    df,
    Boi=Boi,
    Pi=Pi,
    Swi=Swi,
    cf=cf,
    cw=cw,
    m=m
)

print("\nFormation + Connate-Water Expansion")
print("-" * 60)

print(Efw)


# ============================================================
# Total expansion
# ============================================================

result = calculate_total_expansion(

    df,

    Boi=Boi,

    Bgi=Bgi,

    Rsi=Rsi,

    Pi=Pi,

    Swi=Swi,

    cf=cf,

    cw=cw,

    m=m,
)


print("\nComplete Expansion Calculation")
print("-" * 60)

print(
    result[
        [
            "Pressure_psia",
            "Eo_rb_per_STB",
            "Eg_rb_per_STB",
            "Efw_rb_per_STB",
            "mEg_rb_per_STB",
            "Etotal_rb_per_STB",
        ]
    ]
)


# ============================================================
# Sanity checks
# ============================================================

print("\nSanity Checks")
print("-" * 60)


# At initial pressure, Eo should be zero.
assert abs(
    result.iloc[0]["Eo_rb_per_STB"]
) < 1e-10


# At initial pressure, Eg should be zero.
assert abs(
    result.iloc[0]["Eg_rb_per_STB"]
) < 1e-10


# At initial pressure, Efw should be zero.
assert abs(
    result.iloc[0]["Efw_rb_per_STB"]
) < 1e-10


# Since m = 0, mEg must be zero.
assert np.allclose(
    result["mEg_rb_per_STB"],
    0.0
)


# Total expansion should equal Eo + mEg + Efw.
calculated_total = (
    result["Eo_rb_per_STB"]
    +
    result["mEg_rb_per_STB"]
    +
    result["Efw_rb_per_STB"]
)

assert np.allclose(
    result["Etotal_rb_per_STB"],
    calculated_total
)


print(
    "✓ Initial-condition checks passed."
)

print(
    "✓ Total expansion identity passed."
)

print(
    "\n✓ Expansion module is working correctly."
)