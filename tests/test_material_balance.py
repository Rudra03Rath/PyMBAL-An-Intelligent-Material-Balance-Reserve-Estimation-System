import numpy as np
import pandas as pd

from src.core.reservoir import (
    ReservoirModel
)

from src.core.material_balance import (
    MaterialBalanceEngine
)


# ============================================================
# Reservoir configuration
# ============================================================

CONFIG_PATH = (
    "data/reservoir_config/"
    "reservoir_parameters.json"
)


# ============================================================
# Load reservoir state
# ============================================================

reservoir = ReservoirModel(
    CONFIG_PATH
)

state = reservoir.get_initial_state()

reservoir.display_state(
    state
)


# ============================================================
# Synthetic production + PVT history
# ============================================================

df = pd.DataFrame({

    "Pressure_psia": [
        4000,
        3800,
        3600,
        3400,
        3200,
        3000,
        2800,
    ],

    "Np_STB": [
        0,
        50000,
        110000,
        180000,
        260000,
        350000,
        450000,
    ],

    "Gp_Mscf": [
        0,
        60000,
        140000,
        235000,
        345000,
        475000,
        625000,
    ],

    "Wp_STB": [
        0,
        3000,
        7000,
        12000,
        18000,
        25000,
        34000,
    ],

    "Bo_rb_per_STB": [
    state.Boi_rb_per_STB,
    state.Boi_rb_per_STB + 0.015,
    state.Boi_rb_per_STB + 0.030,
    state.Boi_rb_per_STB + 0.045,
    state.Boi_rb_per_STB + 0.060,
    state.Boi_rb_per_STB + 0.075,
    state.Boi_rb_per_STB + 0.090,
    ],

    "Bg_rb_per_scf": [
    state.Bgi_rb_per_scf,
    state.Bgi_rb_per_scf + 0.000020,
    state.Bgi_rb_per_scf + 0.000040,
    state.Bgi_rb_per_scf + 0.000060,
    state.Bgi_rb_per_scf + 0.000080,
    state.Bgi_rb_per_scf + 0.000100,
    state.Bgi_rb_per_scf + 0.000120,
    ],

    "Bw_rb_per_STB": [
    state.Bwi_rb_per_STB,
    state.Bwi_rb_per_STB + 0.001,
    state.Bwi_rb_per_STB + 0.002,
    state.Bwi_rb_per_STB + 0.003,
    state.Bwi_rb_per_STB + 0.004,
    state.Bwi_rb_per_STB + 0.005,
    state.Bwi_rb_per_STB + 0.006,
    ],

    "Rs_scf_per_STB": [
    state.Rsi_scf_per_STB,
    state.Rsi_scf_per_STB - 40,
    state.Rsi_scf_per_STB - 80,
    state.Rsi_scf_per_STB - 120,
    state.Rsi_scf_per_STB - 160,
    state.Rsi_scf_per_STB - 200,
    state.Rsi_scf_per_STB - 240,
    ],
})


# ============================================================
# Create MBE engine
# ============================================================

mbe = MaterialBalanceEngine(
    reservoir_state=state
)


# ============================================================
# Perform analysis
# ============================================================

result, analysis = mbe.analyze(
    df
)


# ============================================================
# Display summary
# ============================================================

mbe.display_summary(
    analysis
)


# ============================================================
# Display calculation table
# ============================================================

print("\n")
print("=" * 65)
print("MATERIAL BALANCE CALCULATION TABLE")
print("=" * 65)

print(
    result[
        [
            "Pressure_psia",

            "F_rb",

            "Eo_rb_per_STB",

            "Eg_rb_per_STB",

            "Efw_rb_per_STB",

            "mEg_rb_per_STB",

            "Etotal_rb_per_STB",

            "We_rb",

            "F_minus_We_rb",

            "N_apparent_STB",

            "MBE_residual_rb",
        ]
    ]
)


# ============================================================
# Basic sanity checks
# ============================================================

print("\n")
print("=" * 65)
print("SANITY CHECKS")
print("=" * 65)


# Initial F should be zero
assert abs(
    result.iloc[0]["F_rb"]
) < 1e-10


# Initial E should be zero
assert abs(
    result.iloc[0]["Etotal_rb_per_STB"]
) < 1e-10


# Initial apparent N is undefined
assert np.isnan(
    result.iloc[0]["N_apparent_STB"]
)


# No gas cap in the current reservoir
if state.gas_cap_ratio == 0:

    assert np.allclose(
        result["mEg_rb_per_STB"],
        0.0
    )


print(
    "✓ Initial F = 0"
)

print(
    "✓ Initial E = 0"
)

print(
    "✓ Initial apparent OOIP undefined as expected"
)

print(
    "✓ Gas-cap contribution correctly handled"
)

print(
    "\n✓ Material Balance Engine test completed."
)