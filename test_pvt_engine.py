import pandas as pd

from src.pvt import PVTEngine


# ============================================================
# Create sample reservoir pressure history
# ============================================================

df = pd.DataFrame({
    "Pressure_psia": [
        4000,
        3800,
        3600,
        3400,
        3200
    ]
})


# ============================================================
# Test 1: Correlation Mode
# ============================================================

print("\n")
print("=" * 60)
print("TEST 1: CORRELATION MODE")
print("=" * 60)

pvt_engine = PVTEngine(
    mode="correlation",
    temperature=200,
    api=35,
    gas_gravity=0.80,
    salinity=5.0
)

correlation_df = pvt_engine.calculate(df)

print("\nCalculated PVT:")
print(
    correlation_df[
        [
            "Pressure_psia",
            "Rs_scf_per_STB",
            "Bo_rb_per_STB",
            "Bg_rb_per_scf",
            "Bw_rb_per_STB"
        ]
    ]
)


# ============================================================
# Test 2: Laboratory Mode
# ============================================================

print("\n")
print("=" * 60)
print("TEST 2: LABORATORY MODE")
print("=" * 60)


laboratory_df = pd.DataFrame({
    "Pressure_psia": [
        3000,
        3500,
        4000
    ],

    "Bo_rb_per_STB": [
        1.32,
        1.34,
        1.36
    ],

    "Bg_rb_per_scf": [
        0.00085,
        0.00074,
        0.00066
    ],

    "Rs_scf_per_STB": [
        660,
        720,
        780
    ],

    "Bw_rb_per_STB": [
        1.032,
        1.033,
        1.034
    ]
})


lab_engine = PVTEngine(
    mode="laboratory"
)

lab_df = lab_engine.calculate(
    df,
    laboratory_pvt_df=laboratory_df
)

print("\nInterpolated Laboratory PVT:")

print(lab_df)


# ============================================================
# Test 3: Hybrid Mode
# ============================================================

print("\n")
print("=" * 60)
print("TEST 3: HYBRID MODE")
print("=" * 60)


hybrid_df = df.copy()

# Provide only Bo
hybrid_df["Bo_rb_per_STB"] = [
    1.36,
    1.35,
    1.34,
    1.33,
    1.32
]


hybrid_engine = PVTEngine(
    mode="hybrid",
    temperature=200,
    api=35,
    gas_gravity=0.80,
    salinity=5.0
)


hybrid_result = hybrid_engine.calculate(
    hybrid_df
)


print("\nHybrid PVT:")

print(
    hybrid_result[
        [
            "Pressure_psia",
            "Bo_rb_per_STB",
            "Rs_scf_per_STB",
            "Bg_rb_per_scf",
            "Bw_rb_per_STB"
        ]
    ]
)