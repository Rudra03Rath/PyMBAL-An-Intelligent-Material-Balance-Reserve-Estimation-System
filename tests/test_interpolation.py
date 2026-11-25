import pandas as pd

from src.pvt.interpolation import (
    linear_interpolate,
    interpolate_pvt,
    interpolate_pvt_dataframe
)


# ============================================================
# Example laboratory PVT data
# ============================================================

lab_pressure = [
    2000,
    2500,
    3000,
    3500,
    4000
]

bo = [
    1.28,
    1.30,
    1.32,
    1.34,
    1.36
]

bg = [
    0.00120,
    0.00100,
    0.00085,
    0.00074,
    0.00066
]

rs = [
    520,
    590,
    660,
    720,
    780
]

bw = [
    1.030,
    1.031,
    1.032,
    1.033,
    1.034
]


# ============================================================
# Single-property interpolation
# ============================================================

target_pressure = 3250

bo_result = linear_interpolate(
    pressure=target_pressure,
    pvt_pressure=lab_pressure,
    pvt_values=bo
)

print("\nSingle Property")
print("-" * 40)

print(
    f"Bo at {target_pressure} psi = "
    f"{bo_result:.5f} rb/STB"
)


# ============================================================
# Multiple-property interpolation
# ============================================================

pvt_data = {
    "Bo_rb_per_STB": bo,
    "Bg_rb_per_scf": bg,
    "Rs_scf_per_STB": rs,
    "Bw_rb_per_STB": bw
}

result = interpolate_pvt(
    pressure=target_pressure,
    pvt_pressure=lab_pressure,
    pvt_data=pvt_data
)

print("\nMultiple PVT Properties")
print("-" * 40)

for key, value in result.items():

    print(
        f"{key:20}: {value:.6f}"
    )


# ============================================================
# DataFrame interpolation
# ============================================================

laboratory_df = pd.DataFrame({
    "Pressure_psia": lab_pressure,
    "Bo_rb_per_STB": bo,
    "Bg_rb_per_scf": bg,
    "Rs_scf_per_STB": rs,
    "Bw_rb_per_STB": bw
})

target_pressures = [
    2250,
    2750,
    3250,
    3750
]

result_df = interpolate_pvt_dataframe(
    target_pressure=target_pressures,
    laboratory_df=laboratory_df
)

print("\nInterpolated PVT DataFrame")
print("-" * 40)

print(result_df)