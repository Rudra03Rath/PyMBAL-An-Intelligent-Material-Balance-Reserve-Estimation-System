from src.pvt.water import (
    mccain_bw,
    mccain_water_viscosity_atm,
    mccain_water_viscosity,
    water_pvt
)


pressure = 3000       # psia
temperature = 200     # °F
salinity = 5.0        # wt%


# ------------------------------------------------------------
# Individual calculations
# ------------------------------------------------------------

bw = mccain_bw(
    pressure,
    temperature
)

mu_w1 = mccain_water_viscosity_atm(
    temperature,
    salinity
)

mu_w = mccain_water_viscosity(
    pressure,
    temperature,
    salinity
)


print("\nWater PVT")
print("-" * 40)

print(f"Pressure       : {pressure:.2f} psia")
print(f"Temperature    : {temperature:.2f} °F")
print(f"Salinity       : {salinity:.2f} wt%")

print(f"\nBw             : {bw:.5f} rb/STB")
print(f"μw @ atm       : {mu_w1:.5f} cP")
print(f"μw @ reservoir : {mu_w:.5f} cP")


# ------------------------------------------------------------
# Complete Water PVT
# ------------------------------------------------------------

result = water_pvt(
    pressure,
    temperature,
    salinity
)

print("\nComplete Water PVT")
print("-" * 40)

for key, value in result.items():
    print(f"{key:20}: {value}")