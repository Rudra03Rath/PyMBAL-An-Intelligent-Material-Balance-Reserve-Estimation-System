from src.pvt.oil import (
    standing_rs,
    standing_bubble_point,
    standing_bo,
    standing_pvt
)


# Example reservoir fluid
pressure = 3000       # psia
temperature = 200     # °F
api = 35              # API
gas_gravity = 0.80


# Individual calculations
rs = standing_rs(
    pressure,
    temperature,
    api,
    gas_gravity
)

pb = standing_bubble_point(
    rs,
    temperature,
    api,
    gas_gravity
)

bo = standing_bo(
    rs,
    temperature,
    api,
    gas_gravity
)


print("\nStanding Oil PVT")
print("-" * 40)

print(f"Pressure      : {pressure:.2f} psia")
print(f"Temperature   : {temperature:.2f} °F")
print(f"API Gravity   : {api:.2f}")
print(f"Gas Gravity   : {gas_gravity:.2f}")

print(f"\nRs            : {rs:.2f} scf/STB")
print(f"Bubble Point  : {pb:.2f} psia")
print(f"Bo            : {bo:.4f} rb/STB")


# Complete package
result = standing_pvt(
    pressure,
    temperature,
    api,
    gas_gravity
)

print("\nComplete PVT Dictionary")
print("-" * 40)

for key, value in result.items():
    print(f"{key:20}: {value}")