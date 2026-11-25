import numpy as np
from src.pvt.gas import (
    sutton_pseudo_critical,
    pseudo_reduced_pressure,
    pseudo_reduced_temperature,
    hall_yarborough_z,
    gas_fvf,
    gas_pvt
)


pressure = 3000       # psia
temperature = 200     # °F
gas_gravity = 0.80


ppc, tpc = sutton_pseudo_critical(
    gas_gravity
)

ppr = pseudo_reduced_pressure(
    pressure,
    ppc
)

tpr = pseudo_reduced_temperature(
    temperature,
    tpc
)

z = hall_yarborough_z(
    pressure,
    temperature,
    gas_gravity
)

bg = gas_fvf(
    pressure,
    temperature,
    z
)


print("\nGas PVT")
print("-" * 40)

print(f"Ppc           : {ppc:.2f} psia")
print(f"Tpc           : {tpc:.2f} °R")
print(f"Ppr           : {ppr:.4f}")
print(f"Tpr           : {tpr:.4f}")
print(f"Z             : {z:.4f}")
print(f"Bg            : {bg:.7f} rb/scf")


print("\nComplete Gas PVT")
print("-" * 40)

result = gas_pvt(
    pressure,
    temperature,
    gas_gravity
)

for key, value in result.items():

    if np.isscalar(value):
        print(f"{key:20}: {value}")