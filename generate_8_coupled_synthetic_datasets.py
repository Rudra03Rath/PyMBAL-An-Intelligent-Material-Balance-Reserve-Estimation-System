"""
PyMBAL - Generate the 8 coupled gas-cap / aquifer synthetic datasets.

Run this script from the PyMBAL project root:

    python generate_8_coupled_synthetic_datasets.py

It uses the project's existing SyntheticReservoir implementation so the
datasets remain consistent with the equations and aquifer models already
validated in PyMBAL.

Eight physical scenarios:
1. m = 0,   We = 0
2. m > 0,   We = 0
3. m = 0,   We > 0  (Schilthuis)
4. m = 0,   We > 0  (Fetkovich)
5. m = 0,   We > 0  (V-E-H)
6. m > 0,   We > 0  (Schilthuis)
7. m > 0,   We > 0  (Fetkovich)
8. m > 0,   We > 0  (V-E-H)
"""

from pathlib import Path

from src.models.synthetic_reservoir import (
    SyntheticReservoir,
    SyntheticReservoirConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "data" / "sample_data" / "synthetic_8_cases"

TRUE_N = 10_000_000.0
TRUE_M = 0.8


SCENARIOS = [
    {
        "case": "case_1_no_gas_no_aquifer",
        "gas_cap_ratio": 0.0,
        "aquifer": "none",
    },
    {
        "case": "case_2_gas_cap_only",
        "gas_cap_ratio": TRUE_M,
        "aquifer": "none",
    },
    {
        "case": "case_3_schilthuis_only",
        "gas_cap_ratio": 0.0,
        "aquifer": "schilthuis",
    },
    {
        "case": "case_4_fetkovich_only",
        "gas_cap_ratio": 0.0,
        "aquifer": "fetkovich",
    },
    {
        "case": "case_5_veh_only",
        "gas_cap_ratio": 0.0,
        "aquifer": "van_everdingen_hurst",
    },
    {
        "case": "case_6_gas_cap_schilthuis",
        "gas_cap_ratio": TRUE_M,
        "aquifer": "schilthuis",
    },
    {
        "case": "case_7_gas_cap_fetkovich",
        "gas_cap_ratio": TRUE_M,
        "aquifer": "fetkovich",
    },
    {
        "case": "case_8_gas_cap_veh",
        "gas_cap_ratio": TRUE_M,
        "aquifer": "van_everdingen_hurst",
    },
]


def build_generator(gas_cap_ratio):
    config = SyntheticReservoirConfig(
        true_ooip_stb=TRUE_N,
        gas_cap_ratio=gas_cap_ratio,
        number_of_points=30,
        final_pressure_psia=2500.0,

        # Keep the already validated aquifer ground truths.
        schilthuis_aquifer_pressure_psia=4000.0,
        schilthuis_influx_constant_rb_per_day_psi=2.0,

        fetkovich_initial_aquifer_pressure_psia=4000.0,
        fetkovich_initial_water_volume_rb=5_000_000.0,
        fetkovich_total_compressibility_psi_inv=2.0e-5,
        fetkovich_productivity_index_rb_per_day_psi=100.0,

        veh_aquifer_permeability_md=100.0,
        veh_aquifer_porosity=0.20,
        veh_aquifer_thickness_ft=100.0,
        veh_total_compressibility_psi_inv=2.0e-5,
        veh_water_viscosity_cp=0.50,
        veh_reservoir_radius_ft=1000.0,
        veh_encroachment_angle_deg=360.0,
    )
    return SyntheticReservoir(config)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PyMBAL - GENERATING 8 COUPLED SYNTHETIC RESERVOIR DATASETS")
    print("=" * 78)
    print()

    for index, scenario in enumerate(SCENARIOS, start=1):
        generator = build_generator(scenario["gas_cap_ratio"])

        df = generator.generate(
            water_influx_model=scenario["aquifer"]
        )

        # Explicit metadata columns make the ground truth unambiguous.
        df["Synthetic_Case"] = scenario["case"]
        df["True_OOIP_STB"] = TRUE_N
        df["True_Gas_Cap_Ratio_m"] = scenario["gas_cap_ratio"]
        df["True_Aquifer_Model"] = scenario["aquifer"]

        output_dir = OUTPUT_ROOT / scenario["case"]
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = (
            output_dir
            / f"{scenario['case']}.csv"
        )

        df.to_csv(output_file, index=False)

        final_we = float(df["We_rb"].iloc[-1])

        print(
            f"{index}. {scenario['case']:<34} "
            f"m={scenario['gas_cap_ratio']:<4.1f}  "
            f"aquifer={scenario['aquifer']:<22} "
            f"We_final={final_we:,.2f} rb"
        )
        print(f"   -> {output_file}")

    print()
    print("=" * 78)
    print("✓ ALL 8 SYNTHETIC DATASETS GENERATED")
    print("=" * 78)
    print(f"Output directory: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
