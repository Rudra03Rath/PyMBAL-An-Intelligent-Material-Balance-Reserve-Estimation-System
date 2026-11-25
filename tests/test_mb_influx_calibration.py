"""
PyMBAL - MATERIAL-BALANCE-BASED V-E-H WATER INFLUX CALIBRATION

The coupled calibrator receives only:
    Days, Pressure, F and E.

We_rb exists in the synthetic CSV only as ground truth and is NEVER
passed into MaterialBalanceInfluxCalibrator.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.influx.mb_calibration import MaterialBalanceInfluxCalibrator


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SYNTHETIC_FILE = (
    PROJECT_ROOT
    / "data"
    / "sample_data"
    / "synthetic_van_everdingen_hurst"
    / "synthetic_van_everdingen_hurst_reservoir.csv"
)

TRUE_OOIP_STB = 10_000_000.0
TRUE_AQUIFER_PRESSURE_PSIA = 4000.0
TRUE_K_MD = 100.0
TRUE_POROSITY = 0.20
TRUE_THICKNESS_FT = 100.0
TRUE_CT = 2.0e-5
TRUE_MU_CP = 0.50
TRUE_RADIUS_FT = 1000.0
TRUE_ANGLE_DEG = 360.0

OOIP_RELATIVE_TOLERANCE = 1e-5
K_RELATIVE_TOLERANCE = 1e-5
RMSE_TOLERANCE_RB = 1e-3
WE_TOLERANCE_RB = 1e-2


def load_dataset():
    if not SYNTHETIC_FILE.exists():
        raise FileNotFoundError(
            f"\nSynthetic V-E-H dataset not found:\n{SYNTHETIC_FILE}"
        )

    df = pd.read_csv(SYNTHETIC_FILE)

    print("✓ Loaded synthetic V-E-H dataset.")
    print(f"  File   : {SYNTHETIC_FILE}")
    print(f"  Rows   : {len(df)}")

    return df


def validate_dataset(df):
    print()
    print("=" * 70)
    print("DATASET VALIDATION")
    print("=" * 70)

    required = [
        "Days",
        "Pressure_psia",
        "F_rb",
        "Etotal_rb_per_STB",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    days = df["Days"].to_numpy(dtype=float)
    pressure = df["Pressure_psia"].to_numpy(dtype=float)
    F = df["F_rb"].to_numpy(dtype=float)
    E = df["Etotal_rb_per_STB"].to_numpy(dtype=float)

    if len(df) < 3:
        raise ValueError("At least three reservoir data points are required.")
    if not np.all(np.isfinite(days)):
        raise ValueError("Days contains invalid values.")
    if not np.isclose(days[0], 0.0):
        raise ValueError("Initial Days value must be zero.")
    if np.any(np.diff(days) <= 0.0):
        raise ValueError("Days must be strictly increasing.")
    if not np.all(np.isfinite(pressure)) or np.any(pressure <= 0.0):
        raise ValueError("Pressure contains invalid values.")
    if pressure[-1] >= pressure[0]:
        raise ValueError("Pressure history must decline.")
    if not np.all(np.isfinite(F)) or np.any(F < 0.0):
        raise ValueError("F_rb contains invalid values.")
    if not np.all(np.isfinite(E)) or np.any(E < 0.0):
        raise ValueError("Expansion contains invalid values.")
    if abs(F[0]) > 1e-12:
        raise ValueError("Initial F_rb must be zero.")
    if abs(E[0]) > 1e-12:
        raise ValueError("Initial expansion must be zero.")

    print("✓ Required material-balance columns present.")
    print("✓ Time history is valid.")
    print("✓ Pressure history is valid.")
    print("✓ Underground withdrawal history is valid.")
    print("✓ Expansion history is valid.")

    if "We_rb" in df.columns:
        We = df["We_rb"].to_numpy(dtype=float)

        if not np.all(np.isfinite(We)):
            raise ValueError("Ground-truth We contains invalid values.")
        if np.any(We < 0.0):
            raise ValueError("Ground-truth We cannot be negative.")
        if np.any(np.diff(We) < -1e-10):
            raise ValueError(
                "Ground-truth cumulative We must be non-decreasing."
            )

        print(
            "✓ Ground-truth We history is valid "
            "(validation only; not supplied to calibration)."
        )


def print_result(result):
    print()
    print("Model")
    print("-" * 70)
    print(f"Name       : {result.model_name}")
    print(f"Success    : {result.success}")
    print(f"RMSE       : {result.rmse_rb:.12f} rb")
    print(f"MAE        : {result.mae_rb:.12f} rb")
    print(f"R²         : {result.r2:.12f}")

    print()
    print("Estimated OOIP")
    print("-" * 70)
    print(f"N           : {result.estimated_ooip_stb:.6f} STB")
    print(
        f"N           : "
        f"{result.estimated_ooip_stb / 1_000_000:.6f} MMSTB"
    )

    print()
    print("Calibrated Parameters")
    print("-" * 70)

    for key, value in result.parameters.items():
        print(f"{key:<45}: {value:.12g}")


def validate_result_structure(result):
    print()
    print("=" * 70)
    print("RESULT STRUCTURE VALIDATION")
    print("=" * 70)

    required_attributes = [
        "model_name",
        "success",
        "estimated_ooip_stb",
        "parameters",
        "rmse_rb",
        "mae_rb",
        "r2",
        "residuals_rb",
        "predicted_we_rb",
        "f_minus_we_rb",
        "expansion_rb_per_stb",
    ]

    for attribute in required_attributes:
        assert hasattr(result, attribute), (
            f"Missing result attribute: {attribute}"
        )

    required_parameters = [
        "aquifer_permeability_md",
        "aquifer_porosity",
        "aquifer_thickness_ft",
        "total_compressibility_psi_inv",
        "water_viscosity_cp",
        "reservoir_radius_ft",
        "encroachment_angle_deg",
    ]

    for parameter in required_parameters:
        assert parameter in result.parameters, (
            f"Missing calibrated parameter: {parameter}"
        )

    assert result.model_name == "Van Everdingen-Hurst"

    print("✓ Calibration result structure PASSED.")


def validate_material_balance_fit(result):
    print()
    print("=" * 70)
    print("MATERIAL-BALANCE FIT VALIDATION")
    print("=" * 70)

    print(f"RMSE : {result.rmse_rb:.12f} rb")
    print(f"MAE  : {result.mae_rb:.12f} rb")
    print(f"R²   : {result.r2:.12f}")

    assert result.success, (
        "V-E-H coupled calibration did not converge."
    )

    assert result.rmse_rb < RMSE_TOLERANCE_RB, (
        "V-E-H material-balance RMSE is too large."
    )

    assert result.r2 > 0.999999999, (
        "V-E-H material-balance R² is too low."
    )

    print("✓ Calibration convergence check PASSED.")
    print("✓ RMSE check PASSED.")
    print("✓ R² check PASSED.")


def validate_recovered_parameters(result):
    print()
    print("=" * 70)
    print("PARAMETER RECOVERY VALIDATION")
    print("=" * 70)

    N = result.estimated_ooip_stb
    k = result.parameters["aquifer_permeability_md"]

    N_error = abs(N - TRUE_OOIP_STB) / TRUE_OOIP_STB
    k_error = abs(k - TRUE_K_MD) / TRUE_K_MD

    print(f"True OOIP       : {TRUE_OOIP_STB:.6f} STB")
    print(f"Calibrated OOIP : {N:.6f} STB")
    print(f"Relative error  : {N_error:.6e}")

    print()
    print(f"True k          : {TRUE_K_MD:.6f} md")
    print(f"Calibrated k    : {k:.6f} md")
    print(f"Relative error  : {k_error:.6e}")

    assert N_error < OOIP_RELATIVE_TOLERANCE, (
        "V-E-H coupled calibration did not recover OOIP."
    )

    assert k_error < K_RELATIVE_TOLERANCE, (
        "V-E-H coupled calibration did not recover permeability."
    )

    print("✓ OOIP parameter recovery PASSED.")
    print("✓ V-E-H permeability recovery PASSED.")


def validate_predicted_water_influx(result, df):
    print()
    print("=" * 70)
    print("PREDICTED WATER-INFLUX VALIDATION")
    print("=" * 70)

    if "We_rb" not in df.columns:
        raise ValueError(
            "Synthetic V-E-H CSV must contain We_rb for independent "
            "ground-truth validation."
        )

    true_we = df["We_rb"].to_numpy(dtype=float)
    predicted_we = np.asarray(result.predicted_we_rb, dtype=float)

    assert len(true_we) == len(predicted_we)

    errors = predicted_we - true_we
    max_error = float(np.max(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    print(f"True final We      : {true_we[-1]:.6f} rb")
    print(f"Predicted final We : {predicted_we[-1]:.6f} rb")
    print(f"Maximum |We error| : {max_error:.9e} rb")
    print(f"We RMSE            : {rmse:.9e} rb")

    assert max_error < WE_TOLERANCE_RB, (
        "Calibrated V-E-H We does not reproduce the "
        "synthetic ground-truth history."
    )

    print("✓ Predicted water-influx check PASSED.")


def display_calibration_history(result, df):
    output = pd.DataFrame(
        {
            "Days": df["Days"].to_numpy(dtype=float),
            "Pressure_psia": df["Pressure_psia"].to_numpy(dtype=float),
            "F_rb": df["F_rb"].to_numpy(dtype=float),
            "Etotal_rb_per_STB": df[
                "Etotal_rb_per_STB"
            ].to_numpy(dtype=float),
            "Predicted_We_rb": result.predicted_we_rb,
            "F_minus_We_rb": result.f_minus_we_rb,
            "MBE_residual_rb": result.residuals_rb,
        }
    )

    print()
    print("=" * 70)
    print("CALIBRATED MATERIAL-BALANCE HISTORY")
    print("=" * 70)
    print(output.head(10).to_string(index=False))


def main():
    print("=" * 70)
    print(
        "PyMBAL - MATERIAL-BALANCE-BASED "
        "VAN EVERDINGEN-HURST WATER INFLUX CALIBRATION VALIDATION"
    )
    print("=" * 70)

    print()
    print("GROUND TRUTH")
    print("-" * 70)
    print(f"True OOIP              : {TRUE_OOIP_STB:,.0f} STB")
    print(
        f"True Aquifer Pressure  : "
        f"{TRUE_AQUIFER_PRESSURE_PSIA:.2f} psia"
    )
    print(f"True V-E-H permeability : {TRUE_K_MD:.2f} md")
    print(f"Aquifer porosity       : {TRUE_POROSITY:.4f}")
    print(f"Aquifer thickness      : {TRUE_THICKNESS_FT:.2f} ft")
    print(f"Aquifer ct             : {TRUE_CT:.6e} psi⁻¹")
    print(f"Water viscosity        : {TRUE_MU_CP:.4f} cp")
    print(f"Reservoir radius       : {TRUE_RADIUS_FT:.2f} ft")
    print(f"Encroachment angle     : {TRUE_ANGLE_DEG:.2f}°")

    print()
    print("IMPORTANT:")
    print("We_rb is NOT supplied to the calibration engine.")
    print("The engine must infer k and OOIP from F and E.")

    df = load_dataset()
    validate_dataset(df)

    print()
    print("=" * 70)
    print("COUPLED VAN EVERDINGEN-HURST CALIBRATION")
    print("=" * 70)

    calibrator = MaterialBalanceInfluxCalibrator(
        time_days=df["Days"].to_numpy(dtype=float),
        pressure_psia=df["Pressure_psia"].to_numpy(dtype=float),
        F_rb=df["F_rb"].to_numpy(dtype=float),
        E_rb_per_STB=df[
            "Etotal_rb_per_STB"
        ].to_numpy(dtype=float),
    )

    print("✓ MaterialBalanceInfluxCalibrator constructed.")

    result = calibrator.calibrate_van_everdingen_hurst(
        aquifer_porosity=TRUE_POROSITY,
        aquifer_thickness_ft=TRUE_THICKNESS_FT,
        total_compressibility_psi_inv=TRUE_CT,
        water_viscosity_cp=TRUE_MU_CP,
        reservoir_radius_ft=TRUE_RADIUS_FT,
        encroachment_angle_deg=TRUE_ANGLE_DEG,
        permeability_bounds=(1.0, 1000.0),
    )

    print_result(result)
    validate_result_structure(result)
    validate_material_balance_fit(result)
    validate_recovered_parameters(result)
    validate_predicted_water_influx(result, df)
    display_calibration_history(result, df)

    print()
    print("=" * 70)
    print(
        "✓ ALL MATERIAL-BALANCE-BASED "
        "VAN EVERDINGEN-HURST CALIBRATION CHECKS PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
