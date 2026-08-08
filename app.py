"""
PyMBAL - Main Application

Current workflow:

    Load
      ↓
    Validate
      ↓
    Clean
      ↓
    PVT Engine
      ↓
    Output
"""

from src.utils.loader import DataLoader
from src.utils.validator import DataValidator
from src.utils.cleaner import DataCleaner
from src.pvt import PVTEngine


# ============================================================
# USER CONFIGURATION
# ============================================================

# Choose:
# "laboratory"
# "correlation"
# "hybrid"

PVT_MODE = "hybrid"


FILEPATH = (
    "data/sample_data/"
    "sample_hybrid_pvt_data.xlsx"
)


# Correlation inputs
# These are only required for correlation/hybrid
# calculations when properties need to be calculated.

TEMPERATURE = 200.0       # °F
OIL_API = 35.0
GAS_GRAVITY = 0.80
SALINITY = 5.0             # wt%


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print(
        "PyMBAL: An Intelligent Material Balance "
        "& Reserve Estimation System"
    )
    print("=" * 65)

    # --------------------------------------------------------
    # 1. LOAD
    # --------------------------------------------------------

    loader = DataLoader(
        FILEPATH
    )

    df = loader.load()

    # --------------------------------------------------------
    # 2. VALIDATE BASE DATA
    # --------------------------------------------------------

    DataValidator.validate_base_columns(
        df
    )

    DataValidator.validate_missing_values(
        df
    )

    DataValidator.validate_negative_values(
        df
    )

    # --------------------------------------------------------
    # 3. VALIDATE PVT INPUTS
    # --------------------------------------------------------

    if PVT_MODE in [
        "correlation",
        "hybrid"
    ]:

        DataValidator.validate_pvt_inputs(
            df
        )

    # --------------------------------------------------------
    # 4. Check supplied PVT
    # --------------------------------------------------------

    DataValidator.validate_pvt_properties(
        df
    )

    # --------------------------------------------------------
    # 5. CLEAN
    # --------------------------------------------------------

    df = DataCleaner.clean(
        df
    )

    # --------------------------------------------------------
    # 6. PVT ENGINE
    # --------------------------------------------------------

    pvt_engine = PVTEngine(

        mode=PVT_MODE,

        temperature=TEMPERATURE,

        api=OIL_API,

        gas_gravity=GAS_GRAVITY,

        salinity=SALINITY,
    )

    df = pvt_engine.calculate(
        df
    )

    # --------------------------------------------------------
    # 7. OUTPUT
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("FINAL PVT DATA")
    print("=" * 65)

    display_columns = [
        "Date",
        "Pressure_psia",
        "Bo_rb_per_STB",
        "Bg_rb_per_scf",
        "Bw_rb_per_STB",
        "Rs_scf_per_STB",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in df.columns
    ]

    print(
        df[available_columns].head(10)
    )

    print("\n")
    print(
        f"Final dataset shape: {df.shape}"
    )

    print(
        "\n✓ PyMBAL PVT workflow completed successfully."
    )


if __name__ == "__main__":
    main()