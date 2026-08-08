"""
PyMBAL - PVT Correlation Registry

This module stores metadata about the PVT correlations
implemented in PyMBAL.

The actual mathematical equations are implemented in:
    oil.py
    gas.py
    water.py
"""


# ============================================================
# Oil Correlations
# ============================================================

OIL_CORRELATIONS = {

    "standing": {
        "name": "Standing",
        "author": "Standing",
        "year": 1947,
        "properties": [
            "Rs",
            "Pb",
            "Bo"
        ],
        "inputs": [
            "pressure",
            "temperature",
            "api",
            "gas_gravity"
        ]
    },

    "vasquez_beggs": {
        "name": "Vasquez-Beggs",
        "author": "Vasquez & Beggs",
        "year": 1980,
        "properties": [
            "Rs",
            "Pb",
            "Bo"
        ],
        "inputs": [
            "pressure",
            "temperature",
            "api",
            "gas_gravity"
        ]
    },

    "glaso": {
        "name": "Glaso",
        "author": "Glaso",
        "year": 1980,
        "properties": [
            "Rs",
            "Pb",
            "Bo"
        ],
        "inputs": [
            "pressure",
            "temperature",
            "api",
            "gas_gravity"
        ]
    }
}


# ============================================================
# Gas Correlations
# ============================================================

GAS_CORRELATIONS = {

    "hall_yarborough": {
        "name": "Hall-Yarborough",
        "properties": [
            "Z"
        ],
        "inputs": [
            "pressure",
            "temperature",
            "gas_gravity"
        ]
    },

    "dranchuk_abou_kassem": {
        "name": "Dranchuk-Abou-Kassem",
        "properties": [
            "Z"
        ],
        "inputs": [
            "pressure",
            "temperature",
            "gas_gravity"
        ]
    }
}


# ============================================================
# Water Correlations
# ============================================================

WATER_CORRELATIONS = {

    "mccain": {
        "name": "McCain",
        "properties": [
            "Bw",
            "water_viscosity"
        ],
        "inputs": [
            "pressure",
            "temperature",
            "salinity"
        ]
    }
}


# ============================================================
# Helper Functions
# ============================================================

def get_oil_correlations():
    """
    Return available oil PVT correlations.
    """
    return OIL_CORRELATIONS


def get_gas_correlations():
    """
    Return available gas PVT correlations.
    """
    return GAS_CORRELATIONS


def get_water_correlations():
    """
    Return available water PVT correlations.
    """
    return WATER_CORRELATIONS


def list_all_correlations():
    """
    Display all available PVT correlations.
    """

    print("\nAvailable PyMBAL PVT Correlations")
    print("=" * 50)

    print("\nOil:")
    for key, value in OIL_CORRELATIONS.items():
        print(f"  {key:25} → {value['name']}")

    print("\nGas:")
    for key, value in GAS_CORRELATIONS.items():
        print(f"  {key:25} → {value['name']}")

    print("\nWater:")
    for key, value in WATER_CORRELATIONS.items():
        print(f"  {key:25} → {value['name']}")