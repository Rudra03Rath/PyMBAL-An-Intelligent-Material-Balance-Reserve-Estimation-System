"""
PyMBAL - VISUALIZATION LAYER VALIDATION

The visualization layer must:
    - consume already-computed results,
    - return Matplotlib Figure objects,
    - perform no reservoir calculations,
    - validate input structure,
    - remain usable independently of Streamlit.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.visualization.plots import (
    plot_pressure_history,
    plot_f_vs_eo,
    plot_f_vs_eg,
    plot_f_vs_efw,
    plot_ddi_history,
    plot_final_ddi_pie,
    plot_dca_fits,
    plot_cumulative_production,
    plot_candidate_volumes,
    plot_reserves_comparison,
)


class MockDCAResult:
    def __init__(self, model_name, offset):
        self.model_name = model_name
        self.success = True

        self.historical_time = np.array(
            [0.0, 100.0, 200.0, 300.0]
        )

        self.historical_rate = np.array(
            [1000.0, 900.0, 820.0, 750.0]
        )

        self.fitted_rate = (
            self.historical_rate
            + offset
        )

        self.forecast_time = np.array(
            [300.0, 400.0, 500.0, 600.0]
        )

        self.forecast_rate = np.array(
            [750.0, 690.0, 640.0, 590.0]
        )

        self.forecast_cumulative = np.array(
            [100000.0, 125000.0, 148000.0, 169000.0]
        )


class MockReserves:
    def comparison_table(self):
        return pd.DataFrame(
            {
                "Model": [
                    "Exponential",
                    "Harmonic",
                    "Hyperbolic",
                ],
                "Remaining_Reserves": [
                    1000000.0,
                    1100000.0,
                    1250000.0,
                ],
            }
        )


def close_fig(fig):
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_pressure_plot():
    fig = plot_pressure_history(
        [0, 100, 200],
        [4000, 3800, 3600],
        [300, 400],
        [3400, 3200],
    )
    close_fig(fig)


def test_material_balance_plots():
    eo = [0.0, 0.03, 0.06]
    eg = [0.0, 0.04, 0.08]
    efw = [0.0, 0.002, 0.004]
    f = [0.0, 5000.0, 10000.0]

    close_fig(plot_f_vs_eo(eo, f))
    close_fig(plot_f_vs_eg(eg, f))
    close_fig(plot_f_vs_efw(efw, f))


def test_ddi_plots():
    ddi = pd.DataFrame(
        {
            "Solution-Gas Expansion": [70.0, 68.0, 65.0],
            "Gas-Cap Expansion": [0.0, 0.0, 0.0],
            "Aquifer Influx": [0.0, 10.0, 20.0],
            "Rock + Connate-Water Expansion": [30.0, 22.0, 15.0],
        }
    )

    close_fig(plot_ddi_history(ddi))

    close_fig(
        plot_final_ddi_pie(
            {
                "Solution-Gas Expansion": 65.0,
                "Gas-Cap Expansion": 0.0,
                "Aquifer Influx": 20.0,
                "Rock + Connate-Water Expansion": 15.0,
            }
        )
    )


def test_dca_plots():
    results = {
        "Exponential": MockDCAResult(
            "Exponential",
            0.0,
        ),
        "Harmonic": MockDCAResult(
            "Harmonic",
            5.0,
        ),
        "Hyperbolic": MockDCAResult(
            "Hyperbolic",
            -5.0,
        ),
    }

    close_fig(plot_dca_fits(results))
    close_fig(plot_cumulative_production(results))


def test_volume_plots():
    candidates = pd.DataFrame(
        {
            "Configuration": [
                "No Gas Cap + No Aquifer",
                "Gas Cap + No Aquifer",
                "Gas Cap + Aquifer",
            ],
            "OOIP_STB": [
                10_000_000.0,
                10_500_000.0,
                9_800_000.0,
            ],
            "GIIP_Bscf": [
                0.0,
                12.0,
                10.0,
            ],
        }
    )

    close_fig(
        plot_candidate_volumes(
            candidates,
            volume_type="OOIP",
        )
    )

    close_fig(
        plot_candidate_volumes(
            candidates,
            volume_type="GIIP",
        )
    )


def test_reserves_plot():
    close_fig(
        plot_reserves_comparison(
            MockReserves()
        )
    )


def test_invalid_inputs():
    try:
        plot_pressure_history(
            [0, 1],
            [4000],
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Mismatched pressure inputs were not rejected."
        )

    try:
        plot_final_ddi_pie(
            {
                "Solution-Gas Expansion": -1.0,
            }
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Negative DDI was not rejected."
        )

    try:
        plot_candidate_volumes(
            pd.DataFrame(
                {
                    "Configuration": ["A"],
                }
            ),
            volume_type="OOIP",
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Missing OOIP column was not rejected."
        )


def main():
    print("=" * 70)
    print("PyMBAL - VISUALIZATION LAYER VALIDATION")
    print("=" * 70)
    print()
    print(
        "This test validates plotting utilities above the already "
        "validated analysis layers."
    )
    print()

    print("=" * 70)
    print("PRESSURE VISUALIZATION")
    print("=" * 70)
    test_pressure_plot()
    print("✓ Pressure history/forecast plot returned.")
    print("✓ PASSED.")
    print()

    print("=" * 70)
    print("MATERIAL-BALANCE DIAGNOSTICS")
    print("=" * 70)
    test_material_balance_plots()
    print("✓ F-vs-Eo plot returned.")
    print("✓ F-vs-Eg plot returned.")
    print("✓ F-vs-Efw plot returned.")
    print("✓ PASSED.")
    print()

    print("=" * 70)
    print("DRIVE-MECHANISM VISUALIZATION")
    print("=" * 70)
    test_ddi_plots()
    print("✓ DDI history plot returned.")
    print("✓ Final DDI pie chart returned.")
    print("✓ PASSED.")
    print()

    print("=" * 70)
    print("DCA VISUALIZATION")
    print("=" * 70)
    test_dca_plots()
    print("✓ DCA fitted/forecast plot returned.")
    print("✓ Cumulative-production plot returned.")
    print("✓ PASSED.")
    print()

    print("=" * 70)
    print("VOLUMETRIC / RESERVES VISUALIZATION")
    print("=" * 70)
    test_volume_plots()
    test_reserves_plot()
    print("✓ OOIP comparison plot returned.")
    print("✓ GIIP comparison plot returned.")
    print("✓ Reserves comparison plot returned.")
    print("✓ PASSED.")
    print()

    print("=" * 70)
    print("INPUT VALIDATION")
    print("=" * 70)
    test_invalid_inputs()
    print("✓ Invalid plotting inputs rejected.")
    print("✓ PASSED.")
    print()

    print("=" * 70)
    print("✓ ALL VISUALIZATION CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
