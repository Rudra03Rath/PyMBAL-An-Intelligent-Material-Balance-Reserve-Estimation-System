"""
PyMBAL - Visualization Layer

Pure plotting utilities for already-computed PyMBAL results.

This module performs NO reservoir calculations. It only converts
validated result objects / arrays into Matplotlib figures.

The functions intentionally return Figure objects so that the same
visualization layer can be used by:
    - local Python scripts
    - notebooks
    - Streamlit
    - reporting/export code
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _to_array(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if len(arr) == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def _validate_same_length(*arrays: tuple[str, np.ndarray]) -> None:
    lengths = {len(arr) for _, arr in arrays}
    if len(lengths) != 1:
        raise ValueError(
            "All supplied series must have identical lengths."
        )


def _new_figure(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, ax


# ----------------------------------------------------------------------
# 1. Pressure history / forecast
# ----------------------------------------------------------------------

def plot_pressure_history(
    time_days,
    pressure_psia,
    forecast_time_days=None,
    forecast_pressure_psia=None,
):
    """
    Plot historical pressure and, optionally, forecast pressure.
    """
    t = _to_array(time_days, "time_days")
    p = _to_array(pressure_psia, "pressure_psia")
    _validate_same_length(("time_days", t), ("pressure_psia", p))

    fig, ax = _new_figure(
        "Reservoir Pressure History and Forecast",
        "Time (days)",
        "Pressure (psia)",
    )

    ax.plot(t, p, marker="o", label="Historical")

    if forecast_time_days is not None or forecast_pressure_psia is not None:
        if forecast_time_days is None or forecast_pressure_psia is None:
            raise ValueError(
                "Both forecast_time_days and forecast_pressure_psia "
                "must be supplied together."
            )

        ft = _to_array(forecast_time_days, "forecast_time_days")
        fp = _to_array(
            forecast_pressure_psia,
            "forecast_pressure_psia",
        )
        _validate_same_length(
            ("forecast_time_days", ft),
            ("forecast_pressure_psia", fp),
        )
        ax.plot(ft, fp, marker="o", linestyle="--", label="Forecast")

    ax.legend()
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 2. Material-balance diagnostic plots
# ----------------------------------------------------------------------

def plot_f_vs_expansion(
    expansion,
    f_rb,
    *,
    expansion_name: str = "Expansion",
):
    """
    Generic F versus expansion diagnostic.

    Can be used for:
        F vs Eo
        F vs Eg
        F vs Efw
    """
    x = _to_array(expansion, expansion_name)
    y = _to_array(f_rb, "f_rb")
    _validate_same_length((expansion_name, x), ("f_rb", y))

    fig, ax = _new_figure(
        f"Material-Balance Diagnostic: F vs {expansion_name}",
        expansion_name,
        "F (rb)",
    )

    ax.scatter(x, y, label="History")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_f_vs_eo(eo_rb_per_stb, f_rb):
    return plot_f_vs_expansion(
        eo_rb_per_stb,
        f_rb,
        expansion_name="Eo (rb/STB)",
    )


def plot_f_vs_eg(eg_rb_per_stb, f_rb):
    return plot_f_vs_expansion(
        eg_rb_per_stb,
        f_rb,
        expansion_name="Eg (rb/STB)",
    )


def plot_f_vs_efw(efw_rb_per_stb, f_rb):
    return plot_f_vs_expansion(
        efw_rb_per_stb,
        f_rb,
        expansion_name="Efw (rb/STB)",
    )


# ----------------------------------------------------------------------
# 3. Drive-mechanism DDI
# ----------------------------------------------------------------------

def plot_ddi_history(ddi_history: pd.DataFrame):
    """
    Plot DDI percentages through reservoir history.

    Expected columns:
        Solution-Gas Expansion
        Gas-Cap Expansion
        Aquifer Influx
        Rock + Connate-Water Expansion
    """
    if not isinstance(ddi_history, pd.DataFrame):
        raise TypeError("ddi_history must be a pandas DataFrame.")

    if ddi_history.empty:
        raise ValueError("ddi_history cannot be empty.")

    required = [
        "Solution-Gas Expansion",
        "Gas-Cap Expansion",
        "Aquifer Influx",
        "Rock + Connate-Water Expansion",
    ]

    missing = [c for c in required if c not in ddi_history.columns]
    if missing:
        raise ValueError(
            f"ddi_history is missing required columns: {missing}"
        )

    values = ddi_history[required].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError("ddi_history contains invalid numeric values.")

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(values))

    for column in required:
        ax.plot(
            x,
            values[column].to_numpy(dtype=float),
            label=column,
        )

    ax.set_title("Drive-Mechanism Contribution History")
    ax.set_xlabel("History point")
    ax.set_ylabel("DDI (%)")
    ax.set_ylim(0.0, 100.0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    return fig


def plot_final_ddi_pie(final_ddi: Mapping[str, float]):
    """
    Plot the final DDI contribution as a pie chart.

    Values must represent percentages or any positive values that
    form a meaningful part-to-whole composition.
    """
    if not final_ddi:
        raise ValueError("final_ddi cannot be empty.")

    labels = list(final_ddi.keys())
    values = np.asarray(
        [final_ddi[label] for label in labels],
        dtype=float,
    )

    if not np.all(np.isfinite(values)):
        raise ValueError("final_ddi contains invalid values.")

    if np.any(values < 0.0):
        raise ValueError("DDI values cannot be negative.")

    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("DDI total must be positive.")

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
    )

    ax.set_title("Final Drive-Mechanism Contribution")
    fig.tight_layout()

    return fig


# ----------------------------------------------------------------------
# 4. DCA fitted curves
# ----------------------------------------------------------------------

def plot_dca_fits(
    dca_results: Mapping,
    *,
    show_historical: bool = True,
):
    """
    Plot fitted and forecast DCA curves.

    dca_results is expected to be the dictionary exposed by
    ForecastScenarioResult.dca_results.
    """
    if not dca_results:
        raise ValueError("dca_results cannot be empty.")

    fig, ax = plt.subplots(figsize=(9, 5))

    historical_plotted = False

    for model_name, result in dca_results.items():
        if not getattr(result, "success", False):
            continue

        if show_historical and not historical_plotted:
            ax.scatter(
                result.historical_time,
                result.historical_rate,
                label="Historical",
                zorder=5,
            )
            historical_plotted = True

        ax.plot(
            result.historical_time,
            result.fitted_rate,
            label=f"{model_name} fit",
        )

        ax.plot(
            result.forecast_time,
            result.forecast_rate,
            linestyle="--",
            label=f"{model_name} forecast",
        )

    if not historical_plotted and not any(
        getattr(result, "success", False)
        for result in dca_results.values()
    ):
        raise ValueError("No successful DCA result is available.")

    ax.set_title("Decline-Curve Analysis")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Production rate")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    return fig


# ----------------------------------------------------------------------
# 5. Cumulative production / EUR
# ----------------------------------------------------------------------

def plot_cumulative_production(dca_results: Mapping):
    """
    Plot cumulative production forecasts for successful DCA models.
    """
    if not dca_results:
        raise ValueError("dca_results cannot be empty.")

    fig, ax = plt.subplots(figsize=(9, 5))

    plotted = False

    for model_name, result in dca_results.items():
        if not getattr(result, "success", False):
            continue

        ax.plot(
            result.forecast_time,
            result.forecast_cumulative,
            label=model_name,
        )
        plotted = True

    if not plotted:
        raise ValueError("No successful DCA result is available.")

    ax.set_title("Forecast Cumulative Production")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Cumulative production")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    return fig


# ----------------------------------------------------------------------
# 6. Candidate OOIP / GIIP comparison
# ----------------------------------------------------------------------

def plot_candidate_volumes(
    candidates: pd.DataFrame,
    *,
    volume_type: str = "OOIP",
):
    """
    Plot candidate OOIP or GIIP values.

    volume_type:
        "OOIP" -> OOIP_STB
        "GIIP" -> GIIP_Bscf
    """
    if not isinstance(candidates, pd.DataFrame):
        raise TypeError("candidates must be a pandas DataFrame.")

    if candidates.empty:
        raise ValueError("candidates cannot be empty.")

    mapping = {
        "OOIP": ("OOIP_STB", "OOIP (STB)"),
        "GIIP": ("GIIP_Bscf", "GIIP (Bscf)"),
    }

    key = volume_type.upper()
    if key not in mapping:
        raise ValueError(
            "volume_type must be either 'OOIP' or 'GIIP'."
        )

    column, ylabel = mapping[key]

    if column not in candidates.columns:
        raise ValueError(
            f"Candidate table is missing '{column}'."
        )

    labels = candidates.get(
        "Configuration",
        pd.Series(
            [f"Candidate {i+1}" for i in range(len(candidates))]
        ),
    ).astype(str)

    values = pd.to_numeric(
        candidates[column],
        errors="coerce",
    )

    if values.isna().any():
        raise ValueError(
            f"Candidate table contains invalid {column} values."
        )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        labels,
        values,
    )

    ax.set_title(f"Candidate {key} Comparison")
    ax.set_xlabel("Candidate configuration")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    return fig


# ----------------------------------------------------------------------
# 7. Reserves comparison
# ----------------------------------------------------------------------

def plot_reserves_comparison(
    reserves,
    *,
    model_column: str = "Model",
    reserves_column: str = "Remaining_Reserves",
):
    """
    Plot remaining technical reserves by DCA model.

    Accepts either a DataFrame or an object exposing a
    comparison_table() method.
    """
    if hasattr(reserves, "comparison_table"):
        table = reserves.comparison_table()
    elif isinstance(reserves, pd.DataFrame):
        table = reserves.copy()
    else:
        raise TypeError(
            "reserves must be a DataFrame or expose comparison_table()."
        )

    if table.empty:
        raise ValueError("Reserves table cannot be empty.")

    if model_column not in table.columns:
        raise ValueError(
            f"Reserves table is missing '{model_column}'."
        )

    if reserves_column not in table.columns:
        raise ValueError(
            f"Reserves table is missing '{reserves_column}'."
        )

    labels = table[model_column].astype(str)
    values = pd.to_numeric(
        table[reserves_column],
        errors="coerce",
    )

    if values.isna().any():
        raise ValueError(
            f"Invalid values found in '{reserves_column}'."
        )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        labels,
        values,
    )

    ax.set_title("Remaining Technical Reserves by DCA Model")
    ax.set_xlabel("DCA model")
    ax.set_ylabel("Remaining reserves")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    return fig


__all__ = [
    "plot_pressure_history",
    "plot_f_vs_expansion",
    "plot_f_vs_eo",
    "plot_f_vs_eg",
    "plot_f_vs_efw",
    "plot_ddi_history",
    "plot_final_ddi_pie",
    "plot_dca_fits",
    "plot_cumulative_production",
    "plot_candidate_volumes",
    "plot_reserves_comparison",
]
