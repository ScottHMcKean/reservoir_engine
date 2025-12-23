"""EDA module - Exploratory data analysis for reservoir data."""

from reservoir_engine.eda.analysis import (
    calculate_statistics,
    detect_outliers,
    production_summary,
)
from reservoir_engine.eda.plots import (
    plot_decline_curve,
    plot_production_heatmap,
    plot_production_timeseries,
    plot_rate_cumulative,
)

__all__ = [
    # Analysis
    "calculate_statistics",
    "detect_outliers",
    "production_summary",
    # Plots
    "plot_production_timeseries",
    "plot_decline_curve",
    "plot_production_heatmap",
    "plot_rate_cumulative",
]

