"""DCA module - Decline Curve Analysis for production forecasting."""

from reservoir_engine.dca.autofit import fit_arps, fit_butler
from reservoir_engine.dca.forecast import arps_forecast, butler_forecast, calculate_eur
from reservoir_engine.dca.models import ArpsParams, ButlerParams, arps_rate, butler_rate

__all__ = [
    # Models
    "ArpsParams",
    "ButlerParams",
    "arps_rate",
    "butler_rate",
    # Autofit
    "fit_arps",
    "fit_butler",
    # Forecast
    "arps_forecast",
    "butler_forecast",
    "calculate_eur",
]

