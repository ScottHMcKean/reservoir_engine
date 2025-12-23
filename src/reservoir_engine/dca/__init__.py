"""DCA module - Decline Curve Analysis for production forecasting.

Includes:
- Arps hyperbolic and Butler SAGD models
- Gas-first DCA engine with CGR-based condensate forecasting
"""

from reservoir_engine.dca.autofit import fit_arps, fit_butler
from reservoir_engine.dca.forecast import arps_forecast, butler_forecast, calculate_eur

# Gas-first DCA engine
from reservoir_engine.dca.gas_dca import (
    CGRConfig,
    CGRParams,
    DCAConfig,
    GasParams,
    TimeSeries,
    calendar_to_t_norm,
    fit_and_forecast,
    fit_cgr,
    fit_gas,
    forecast_all,
    forecast_condensate,
    forecast_gas,
    prepare_time_series,
    t_norm_to_calendar,
)
from reservoir_engine.dca.models import ArpsParams, ButlerParams, arps_rate, butler_rate

# SAGD Butler DCA engine
from reservoir_engine.dca.sagd_dca import (
    SAGDConfig,
    SAGDParams2Stage,
    SAGDParamsMultiSeg,
    SAGDTimeSeries,
    calculate_sagd_sor,
    fit_and_forecast_sagd,
    fit_sagd_2stage,
    fit_sagd_butler,
    fit_sagd_multiseg,
    forecast_sagd,
    prepare_sagd_series,
    sagd_rate,
)

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
    # Gas-first DCA
    "DCAConfig",
    "CGRConfig",
    "TimeSeries",
    "GasParams",
    "CGRParams",
    "prepare_time_series",
    "fit_gas",
    "fit_cgr",
    "forecast_gas",
    "forecast_condensate",
    "forecast_all",
    "fit_and_forecast",
    "t_norm_to_calendar",
    "calendar_to_t_norm",
    # SAGD Butler DCA
    "SAGDConfig",
    "SAGDTimeSeries",
    "SAGDParams2Stage",
    "SAGDParamsMultiSeg",
    "prepare_sagd_series",
    "fit_sagd_butler",
    "fit_sagd_2stage",
    "fit_sagd_multiseg",
    "sagd_rate",
    "forecast_sagd",
    "calculate_sagd_sor",
    "fit_and_forecast_sagd",
]

