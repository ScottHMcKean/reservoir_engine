"""Production forecasting using decline curve models.

Generate forecasts and calculate EUR (Estimated Ultimate Recovery)
using fitted decline curve parameters.

All forecasts use normalized months (fixed 30.4-day periods).
Calendar dates are optional for visualization but the primary
time axis is always normalized months.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from reservoir_engine.dca.models import (
    ArpsParams,
    ButlerParams,
    arps_cumulative,
    arps_rate,
    butler_cumulative,
    butler_rate,
)


def arps_forecast(
    params: ArpsParams,
    months: int = 360,
    start_date: date | None = None,
    economic_limit: float = 0.0,
    include_calendar_dates: bool = True,
) -> pd.DataFrame:
    """Generate production forecast using Arps decline on normalized months.

    All rates and times are in normalized months (fixed 30.4-day periods).
    Calendar dates are optional and provided for visualization convenience.

    Args:
        params: ArpsParams with qi, di, b values (per normalized month)
        months: Forecast horizon in normalized months (default 360)
        start_date: Optional start date for calendar mapping
        economic_limit: Minimum economic rate (forecast stops when reached)
        include_calendar_dates: Whether to include calendar dates column

    Returns:
        DataFrame with columns: t_norm, rate, cumulative, [calendar_date]

    Example:
        >>> params = ArpsParams(qi=1000, di=0.15, b=0.8)
        >>> forecast = arps_forecast(params, months=120)
        >>> print(forecast.tail())
    """
    t = np.arange(months + 1)
    rates = arps_rate(t, params)
    cumulative = arps_cumulative(t, params)

    df = pd.DataFrame(
        {
            "t_norm": t,
            "rate": rates,
            "cumulative": cumulative,
        }
    )

    # Optionally add calendar dates for visualization
    if include_calendar_dates:
        if start_date is None:
            start_date = date.today()
        dates = [start_date + timedelta(days=30 * i) for i in range(months + 1)]
        df["calendar_date"] = dates

    # Apply economic limit
    if economic_limit > 0:
        df = df[df["rate"] >= economic_limit].copy()

    return df


def butler_forecast(
    params: ButlerParams,
    months: int = 240,
    start_date: date | None = None,
    economic_limit: float = 0.0,
    include_sor: bool = True,
    include_calendar_dates: bool = True,
) -> pd.DataFrame:
    """Generate production forecast using Butler SAGD model on normalized months.

    All rates and times are in normalized months (fixed 30.4-day periods).
    Calendar dates are optional and provided for visualization convenience.

    Args:
        params: ButlerParams with q_peak, m, tau values (per normalized month)
        months: Forecast horizon in normalized months (default 240)
        start_date: Optional start date for calendar mapping
        economic_limit: Minimum economic rate
        include_sor: If True, include steam requirement column
        include_calendar_dates: Whether to include calendar dates column

    Returns:
        DataFrame with columns: t_norm, rate, cumulative, [steam_required], [calendar_date]

    Example:
        >>> params = ButlerParams(q_peak=500, m=0.1, tau=12)
        >>> forecast = butler_forecast(params, months=120)
    """
    t = np.arange(months + 1)
    rates = butler_rate(t, params)
    cumulative = butler_cumulative(t, params)

    df = pd.DataFrame(
        {
            "t_norm": t,
            "rate": rates,
            "cumulative": cumulative,
        }
    )

    if include_sor:
        # Calculate steam required based on SOR
        df["steam_required"] = df["rate"] * params.sor

    # Optionally add calendar dates for visualization
    if include_calendar_dates:
        if start_date is None:
            start_date = date.today()
        dates = [start_date + timedelta(days=30 * i) for i in range(months + 1)]
        df["calendar_date"] = dates

    # Apply economic limit
    if economic_limit > 0:
        df = df[df["rate"] >= economic_limit].copy()

    return df


def calculate_eur(
    params: ArpsParams | ButlerParams,
    economic_limit: float,
    max_months: int = 600,
) -> float:
    """Calculate EUR (Estimated Ultimate Recovery).

    EUR is the cumulative production when rate drops to economic limit.

    Args:
        params: Decline curve parameters (ArpsParams or ButlerParams)
        economic_limit: Economic rate limit (production stops below this)
        max_months: Maximum forecast horizon

    Returns:
        EUR value (cumulative production at economic limit)

    Example:
        >>> params = ArpsParams(qi=1000, di=0.15, b=0.8)
        >>> eur = calculate_eur(params, economic_limit=10)
        >>> print(f"EUR: {eur:,.0f} bbl")
    """
    t = np.arange(max_months + 1)

    if isinstance(params, ArpsParams):
        rates = arps_rate(t, params)
        cumulative = arps_cumulative(t, params)
    else:
        rates = butler_rate(t, params)
        cumulative = butler_cumulative(t, params)

    # Find peak production month
    peak_idx = np.argmax(rates)

    # Find first month below economic limit AFTER peak (for ramp-up models)
    # For declining models like Arps, peak is at t=0
    rates_after_peak = rates[peak_idx:]
    below_limit_after_peak = np.where(rates_after_peak < economic_limit)[0]

    if len(below_limit_after_peak) > 0:
        # Index relative to original array
        idx = peak_idx + below_limit_after_peak[0]
        if idx > 0:
            return float(cumulative[idx - 1])
        return float(cumulative[0])
    else:
        # Economic limit not reached in forecast horizon
        return float(cumulative[-1])


def combine_forecasts(
    forecasts: list[tuple[str, pd.DataFrame]],
    agg_method: str = "sum",
) -> pd.DataFrame:
    """Combine multiple well forecasts into aggregate forecast.

    Args:
        forecasts: List of (uwi, forecast_df) tuples
        agg_method: Aggregation method - "sum" or "mean"

    Returns:
        DataFrame with aggregated forecast by normalized month

    Example:
        >>> forecasts = [("WELL-001", fc1), ("WELL-002", fc2)]
        >>> total = combine_forecasts(forecasts, agg_method="sum")
    """
    if not forecasts:
        return pd.DataFrame()

    # Add UWI to each forecast
    dfs = []
    for uwi, fc in forecasts:
        fc_copy = fc.copy()
        fc_copy["uwi"] = uwi
        dfs.append(fc_copy)

    combined = pd.concat(dfs, ignore_index=True)

    # Aggregate by normalized month
    agg_dict = {
        "rate": agg_method,
        "cumulative": agg_method,
    }

    # Include calendar date if available
    if "calendar_date" in combined.columns:
        agg_dict["calendar_date"] = "first"

    result = combined.groupby("t_norm").agg(agg_dict).reset_index()

    return result

