"""Butler-based SAGD DCA engine.

Uses fixed-length normalized months, two-stage or multi-segment
Butler physics (rising -> spreading/decline), and shared terminal decline.

Calendar time stays in I/O and visualization; all fitting and
forecasting runs on normalized months.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

# Import shared time utilities from gas_dca
from reservoir_engine.dca.gas_dca import (
    NORMALIZED_MONTH_DAYS,
    t_norm_to_calendar,
)

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SAGDConfig:
    """Configuration for Butler-based SAGD DCA engine."""

    # Model type
    model: Literal["butler_2stage", "butler_multiseg"] = "butler_2stage"

    # Segment configuration (for multiseg)
    num_segments: int = 2
    segment_boundaries: list[int] = field(default_factory=list)

    # Terminal decline (annual)
    use_terminal_decline: bool = True
    terminal_decline_year: float = 0.06  # 6% annual
    df_year_bounds: tuple[float, float] = (0.02, 0.15)

    # Butler exponent bounds
    alpha_r_bounds: tuple[float, float] = (0.3, 0.7)  # Rising exponent
    alpha_s_bounds: tuple[float, float] = (0.2, 0.8)  # Spreading exponent

    # Multi-segment beta bounds
    beta_rising_bounds: tuple[float, float] = (0.2, 0.8)
    beta_decline_bounds: tuple[float, float] = (-0.9, -0.1)

    # Fitting options
    drop_heatup_months: int = 3  # Drop early heat-up period
    min_months_for_fit: int = 12
    use_global_optimization: bool = False  # Use differential evolution

    @property
    def terminal_decline_month(self) -> float:
        """Convert annual terminal decline to monthly."""
        return 1 - (1 - self.terminal_decline_year) ** (1 / 12)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SAGDTimeSeries:
    """Normalized time series for SAGD fitting."""

    t_norm: np.ndarray  # Normalized month indices (0, 1, 2, ...)
    qo: np.ndarray  # Oil rate per normalized month
    steam: np.ndarray  # Steam volume per normalized month (optional)
    cum_oil: np.ndarray  # Cumulative oil
    cum_steam: np.ndarray  # Cumulative steam
    calendar_dates: list[date]  # Original calendar dates
    calendar_start: date  # Start date for calendar mapping


@dataclass
class SAGDParams2Stage:
    """Two-stage Butler SAGD parameters."""

    # Rising stage: q(t) = A_r * t^alpha_r
    a_r: float  # Rising coefficient
    alpha_r: float  # Rising exponent (~0.5 for classic Butler)

    # Spreading stage: q(t) = A_s * t^(-alpha_s)
    # A_s is computed from continuity: A_s = A_r * t_tr^(alpha_r + alpha_s)
    alpha_s: float  # Spreading/decline exponent (~0.5 for classic Butler)

    # Transition time (normalized months)
    t_tr: float

    # Terminal decline (monthly)
    terminal_decline_month: float = 0.0
    t_terminal: float = np.inf  # Time when terminal decline kicks in

    @property
    def a_s(self) -> float:
        """Compute spreading coefficient from continuity."""
        return self.a_r * (self.t_tr ** (self.alpha_r + self.alpha_s))

    @property
    def q_peak(self) -> float:
        """Peak rate at transition time."""
        return self.a_r * (self.t_tr ** self.alpha_r)


@dataclass
class SAGDParamsMultiSeg:
    """Multi-segment Butler SAGD parameters."""

    # Per-segment parameters
    a: list[float]  # Coefficients (first is free, rest from continuity)
    beta: list[float]  # Exponents (positive for rising, negative for decline)

    # Shared boundaries
    segment_boundaries: list[int]  # [0, T1, T2, ...]

    # Terminal decline
    terminal_decline_month: float = 0.0

    @property
    def num_segments(self) -> int:
        return len(self.a)


# =============================================================================
# Time Series Preparation
# =============================================================================


def prepare_sagd_series(
    df: pd.DataFrame,
    date_col: str = "production_date",
    oil_col: str = "oil",
    steam_col: str = "steam_volume",
    config: SAGDConfig | None = None,
) -> SAGDTimeSeries:
    """Convert calendar monthly SAGD data to normalized time series.

    Args:
        df: DataFrame with monthly production data
        date_col: Column name for production date
        oil_col: Column name for oil volume
        steam_col: Column name for steam volume
        config: SAGD configuration

    Returns:
        SAGDTimeSeries with normalized rates and time indices
    """
    if config is None:
        config = SAGDConfig()

    df = df.copy().sort_values(date_col)

    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Drop early heat-up months
    if config.drop_heatup_months > 0 and len(df) > config.drop_heatup_months:
        df = df.iloc[config.drop_heatup_months:].reset_index(drop=True)

    # Get days in each calendar month
    df["days_in_month"] = df[date_col].dt.days_in_month

    # Normalize rates: volume / days_in_month -> rate per day -> rate per normalized month
    df["qo_norm"] = (df[oil_col] / df["days_in_month"]) * NORMALIZED_MONTH_DAYS

    if steam_col in df.columns:
        df["steam_norm"] = (df[steam_col] / df["days_in_month"]) * NORMALIZED_MONTH_DAYS
    else:
        df["steam_norm"] = 0.0

    # Create normalized time index (0, 1, 2, ...)
    t_norm = np.arange(len(df))

    # Compute cumulative
    cum_oil = np.cumsum(df["qo_norm"].values)
    cum_steam = np.cumsum(df["steam_norm"].values)

    # Extract calendar dates
    calendar_dates = df[date_col].dt.date.tolist()
    calendar_start = calendar_dates[0] if calendar_dates else date.today()

    return SAGDTimeSeries(
        t_norm=t_norm,
        qo=df["qo_norm"].values,
        steam=df["steam_norm"].values,
        cum_oil=cum_oil,
        cum_steam=cum_steam,
        calendar_dates=calendar_dates,
        calendar_start=calendar_start,
    )


# =============================================================================
# Two-Stage Butler Model
# =============================================================================


def _butler_2stage_rate(
    t: np.ndarray,
    params: SAGDParams2Stage,
) -> np.ndarray:
    """Compute two-stage Butler SAGD rate.

    Rising stage: q(t) = A_r * t^alpha_r
    Spreading stage: q(t) = A_s * t^(-alpha_s)

    Args:
        t: Normalized time array
        params: Two-stage Butler parameters

    Returns:
        Oil rate array
    """
    t = np.asarray(t, dtype=float)
    t = np.maximum(t, 1e-6)  # Avoid t=0 issues

    qo = np.zeros_like(t)

    # Masks for each stage
    rising_mask = t <= params.t_tr
    spreading_mask = t > params.t_tr

    # Rising stage: q = A_r * t^alpha_r
    qo[rising_mask] = params.a_r * np.power(t[rising_mask], params.alpha_r)

    # Spreading stage: q = A_s * t^(-alpha_s)
    a_s = params.a_s
    qo[spreading_mask] = a_s * np.power(t[spreading_mask], -params.alpha_s)

    # Apply terminal exponential decline if enabled
    if params.terminal_decline_month > 0 and params.t_terminal < np.inf:
        terminal_mask = t > params.t_terminal
        if terminal_mask.any():
            # Rate at terminal start
            if params.t_terminal <= params.t_tr:
                q_terminal = params.a_r * np.power(params.t_terminal, params.alpha_r)
            else:
                q_terminal = a_s * np.power(params.t_terminal, -params.alpha_s)

            # Apply exponential decay
            dt = t[terminal_mask] - params.t_terminal
            qo[terminal_mask] = q_terminal * np.exp(-params.terminal_decline_month * dt)

    return qo


def _butler_2stage_objective(
    x: np.ndarray,
    t_obs: np.ndarray,
    qo_obs: np.ndarray,
    config: SAGDConfig,
) -> float:
    """Objective function for two-stage Butler fitting."""
    a_r, alpha_r, alpha_s, t_tr = x[:4]

    if config.use_terminal_decline:
        df_year = x[4]
        d_terminal = 1 - (1 - df_year) ** (1 / 12)
        # Terminal decline kicks in at 2x transition time or end of data
        t_terminal = max(t_tr * 2, t_obs.max() * 0.8)
    else:
        d_terminal = 0.0
        t_terminal = np.inf

    params = SAGDParams2Stage(
        a_r=a_r,
        alpha_r=alpha_r,
        alpha_s=alpha_s,
        t_tr=t_tr,
        terminal_decline_month=d_terminal,
        t_terminal=t_terminal,
    )

    # Compute predicted rates
    qo_pred = _butler_2stage_rate(t_obs, params)

    # Log-rate residuals
    mask = (qo_obs > 0) & (qo_pred > 0)
    if mask.sum() < 3:
        return 1e10

    log_obs = np.log(qo_obs[mask])
    log_pred = np.log(qo_pred[mask])

    residuals = log_obs - log_pred
    obj = np.sum(residuals**2)

    # Regularization
    # Penalize exponents far from 0.5 (classic Butler)
    reg = 0.1 * ((alpha_r - 0.5) ** 2 + (alpha_s - 0.5) ** 2)

    return obj + reg


def fit_sagd_2stage(
    ts: SAGDTimeSeries,
    config: SAGDConfig | None = None,
) -> SAGDParams2Stage:
    """Fit two-stage Butler SAGD model.

    Args:
        ts: Normalized SAGD time series
        config: SAGD configuration

    Returns:
        Fitted two-stage Butler parameters
    """
    if config is None:
        config = SAGDConfig()

    t_obs = ts.t_norm
    qo_obs = ts.qo

    # Filter valid observations
    mask = qo_obs > 0
    if mask.sum() < config.min_months_for_fit:
        raise ValueError(
            f"Insufficient data: {mask.sum()} months, need {config.min_months_for_fit}"
        )

    t_obs = t_obs[mask]
    qo_obs = qo_obs[mask]

    # Find approximate peak time for initial guess
    peak_idx = np.argmax(qo_obs)
    t_peak = float(t_obs[peak_idx]) if peak_idx > 0 else float(t_obs.max() / 3)
    t_peak = max(1.0, t_peak)  # Ensure positive

    q_peak = qo_obs[peak_idx]

    # Initial guesses
    alpha_r_init = 0.5
    alpha_s_init = 0.5
    a_r_init = q_peak / (t_peak ** alpha_r_init)
    t_tr_init = t_peak

    # Build x0
    x0 = [a_r_init, alpha_r_init, alpha_s_init, t_tr_init]
    if config.use_terminal_decline:
        x0.append(config.terminal_decline_year)

    # Bounds
    bounds = [
        (a_r_init * 0.01, a_r_init * 100),  # a_r
        config.alpha_r_bounds,  # alpha_r
        config.alpha_s_bounds,  # alpha_s
        (1.0, float(t_obs.max()) * 0.9),  # t_tr
    ]
    if config.use_terminal_decline:
        bounds.append(config.df_year_bounds)

    # Optimize
    if config.use_global_optimization:
        result = differential_evolution(
            _butler_2stage_objective,
            bounds,
            args=(t_obs, qo_obs, config),
            maxiter=200,
            seed=42,
        )
    else:
        result = minimize(
            _butler_2stage_objective,
            x0,
            args=(t_obs, qo_obs, config),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500},
        )

    # Extract parameters
    x = result.x
    a_r, alpha_r, alpha_s, t_tr = x[:4]

    if config.use_terminal_decline:
        df_year = x[4]
        d_terminal = 1 - (1 - df_year) ** (1 / 12)
        t_terminal = max(t_tr * 2, t_obs.max() * 0.8)
    else:
        d_terminal = 0.0
        t_terminal = np.inf

    return SAGDParams2Stage(
        a_r=a_r,
        alpha_r=alpha_r,
        alpha_s=alpha_s,
        t_tr=t_tr,
        terminal_decline_month=d_terminal,
        t_terminal=t_terminal,
    )


# =============================================================================
# Multi-Segment Butler Model
# =============================================================================


def _butler_multiseg_rate(
    t: np.ndarray,
    params: SAGDParamsMultiSeg,
) -> np.ndarray:
    """Compute multi-segment Butler SAGD rate.

    Each segment: q(t) = A_k * (t - T_{k-1})^beta_k

    Args:
        t: Normalized time array
        params: Multi-segment Butler parameters

    Returns:
        Oil rate array
    """
    t = np.asarray(t, dtype=float)
    qo = np.zeros_like(t)

    boundaries = params.segment_boundaries + [np.inf]

    for i, ti in enumerate(t):
        # Find segment
        seg_idx = 0
        for k in range(len(boundaries) - 1):
            if boundaries[k] <= ti < boundaries[k + 1]:
                seg_idx = k
                break

        # Segment-local time
        t_seg = ti - boundaries[seg_idx]
        t_seg = max(t_seg, 1e-6)  # Avoid zero

        # Rate
        a_k = params.a[seg_idx]
        beta_k = params.beta[seg_idx]
        qo[i] = a_k * np.power(t_seg, beta_k)

    # Apply terminal exponential decline for last segment
    if params.terminal_decline_month > 0:
        last_boundary = boundaries[-2]  # Last finite boundary
        terminal_mask = t > last_boundary
        if terminal_mask.any():
            # Rate at last boundary
            t_seg_end = 1.0  # Small positive time in last segment
            q_terminal = params.a[-1] * np.power(t_seg_end, params.beta[-1])

            # Apply exponential decay
            dt = t[terminal_mask] - last_boundary
            qo[terminal_mask] = q_terminal * np.exp(-params.terminal_decline_month * dt)

    return qo


def _butler_multiseg_objective(
    x: np.ndarray,
    t_obs: np.ndarray,
    qo_obs: np.ndarray,
    boundaries: list[int],
    config: SAGDConfig,
) -> float:
    """Objective function for multi-segment Butler fitting."""
    num_seg = len(boundaries)

    # Unpack: [a_0, beta_0, beta_1, ..., df_year]
    a_0 = x[0]
    betas = list(x[1:num_seg + 1])

    if config.use_terminal_decline:
        df_year = x[-1]
        d_terminal = 1 - (1 - df_year) ** (1 / 12)
    else:
        d_terminal = 0.0

    # Compute coefficients from continuity
    a_list = [a_0]
    for k in range(1, num_seg):
        # Rate at boundary from previous segment
        t_boundary = boundaries[k]
        t_seg_prev = t_boundary - boundaries[k - 1]
        t_seg_prev = max(t_seg_prev, 1e-6)

        q_at_boundary = a_list[k - 1] * np.power(t_seg_prev, betas[k - 1])

        # New segment coefficient: q = a_k * (0+epsilon)^beta_k = q_at_boundary
        # So a_k = q_at_boundary / epsilon^beta_k
        # For continuity at t=0 of segment, we use a small offset
        t_start_seg = 1.0  # 1 month into new segment
        a_k = q_at_boundary / np.power(t_start_seg, betas[k])
        a_list.append(a_k)

    params = SAGDParamsMultiSeg(
        a=a_list,
        beta=betas,
        segment_boundaries=boundaries,
        terminal_decline_month=d_terminal,
    )

    # Compute predicted rates
    qo_pred = _butler_multiseg_rate(t_obs, params)

    # Log-rate residuals
    mask = (qo_obs > 0) & (qo_pred > 0)
    if mask.sum() < 3:
        return 1e10

    log_obs = np.log(qo_obs[mask])
    log_pred = np.log(qo_pred[mask])

    residuals = log_obs - log_pred
    obj = np.sum(residuals**2)

    # Regularization: penalize extreme exponents
    reg = 0.05 * np.sum(np.array(betas) ** 2)

    return obj + reg


def fit_sagd_multiseg(
    ts: SAGDTimeSeries,
    config: SAGDConfig | None = None,
) -> SAGDParamsMultiSeg:
    """Fit multi-segment Butler SAGD model.

    Args:
        ts: Normalized SAGD time series
        config: SAGD configuration

    Returns:
        Fitted multi-segment Butler parameters
    """
    if config is None:
        config = SAGDConfig()

    t_obs = ts.t_norm
    qo_obs = ts.qo

    # Filter valid observations
    mask = qo_obs > 0
    if mask.sum() < config.min_months_for_fit:
        raise ValueError(
            f"Insufficient data: {mask.sum()} months, need {config.min_months_for_fit}"
        )

    t_obs = t_obs[mask]
    qo_obs = qo_obs[mask]

    num_seg = config.num_segments

    # Set up boundaries
    if config.segment_boundaries:
        boundaries = config.segment_boundaries[:num_seg]
    else:
        # Auto-generate: split time range
        max_t = int(t_obs.max())
        boundaries = [int(max_t * k / num_seg) for k in range(num_seg)]
        boundaries[0] = 0

    # Initial guesses
    peak_idx = np.argmax(qo_obs)
    q_peak = qo_obs[peak_idx]
    a_0_init = q_peak / 2  # Rough estimate

    # Initial betas: rising for first segment, declining for rest
    beta_init = [0.5]  # First segment rising
    for _ in range(1, num_seg):
        beta_init.append(-0.4)  # Subsequent segments declining

    # Build x0
    x0 = [a_0_init] + beta_init
    if config.use_terminal_decline:
        x0.append(config.terminal_decline_year)

    # Bounds
    bounds = [(a_0_init * 0.01, a_0_init * 100)]  # a_0

    for k in range(num_seg):
        if k == 0:
            bounds.append(config.beta_rising_bounds)  # First segment rising
        else:
            bounds.append(config.beta_decline_bounds)  # Subsequent declining

    if config.use_terminal_decline:
        bounds.append(config.df_year_bounds)

    # Optimize
    if config.use_global_optimization:
        result = differential_evolution(
            _butler_multiseg_objective,
            bounds,
            args=(t_obs, qo_obs, boundaries, config),
            maxiter=200,
            seed=42,
        )
    else:
        result = minimize(
            _butler_multiseg_objective,
            x0,
            args=(t_obs, qo_obs, boundaries, config),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500},
        )

    # Extract parameters
    x = result.x
    a_0 = x[0]
    betas = list(x[1:num_seg + 1])

    if config.use_terminal_decline:
        df_year = x[-1]
        d_terminal = 1 - (1 - df_year) ** (1 / 12)
    else:
        d_terminal = 0.0

    # Compute coefficients from continuity
    a_list = [a_0]
    for k in range(1, num_seg):
        t_boundary = boundaries[k]
        t_seg_prev = t_boundary - boundaries[k - 1]
        t_seg_prev = max(t_seg_prev, 1e-6)

        q_at_boundary = a_list[k - 1] * np.power(t_seg_prev, betas[k - 1])
        t_start_seg = 1.0
        a_k = q_at_boundary / np.power(t_start_seg, betas[k])
        a_list.append(a_k)

    return SAGDParamsMultiSeg(
        a=a_list,
        beta=betas,
        segment_boundaries=boundaries,
        terminal_decline_month=d_terminal,
    )


# =============================================================================
# Unified Fitting Interface
# =============================================================================


def fit_sagd_butler(
    ts: SAGDTimeSeries,
    config: SAGDConfig | None = None,
) -> SAGDParams2Stage | SAGDParamsMultiSeg:
    """Fit Butler SAGD model (auto-selects 2-stage or multi-segment).

    Args:
        ts: Normalized SAGD time series
        config: SAGD configuration

    Returns:
        Fitted Butler parameters (type depends on config.model)
    """
    if config is None:
        config = SAGDConfig()

    if config.model == "butler_2stage":
        return fit_sagd_2stage(ts, config)
    elif config.model == "butler_multiseg":
        return fit_sagd_multiseg(ts, config)
    else:
        raise ValueError(f"Unknown model: {config.model}")


# =============================================================================
# Forecasting
# =============================================================================


def sagd_rate(
    t_norm: np.ndarray,
    params: SAGDParams2Stage | SAGDParamsMultiSeg,
) -> np.ndarray:
    """Compute SAGD oil rate at normalized time points.

    Args:
        t_norm: Normalized month indices for forecast
        params: Fitted SAGD parameters

    Returns:
        Oil rate array
    """
    if isinstance(params, SAGDParams2Stage):
        return _butler_2stage_rate(t_norm, params)
    elif isinstance(params, SAGDParamsMultiSeg):
        return _butler_multiseg_rate(t_norm, params)
    else:
        raise TypeError(f"Unknown params type: {type(params)}")


def forecast_sagd(
    t_norm: np.ndarray,
    params: SAGDParams2Stage | SAGDParamsMultiSeg,
    calendar_start: date | None = None,
) -> pd.DataFrame:
    """Generate SAGD forecast DataFrame.

    Args:
        t_norm: Normalized month indices for forecast
        params: Fitted SAGD parameters
        calendar_start: Optional start date for calendar mapping

    Returns:
        DataFrame with t_norm, calendar_date, qo, cum_oil
    """
    qo = sagd_rate(t_norm, params)
    cum_oil = np.cumsum(qo)

    result = pd.DataFrame({
        "t_norm": t_norm,
        "oil_rate": qo,
        "cum_oil": cum_oil,
    })

    if calendar_start is not None:
        result["calendar_date"] = t_norm_to_calendar(t_norm, calendar_start)

    return result


def calculate_sagd_sor(
    ts: SAGDTimeSeries,
    params: SAGDParams2Stage | SAGDParamsMultiSeg,
) -> dict:
    """Calculate Steam-Oil Ratio (SOR) metrics.

    Args:
        ts: Time series with steam data
        params: Fitted SAGD parameters

    Returns:
        Dict with iSOR (instantaneous) and cSOR (cumulative) metrics
    """
    # Historical SOR
    mask = (ts.qo > 0) & (ts.steam > 0)
    if mask.sum() == 0:
        return {"isor_avg": np.nan, "csor": np.nan}

    isor = ts.steam[mask] / ts.qo[mask]
    isor_avg = np.mean(isor)

    # Cumulative SOR
    csor = ts.cum_steam[-1] / ts.cum_oil[-1] if ts.cum_oil[-1] > 0 else np.nan

    return {
        "isor_avg": isor_avg,
        "isor_min": np.min(isor),
        "isor_max": np.max(isor),
        "csor": csor,
    }


# =============================================================================
# Convenience Functions
# =============================================================================


def fit_and_forecast_sagd(
    df: pd.DataFrame,
    forecast_months: int = 240,
    date_col: str = "production_date",
    oil_col: str = "oil",
    steam_col: str = "steam_volume",
    config: SAGDConfig | None = None,
) -> tuple[SAGDParams2Stage | SAGDParamsMultiSeg, pd.DataFrame]:
    """Convenience function to fit and forecast SAGD in one call.

    Args:
        df: Monthly production DataFrame
        forecast_months: Number of months to forecast
        date_col: Column name for date
        oil_col: Column name for oil volume
        steam_col: Column name for steam volume
        config: SAGD configuration

    Returns:
        Tuple of (SAGDParams, forecast DataFrame)
    """
    if config is None:
        config = SAGDConfig()

    # Prepare time series
    ts = prepare_sagd_series(df, date_col, oil_col, steam_col, config)

    # Fit
    params = fit_sagd_butler(ts, config)

    # Forecast
    t_forecast = np.arange(0, forecast_months)
    forecast_df = forecast_sagd(t_forecast, params, ts.calendar_start)

    return params, forecast_df

