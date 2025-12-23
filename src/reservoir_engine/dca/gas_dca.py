"""Gas-first, calendar-native DCA engine.

Uses fixed-length normalized months, shared terminal decline,
multi-segment Arps behavior, and CGR-based condensate forecasts.

Calendar time stays in I/O and visualization; all fitting and
forecasting runs on normalized months.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# =============================================================================
# Constants
# =============================================================================

NORMALIZED_MONTH_DAYS = 30.4  # Fixed month length for normalization


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DCAConfig:
    """Configuration for gas-first DCA engine."""

    # Segment configuration
    num_segments: int = 1
    segment_boundaries: list[int] = field(default_factory=list)  # normalized month indices

    # Terminal decline (annual)
    terminal_decline_year: float = 0.06  # 6% annual terminal decline
    df_year_bounds: tuple[float, float] = (0.03, 0.12)

    # Hyperbolic exponent bounds
    b_bounds: tuple[float, float] = (0.0, 1.2)

    # Decline rate bounds (monthly)
    di_bounds: tuple[float, float] = (0.01, 0.50)

    # Fitting options
    drop_first_month: bool = True  # Drop partial first month
    min_months_for_fit: int = 6

    @property
    def terminal_decline_month(self) -> float:
        """Convert annual terminal decline to monthly."""
        return 1 - (1 - self.terminal_decline_year) ** (1 / 12)


@dataclass
class CGRConfig:
    """Configuration for CGR model."""

    model_type: Literal["log_linear", "power_law"] = "log_linear"
    min_gas_rate: float = 1.0  # Minimum gas rate for CGR calculation
    smoothing_window: int = 3  # Rolling window for CGR smoothing


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TimeSeries:
    """Normalized time series for DCA fitting."""

    t_norm: np.ndarray  # Normalized month indices (0, 1, 2, ...)
    qg: np.ndarray  # Gas rate per normalized month
    qc: np.ndarray  # Condensate rate per normalized month
    calendar_dates: list[date]  # Original calendar dates
    calendar_start: date  # Start date for calendar mapping


@dataclass
class GasParams:
    """Multi-segment gas decline parameters."""

    # Per-segment parameters (lists of length num_segments)
    qi: list[float]  # Initial rate at segment start
    di: list[float]  # Initial decline per month
    b: list[float]  # Hyperbolic exponent

    # Global parameters
    terminal_decline_month: float  # Shared terminal decline (monthly)
    segment_boundaries: list[int]  # Segment start months [0, T1, T2, ...]

    @property
    def num_segments(self) -> int:
        return len(self.qi)


@dataclass
class CGRParams:
    """CGR model parameters."""

    model_type: Literal["log_linear", "power_law"]
    alpha: float  # Intercept (log space for log_linear)
    beta: float  # Slope

    def cgr(self, qg: np.ndarray) -> np.ndarray:
        """Compute CGR from gas rate."""
        qg = np.maximum(qg, 1e-6)  # Avoid log(0)
        if self.model_type == "log_linear":
            # log(CGR) = alpha + beta * log(qg)
            return np.exp(self.alpha + self.beta * np.log(qg))
        else:  # power_law
            # CGR = alpha * qg^beta
            return self.alpha * np.power(qg, self.beta)


# =============================================================================
# Time Handling
# =============================================================================


def prepare_time_series(
    df: pd.DataFrame,
    date_col: str = "production_date",
    gas_col: str = "gas",
    cond_col: str = "condensate",
    config: DCAConfig | None = None,
) -> TimeSeries:
    """Convert calendar monthly data to normalized time series.

    Args:
        df: DataFrame with monthly production data
        date_col: Column name for production date
        gas_col: Column name for gas volume
        cond_col: Column name for condensate volume
        config: DCA configuration

    Returns:
        TimeSeries with normalized rates and time indices
    """
    if config is None:
        config = DCAConfig()

    df = df.copy().sort_values(date_col)

    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Drop first month if configured (often partial)
    if config.drop_first_month and len(df) > 1:
        df = df.iloc[1:].reset_index(drop=True)

    # Get days in each calendar month
    df["days_in_month"] = df[date_col].dt.days_in_month

    # Normalize rates: volume / days_in_month -> rate per day -> rate per normalized month
    df["qg_norm"] = (df[gas_col] / df["days_in_month"]) * NORMALIZED_MONTH_DAYS

    if cond_col in df.columns:
        df["qc_norm"] = (df[cond_col] / df["days_in_month"]) * NORMALIZED_MONTH_DAYS
    else:
        df["qc_norm"] = 0.0

    # Create normalized time index (0, 1, 2, ...)
    t_norm = np.arange(len(df))

    # Extract calendar dates
    calendar_dates = df[date_col].dt.date.tolist()
    calendar_start = calendar_dates[0] if calendar_dates else date.today()

    return TimeSeries(
        t_norm=t_norm,
        qg=df["qg_norm"].values,
        qc=df["qc_norm"].values,
        calendar_dates=calendar_dates,
        calendar_start=calendar_start,
    )


def t_norm_to_calendar(
    t_norm: np.ndarray,
    calendar_start: date,
) -> list[date]:
    """Map normalized time indices to calendar dates.

    Args:
        t_norm: Normalized month indices
        calendar_start: Start date for mapping

    Returns:
        List of calendar dates (first of each month)
    """
    from dateutil.relativedelta import relativedelta

    start = pd.Timestamp(calendar_start)
    dates = []
    for t in t_norm:
        month_offset = int(t)
        new_date = start + relativedelta(months=month_offset)
        dates.append(new_date.date())
    return dates


def calendar_to_t_norm(
    dates: list[date],
    calendar_start: date,
) -> np.ndarray:
    """Map calendar dates to normalized time indices.

    Args:
        dates: List of calendar dates
        calendar_start: Reference start date

    Returns:
        Array of normalized month indices
    """
    start = pd.Timestamp(calendar_start)
    t_norm = []
    for d in dates:
        dt = pd.Timestamp(d)
        months = (dt.year - start.year) * 12 + (dt.month - start.month)
        t_norm.append(months)
    return np.array(t_norm)


# =============================================================================
# Gas Decline Model
# =============================================================================


def _hyperbolic_rate(
    t: np.ndarray,
    qi: float,
    di: float,
    b: float,
) -> np.ndarray:
    """Hyperbolic decline rate.

    q(t) = qi / (1 + b * di * t)^(1/b)

    For b=0, uses exponential: q(t) = qi * exp(-di * t)
    """
    t = np.asarray(t, dtype=float)

    if abs(b) < 1e-6:
        # Exponential decline
        return qi * np.exp(-di * t)
    else:
        # Hyperbolic decline
        denom = np.power(1 + b * di * t, 1 / b)
        return qi / denom


def _instantaneous_decline(
    t: np.ndarray,
    di: float,
    b: float,
) -> np.ndarray:
    """Instantaneous decline rate D(t) for hyperbolic.

    D(t) = di / (1 + b * di * t)
    """
    t = np.asarray(t, dtype=float)

    if abs(b) < 1e-6:
        return np.full_like(t, di)
    else:
        return di / (1 + b * di * t)


def gas_rate(
    t_norm: np.ndarray,
    params: GasParams,
) -> np.ndarray:
    """Compute gas rate at normalized time points.

    Handles multi-segment Arps with shared terminal decline.

    Args:
        t_norm: Normalized month indices
        params: Gas decline parameters

    Returns:
        Gas rate array
    """
    t_norm = np.asarray(t_norm, dtype=float)
    qg = np.zeros_like(t_norm)

    boundaries = params.segment_boundaries + [np.inf]  # Add infinity as final boundary

    for i, t in enumerate(t_norm):
        # Find which segment this time point belongs to
        seg_idx = 0
        for k in range(len(boundaries) - 1):
            if boundaries[k] <= t < boundaries[k + 1]:
                seg_idx = k
                break

        # Segment-local time
        t_seg = t - boundaries[seg_idx]

        # Get segment parameters
        qi = params.qi[seg_idx]
        di = params.di[seg_idx]
        b = params.b[seg_idx]
        d_terminal = params.terminal_decline_month

        # Check if we should switch to terminal exponential
        d_inst = _instantaneous_decline(np.array([t_seg]), di, b)[0]

        if d_inst <= d_terminal:
            # Switch to terminal exponential
            # Find switch time
            if abs(b) < 1e-6:
                # Already exponential, just use terminal
                t_switch = 0
            else:
                # Solve: di / (1 + b * di * t_switch) = d_terminal
                t_switch = (di / d_terminal - 1) / (b * di)
                t_switch = max(0, t_switch)

            # Rate at switch point
            q_switch = _hyperbolic_rate(np.array([t_switch]), qi, di, b)[0]

            # Exponential from switch point
            qg[i] = q_switch * np.exp(-d_terminal * (t_seg - t_switch))
        else:
            # Still in hyperbolic phase
            qg[i] = _hyperbolic_rate(np.array([t_seg]), qi, di, b)[0]

    return qg


def gas_rate_vectorized(
    t_norm: np.ndarray,
    params: GasParams,
) -> np.ndarray:
    """Vectorized gas rate computation (faster for forecasting).

    Single-segment optimized version.
    """
    if params.num_segments == 1:
        qi = params.qi[0]
        di = params.di[0]
        b = params.b[0]
        d_terminal = params.terminal_decline_month

        # Find switch time to terminal decline
        if abs(b) < 1e-6 or di <= d_terminal:
            t_switch = 0.0
            q_switch = qi
        else:
            t_switch = (di / d_terminal - 1) / (b * di)
            t_switch = max(0, t_switch)
            q_switch = _hyperbolic_rate(np.array([t_switch]), qi, di, b)[0]

        # Compute rates
        qg = np.where(
            t_norm <= t_switch,
            _hyperbolic_rate(t_norm, qi, di, b),
            q_switch * np.exp(-d_terminal * (t_norm - t_switch)),
        )
        return qg
    else:
        # Fall back to loop for multi-segment
        return gas_rate(t_norm, params)


# =============================================================================
# Gas Fitting
# =============================================================================


def _gas_objective(
    x: np.ndarray,
    t_obs: np.ndarray,
    qg_obs: np.ndarray,
    config: DCAConfig,
) -> float:
    """Objective function for gas fitting (log-rate least squares)."""
    num_seg = config.num_segments

    # Unpack parameters
    # x = [qi_0, di_0, b_0, di_1, b_1, ..., df_year]
    qi_0 = x[0]
    df_year = x[-1]

    # Build GasParams
    qi_list = [qi_0]
    di_list = []
    b_list = []

    idx = 1
    for _ in range(num_seg):
        di_list.append(x[idx])
        b_list.append(x[idx + 1])
        idx += 2

    # Compute qi for segments > 0 from continuity
    boundaries = config.segment_boundaries if config.segment_boundaries else [0]
    if len(boundaries) < num_seg:
        # Auto-generate boundaries if not specified
        max_t = int(t_obs.max()) if len(t_obs) > 0 else 60
        boundaries = [0] + [int(max_t * (k + 1) / num_seg) for k in range(num_seg - 1)]

    d_terminal = 1 - (1 - df_year) ** (1 / 12)

    for k in range(1, num_seg):
        # Rate at end of previous segment
        t_boundary = boundaries[k]
        t_seg_prev = t_boundary - boundaries[k - 1]

        # Rate at boundary from previous segment
        q_prev = _hyperbolic_rate(
            np.array([t_seg_prev]), qi_list[k - 1], di_list[k - 1], b_list[k - 1]
        )[0]
        qi_list.append(q_prev)

    params = GasParams(
        qi=qi_list,
        di=di_list,
        b=b_list,
        terminal_decline_month=d_terminal,
        segment_boundaries=boundaries,
    )

    # Compute predicted rates
    qg_pred = gas_rate(t_obs, params)

    # Log-rate residuals (filter out zeros)
    mask = (qg_obs > 0) & (qg_pred > 0)
    if mask.sum() < 3:
        return 1e10

    log_obs = np.log(qg_obs[mask])
    log_pred = np.log(qg_pred[mask])

    residuals = log_obs - log_pred
    obj = np.sum(residuals**2)

    # Regularization: penalize very high initial decline
    reg = 0.01 * np.sum(np.array(di_list) ** 2)

    return obj + reg


def fit_gas(
    ts: TimeSeries,
    config: DCAConfig | None = None,
) -> GasParams:
    """Fit multi-segment gas decline model.

    Args:
        ts: Normalized time series
        config: DCA configuration

    Returns:
        Fitted gas parameters
    """
    if config is None:
        config = DCAConfig()

    t_obs = ts.t_norm
    qg_obs = ts.qg

    # Filter valid observations
    mask = qg_obs > 0
    if mask.sum() < config.min_months_for_fit:
        raise ValueError(
            f"Insufficient data: {mask.sum()} months, need {config.min_months_for_fit}"
        )

    t_obs = t_obs[mask]
    qg_obs = qg_obs[mask]

    num_seg = config.num_segments

    # Initial guess
    qi_init = float(qg_obs[0]) if len(qg_obs) > 0 else 100.0
    di_init = 0.10  # 10% monthly decline
    b_init = 0.5
    df_year_init = config.terminal_decline_year

    # x = [qi_0, di_0, b_0, di_1, b_1, ..., df_year]
    x0 = [qi_init]
    for _ in range(num_seg):
        x0.extend([di_init, b_init])
    x0.append(df_year_init)

    # Bounds
    bounds = [(qi_init * 0.1, qi_init * 10)]  # qi_0
    for _ in range(num_seg):
        bounds.append(config.di_bounds)  # di
        bounds.append(config.b_bounds)  # b
    bounds.append(config.df_year_bounds)  # df_year

    # Optimize
    result = minimize(
        _gas_objective,
        x0,
        args=(t_obs, qg_obs, config),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500},
    )

    # Extract parameters
    x = result.x
    qi_0 = x[0]
    df_year = x[-1]
    d_terminal = 1 - (1 - df_year) ** (1 / 12)

    qi_list = [qi_0]
    di_list = []
    b_list = []

    idx = 1
    for _ in range(num_seg):
        di_list.append(x[idx])
        b_list.append(x[idx + 1])
        idx += 2

    # Compute qi for segments > 0
    boundaries = config.segment_boundaries if config.segment_boundaries else [0]
    if len(boundaries) < num_seg:
        max_t = int(t_obs.max())
        boundaries = [0] + [int(max_t * (k + 1) / num_seg) for k in range(num_seg - 1)]

    for k in range(1, num_seg):
        t_boundary = boundaries[k]
        t_seg_prev = t_boundary - boundaries[k - 1]
        q_prev = _hyperbolic_rate(
            np.array([t_seg_prev]), qi_list[k - 1], di_list[k - 1], b_list[k - 1]
        )[0]
        qi_list.append(q_prev)

    return GasParams(
        qi=qi_list,
        di=di_list,
        b=b_list,
        terminal_decline_month=d_terminal,
        segment_boundaries=boundaries,
    )


# =============================================================================
# CGR Model
# =============================================================================


def fit_cgr(
    ts: TimeSeries,
    gas_params: GasParams,
    config: CGRConfig | None = None,
) -> CGRParams:
    """Fit CGR model from historical data.

    Args:
        ts: Normalized time series
        gas_params: Fitted gas parameters (for rate reference)
        config: CGR configuration

    Returns:
        Fitted CGR parameters
    """
    if config is None:
        config = CGRConfig()

    qg = ts.qg
    qc = ts.qc

    # Filter valid points
    mask = (qg > config.min_gas_rate) & (qc > 0)
    if mask.sum() < 3:
        # Default to constant CGR if insufficient data
        mean_cgr = np.mean(qc[qc > 0] / qg[qc > 0]) if (qc > 0).sum() > 0 else 0.01
        return CGRParams(
            model_type=config.model_type,
            alpha=np.log(mean_cgr) if config.model_type == "log_linear" else mean_cgr,
            beta=0.0,
        )

    qg_valid = qg[mask]
    qc_valid = qc[mask]
    cgr_obs = qc_valid / qg_valid

    # Smooth CGR if configured
    if config.smoothing_window > 1 and len(cgr_obs) >= config.smoothing_window:
        cgr_obs = pd.Series(cgr_obs).rolling(config.smoothing_window, center=True).mean().bfill().ffill().values

    # Fit log-linear: log(CGR) = alpha + beta * log(qg)
    log_cgr = np.log(cgr_obs)
    log_qg = np.log(qg_valid)

    # OLS fit
    X = np.column_stack([np.ones(len(log_qg)), log_qg])
    coeffs, _, _, _ = np.linalg.lstsq(X, log_cgr, rcond=None)

    alpha = coeffs[0]
    beta = coeffs[1]

    return CGRParams(
        model_type=config.model_type,
        alpha=alpha,
        beta=beta,
    )


# =============================================================================
# Forecasting
# =============================================================================


def forecast_gas(
    t_norm: np.ndarray,
    gas_params: GasParams,
) -> np.ndarray:
    """Forecast gas rates at normalized time points.

    Args:
        t_norm: Normalized month indices for forecast
        gas_params: Fitted gas parameters

    Returns:
        Forecasted gas rates
    """
    return gas_rate_vectorized(t_norm, gas_params)


def forecast_condensate(
    t_norm: np.ndarray,
    gas_params: GasParams,
    cgr_params: CGRParams,
) -> np.ndarray:
    """Forecast condensate rates from gas + CGR.

    Args:
        t_norm: Normalized month indices for forecast
        gas_params: Fitted gas parameters
        cgr_params: Fitted CGR parameters

    Returns:
        Forecasted condensate rates
    """
    qg = forecast_gas(t_norm, gas_params)
    cgr = cgr_params.cgr(qg)
    return cgr * qg


def forecast_all(
    t_norm: np.ndarray,
    gas_params: GasParams,
    cgr_params: CGRParams,
    calendar_start: date | None = None,
) -> pd.DataFrame:
    """Generate complete forecast DataFrame.

    Args:
        t_norm: Normalized month indices for forecast
        gas_params: Fitted gas parameters
        cgr_params: Fitted CGR parameters
        calendar_start: Optional start date for calendar mapping

    Returns:
        DataFrame with t_norm, calendar_date (if provided), qg, qc, cgr, cum_gas, cum_cond
    """
    qg = forecast_gas(t_norm, gas_params)
    cgr = cgr_params.cgr(qg)
    qc = cgr * qg

    # Cumulative (trapezoidal integration approximation)
    cum_gas = np.cumsum(qg)
    cum_cond = np.cumsum(qc)

    result = pd.DataFrame({
        "t_norm": t_norm,
        "gas_rate": qg,
        "condensate_rate": qc,
        "cgr": cgr,
        "cum_gas": cum_gas,
        "cum_condensate": cum_cond,
    })

    if calendar_start is not None:
        result["calendar_date"] = t_norm_to_calendar(t_norm, calendar_start)

    return result


# =============================================================================
# Convenience Functions
# =============================================================================


def fit_and_forecast(
    df: pd.DataFrame,
    forecast_months: int = 360,
    date_col: str = "production_date",
    gas_col: str = "gas",
    cond_col: str = "condensate",
    dca_config: DCAConfig | None = None,
    cgr_config: CGRConfig | None = None,
) -> tuple[GasParams, CGRParams, pd.DataFrame]:
    """Convenience function to fit and forecast in one call.

    Args:
        df: Monthly production DataFrame
        forecast_months: Number of months to forecast
        date_col: Column name for date
        gas_col: Column name for gas volume
        cond_col: Column name for condensate volume
        dca_config: DCA configuration
        cgr_config: CGR configuration

    Returns:
        Tuple of (GasParams, CGRParams, forecast DataFrame)
    """
    if dca_config is None:
        dca_config = DCAConfig()
    if cgr_config is None:
        cgr_config = CGRConfig()

    # Prepare time series
    ts = prepare_time_series(df, date_col, gas_col, cond_col, dca_config)

    # Fit gas
    gas_params = fit_gas(ts, dca_config)

    # Fit CGR
    cgr_params = fit_cgr(ts, gas_params, cgr_config)

    # Forecast
    t_forecast = np.arange(0, forecast_months)
    forecast_df = forecast_all(t_forecast, gas_params, cgr_params, ts.calendar_start)

    return gas_params, cgr_params, forecast_df

