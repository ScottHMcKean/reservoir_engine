"""Decline curve models for reservoir engineering.

Implements Arps hyperbolic/harmonic decline for unconventional assets
and Butler SAGD model for thermal assets.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArpsParams:
    """Arps hyperbolic decline curve parameters.

    The Arps equation: q(t) = qi / (1 + b * di * t)^(1/b)

    For b=0: Exponential decline: q(t) = qi * exp(-di * t)
    For b=1: Harmonic decline: q(t) = qi / (1 + di * t)
    For 0<b<1: Hyperbolic decline

    Attributes:
        qi: Initial production rate (units/time, e.g., bbl/day)
        di: Initial nominal decline rate (1/time, e.g., 1/month)
        b: Hyperbolic exponent (0 <= b <= 2, typically 0-1 for oil)
        d_min: Minimum terminal decline rate for modified hyperbolic
    """

    qi: float
    di: float
    b: float
    d_min: float = 0.05  # 5% annual = ~0.004 monthly

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.qi <= 0:
            raise ValueError(f"qi must be positive, got {self.qi}")
        if self.di <= 0:
            raise ValueError(f"di must be positive, got {self.di}")
        if not 0 <= self.b <= 2:
            raise ValueError(f"b must be between 0 and 2, got {self.b}")

    @property
    def t_switch(self) -> float:
        """Calculate time to switch to exponential (modified hyperbolic).

        When instantaneous decline rate equals d_min, switch to exponential.
        """
        if self.b == 0 or self.di <= self.d_min:
            return float("inf")
        return (self.di / self.d_min - 1) / (self.b * self.di)


@dataclass(frozen=True)
class ButlerParams:
    """Butler SAGD (Steam Assisted Gravity Drainage) model parameters.

    The Butler-Stephens equation for SAGD oil rate:
    q = 2L * sqrt(k * g * phi * So * Δρ * α * h / (m * μo))

    Simplified for curve fitting: q(t) = q_peak * f(t, m, τ)

    Attributes:
        q_peak: Peak oil production rate
        m: Dimensionless rate constant (drainage efficiency)
        tau: Time constant (months to reach peak production)
        sor: Steam-oil ratio (cumulative)
        h: Pay thickness (meters)
        L: Well pair length (meters)
    """

    q_peak: float
    m: float
    tau: float
    sor: float = 3.0  # Typical SAGD SOR
    h: float = 20.0
    L: float = 800.0

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.q_peak <= 0:
            raise ValueError(f"q_peak must be positive, got {self.q_peak}")
        if self.m <= 0:
            raise ValueError(f"m must be positive, got {self.m}")
        if self.tau <= 0:
            raise ValueError(f"tau must be positive, got {self.tau}")


def arps_rate(t: float | np.ndarray, params: ArpsParams) -> float | np.ndarray:
    """Calculate production rate using Arps decline equation.

    Uses modified hyperbolic: switches to exponential when d(t) = d_min.

    Args:
        t: Time since production start (months)
        params: ArpsParams with qi, di, b, d_min

    Returns:
        Production rate at time t

    Example:
        >>> params = ArpsParams(qi=1000, di=0.15, b=0.8)
        >>> rate = arps_rate(12, params)  # Rate at 12 months
    """
    t = np.asarray(t)
    qi, di, b, d_min = params.qi, params.di, params.b, params.d_min
    t_switch = params.t_switch

    if b == 0:
        # Exponential decline
        return qi * np.exp(-di * t)

    # Compute rate at switch point for exponential tail
    # Guard against numerical issues with very small or negative base values
    base_at_switch = max(1 + b * di * t_switch, 1e-10)
    q_at_switch = qi / (base_at_switch ** (1 / b))

    # Compute rates element-wise to avoid np.where evaluating both branches
    scalar_input = np.ndim(t) == 0
    t_arr = np.atleast_1d(t)
    rate = np.empty(t_arr.shape, dtype=float)

    for i, ti in enumerate(t_arr.flat):
        if ti <= t_switch:
            # Hyperbolic: q(t) = qi / (1 + b * di * t)^(1/b)
            base = max(1 + b * di * ti, 1e-10)
            rate.flat[i] = qi / (base ** (1 / b))
        else:
            # Exponential after switch
            rate.flat[i] = q_at_switch * np.exp(-d_min * (ti - t_switch))

    return float(rate[0]) if scalar_input else rate


def arps_cumulative(t: float | np.ndarray, params: ArpsParams) -> float | np.ndarray:
    """Calculate cumulative production using Arps decline.

    Args:
        t: Time since production start (months)
        params: ArpsParams with qi, di, b

    Returns:
        Cumulative production up to time t
    """
    t = np.asarray(t)
    qi, di, b = params.qi, params.di, params.b

    if b == 0:
        # Exponential: Np = qi/di * (1 - exp(-di*t))
        return (qi / di) * (1 - np.exp(-di * t))
    elif b == 1:
        # Harmonic: Np = qi/di * ln(1 + di*t)
        return (qi / di) * np.log(1 + di * t)
    else:
        # Hyperbolic: Np = qi / ((1-b)*di) * (1 - (1+b*di*t)^(1-1/b))
        cum = (qi / ((1 - b) * di)) * (1 - np.power(1 + b * di * t, 1 - 1 / b))
        return float(cum) if cum.ndim == 0 else cum


def butler_rate(t: float | np.ndarray, params: ButlerParams) -> float | np.ndarray:
    """Calculate SAGD production rate using simplified Butler model.

    Uses a ramp-up to peak followed by decline function:
    q(t) = q_peak * (t/tau) * exp(1 - t/tau) for ramp-up phase
    q(t) = q_peak * exp(-m * (t-tau)/tau) for decline phase

    Args:
        t: Time since steam injection start (months)
        params: ButlerParams with q_peak, m, tau

    Returns:
        Production rate at time t

    Example:
        >>> params = ButlerParams(q_peak=500, m=0.1, tau=12)
        >>> rate = butler_rate(24, params)  # Rate at 24 months
    """
    t = np.asarray(t)
    q_peak, m, tau = params.q_peak, params.m, params.tau

    # Ramp-up phase (t <= tau): rate increases to peak
    # Decline phase (t > tau): exponential decline

    ramp_up = q_peak * (t / tau) * np.exp(1 - t / tau)
    decline = q_peak * np.exp(-m * (t - tau) / tau)

    rate = np.where(t <= tau, ramp_up, decline)

    return float(rate) if rate.ndim == 0 else rate


def butler_cumulative(t: float | np.ndarray, params: ButlerParams) -> float | np.ndarray:
    """Calculate cumulative SAGD production.

    Args:
        t: Time since steam injection start (months)
        params: ButlerParams

    Returns:
        Cumulative production up to time t
    """
    from scipy import integrate

    t = np.asarray(t)
    if t.ndim == 0:
        t_array = np.linspace(0, float(t), max(int(t) + 1, 100))
        rates = butler_rate(t_array, params)
        return float(integrate.trapezoid(rates, t_array))
    else:
        cumulative = np.zeros_like(t, dtype=float)
        for i, ti in enumerate(t):
            t_array = np.linspace(0, ti, max(int(ti) + 1, 100))
            rates = butler_rate(t_array, params)
            cumulative[i] = integrate.trapezoid(rates, t_array)
        return cumulative

