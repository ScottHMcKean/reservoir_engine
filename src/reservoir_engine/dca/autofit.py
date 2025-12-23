"""Automatic curve fitting for decline curve analysis.

Uses scipy.optimize for robust fitting of Arps and Butler models
to historical production data.
"""

from typing import Literal

import numpy as np
import pandas as pd
from scipy import optimize

from reservoir_engine.dca.models import ArpsParams, ButlerParams, arps_rate, butler_rate


def fit_arps(
    df: pd.DataFrame,
    uwi: str,
    rate_column: str = "oil",
    date_column: str = "production_date",
    method: Literal["hyperbolic", "harmonic", "exponential", "auto"] = "auto",
    min_months: int = 6,
) -> ArpsParams:
    """Fit Arps decline curve to production data.

    Args:
        df: DataFrame with production data
        uwi: Well identifier to fit
        rate_column: Column containing production rates
        date_column: Column containing dates
        method: Decline type to fit:
            - "hyperbolic": Fit hyperbolic (0 < b < 1)
            - "harmonic": Fix b=1
            - "exponential": Fix b=0
            - "auto": Try all and select best fit
        min_months: Minimum months of data required

    Returns:
        ArpsParams with fitted qi, di, b values

    Raises:
        ValueError: If insufficient data for fitting

    Example:
        >>> params = fit_arps(production_df, "WELL-001", method="auto")
        >>> print(f"qi={params.qi:.1f}, di={params.di:.3f}, b={params.b:.2f}")
    """
    # Filter and prepare data
    well_df = df[df["uwi"] == uwi].copy()
    well_df = well_df.sort_values(date_column)

    if len(well_df) < min_months:
        raise ValueError(f"Insufficient data: {len(well_df)} months, need {min_months}")

    # Calculate time in months from first production
    well_df["t"] = np.arange(len(well_df))
    rates = well_df[rate_column].values
    t = well_df["t"].values

    # Filter out zeros and very low values
    mask = rates > 0.1 * np.max(rates)
    t_fit = t[mask]
    rates_fit = rates[mask]

    if len(rates_fit) < min_months:
        raise ValueError("Insufficient non-zero data for fitting")

    def objective(params_array: np.ndarray, b_fixed: float | None = None) -> float:
        """Objective function for optimization."""
        qi, di = params_array[0], params_array[1]
        b = b_fixed if b_fixed is not None else params_array[2]

        if qi <= 0 or di <= 0 or b < 0 or b > 2:
            return 1e10

        try:
            arps_params = ArpsParams(qi=qi, di=di, b=b)
            predicted = arps_rate(t_fit, arps_params)
            residuals = np.log(rates_fit + 1) - np.log(predicted + 1)
            return float(np.sum(residuals**2))
        except (ValueError, RuntimeWarning):
            return 1e10

    # Initial guesses
    qi_init = float(np.max(rates_fit))
    di_init = 0.1  # 10% per month initial guess

    if method == "exponential":
        result = optimize.minimize(
            lambda x: objective(x, b_fixed=0.0),
            x0=[qi_init, di_init],
            bounds=[(qi_init * 0.5, qi_init * 2), (0.001, 1.0)],
            method="L-BFGS-B",
        )
        return ArpsParams(qi=result.x[0], di=result.x[1], b=0.0)

    elif method == "harmonic":
        result = optimize.minimize(
            lambda x: objective(x, b_fixed=1.0),
            x0=[qi_init, di_init],
            bounds=[(qi_init * 0.5, qi_init * 2), (0.001, 1.0)],
            method="L-BFGS-B",
        )
        return ArpsParams(qi=result.x[0], di=result.x[1], b=1.0)

    elif method == "hyperbolic":
        result = optimize.minimize(
            objective,
            x0=[qi_init, di_init, 0.5],
            bounds=[(qi_init * 0.5, qi_init * 2), (0.001, 1.0), (0.01, 1.5)],
            method="L-BFGS-B",
        )
        return ArpsParams(qi=result.x[0], di=result.x[1], b=result.x[2])

    else:  # auto
        # Try all three and pick best
        results = []

        for _method_name, b_val in [("exponential", 0.0), ("harmonic", 1.0)]:
            try:
                result = optimize.minimize(
                    lambda x, b=b_val: objective(x, b_fixed=b),
                    x0=[qi_init, di_init],
                    bounds=[(qi_init * 0.5, qi_init * 2), (0.001, 1.0)],
                    method="L-BFGS-B",
                )
                results.append((result.fun, result.x[0], result.x[1], b_val))
            except Exception:
                pass

        # Try hyperbolic
        try:
            result = optimize.minimize(
                objective,
                x0=[qi_init, di_init, 0.5],
                bounds=[(qi_init * 0.5, qi_init * 2), (0.001, 1.0), (0.01, 1.5)],
                method="L-BFGS-B",
            )
            results.append((result.fun, result.x[0], result.x[1], result.x[2]))
        except Exception:
            pass

        if not results:
            raise ValueError("All fitting methods failed")

        # Select best fit (lowest residual)
        best = min(results, key=lambda x: x[0])
        return ArpsParams(qi=best[1], di=best[2], b=best[3])


def fit_butler(
    production_df: pd.DataFrame,
    steam_df: pd.DataFrame | None = None,
    uwi: str = "",
    rate_column: str = "oil",
    date_column: str = "production_date",
    min_months: int = 12,
) -> ButlerParams:
    """Fit Butler SAGD model to thermal production data.

    Args:
        production_df: DataFrame with production data
        steam_df: Optional DataFrame with steam injection data
        uwi: Well identifier to fit
        rate_column: Column containing production rates
        date_column: Column containing dates
        min_months: Minimum months of data required

    Returns:
        ButlerParams with fitted parameters

    Raises:
        ValueError: If insufficient data for fitting

    Example:
        >>> params = fit_butler(production_df, steam_df, uwi="PAD-01")
    """
    # Filter and prepare data
    well_df = production_df[production_df["uwi"] == uwi].copy()
    well_df = well_df.sort_values(date_column)

    if len(well_df) < min_months:
        raise ValueError(f"Insufficient data: {len(well_df)} months, need {min_months}")

    # Calculate time in months from first production
    well_df["t"] = np.arange(len(well_df))
    rates = well_df[rate_column].values
    t = well_df["t"].values

    # Initial estimates
    q_peak_init = float(np.max(rates))
    peak_idx = np.argmax(rates)
    tau_init = float(max(peak_idx, 1))

    def objective(params_array: np.ndarray) -> float:
        """Objective function for Butler model."""
        q_peak, m, tau = params_array

        if q_peak <= 0 or m <= 0 or tau <= 0:
            return 1e10

        try:
            butler_params = ButlerParams(q_peak=q_peak, m=m, tau=tau)
            predicted = butler_rate(t, butler_params)
            residuals = rates - predicted
            return float(np.sum(residuals**2))
        except (ValueError, RuntimeWarning):
            return 1e10

    result = optimize.minimize(
        objective,
        x0=[q_peak_init, 0.1, tau_init],
        bounds=[
            (q_peak_init * 0.5, q_peak_init * 2),
            (0.01, 1.0),
            (1, max(len(t) / 2, 2)),
        ],
        method="L-BFGS-B",
    )

    # Calculate SOR if steam data provided
    sor = 3.0  # Default
    if steam_df is not None and uwi:
        steam_well = steam_df[steam_df["uwi"] == uwi]
        if len(steam_well) > 0 and "steam_tonnes" in steam_well.columns:
            total_steam = steam_well["steam_tonnes"].sum()
            total_oil = well_df[rate_column].sum()
            if total_oil > 0:
                sor = total_steam / total_oil

    return ButlerParams(
        q_peak=result.x[0],
        m=result.x[1],
        tau=result.x[2],
        sor=sor,
    )

