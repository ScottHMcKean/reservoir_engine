"""Probabilistic type curve generation.

Generate P10/P50/P90 type curves and perform Monte Carlo sampling
for uncertainty quantification.

Uses IHS column naming conventions.
"""

from typing import Literal

import numpy as np
import pandas as pd

from reservoir_engine.dca.autofit import fit_arps
from reservoir_engine.dca.forecast import arps_forecast
from reservoir_engine.dca.models import ArpsParams


def generate_type_curve(
    df: pd.DataFrame,
    wells: list[str] | None = None,
    percentiles: list[int] | None = None,
    rate_column: str = "oil",
    month_column: str = "month",
    uwi_column: str = "uwi",
    min_wells_per_month: int = 3,
) -> pd.DataFrame:
    """Generate probabilistic type curve from aligned production data.

    Args:
        df: DataFrame with aligned production data (must have 'month' column)
        wells: Optional list of wells to include
        percentiles: Percentile values to calculate (default: P10, P50, P90)
        rate_column: Name of rate column
        month_column: Name of month column
        uwi_column: Name of UWI column
        min_wells_per_month: Minimum wells required for statistics

    Returns:
        DataFrame with columns: month, p{percentile} for each percentile, well_count

    Example:
        >>> type_curve = generate_type_curve(aligned_df, percentiles=[10, 50, 90])
    """
    if percentiles is None:
        percentiles = [10, 50, 90]

    result_df = df.copy()

    if wells is not None:
        result_df = result_df[result_df[uwi_column].isin(wells)]

    monthly_data = []

    for month in sorted(result_df[month_column].unique()):
        month_df = result_df[result_df[month_column] == month]
        well_count = month_df[uwi_column].nunique()

        if well_count < min_wells_per_month:
            continue

        rates = month_df[rate_column].values

        row = {"month": month, "well_count": well_count}
        for pct in percentiles:
            row[f"p{pct}"] = np.percentile(rates, pct)

        monthly_data.append(row)

    return pd.DataFrame(monthly_data)


def fit_type_curve(
    type_curve_df: pd.DataFrame,
    percentile: int = 50,
    month_column: str = "month",
    forecast_months: int = 360,
) -> tuple[ArpsParams, pd.DataFrame]:
    """Fit Arps decline curve to a type curve percentile.

    Args:
        type_curve_df: DataFrame from generate_type_curve
        percentile: Which percentile to fit (10, 50, or 90)
        month_column: Name of month column
        forecast_months: Number of months to forecast

    Returns:
        Tuple of (ArpsParams, forecast DataFrame)

    Example:
        >>> type_curve = generate_type_curve(df)
        >>> params, forecast = fit_type_curve(type_curve, percentile=50)
    """
    rate_col = f"p{percentile}"

    if rate_col not in type_curve_df.columns:
        raise ValueError(f"Column {rate_col} not found in type curve DataFrame")

    # Prepare data for fitting
    fit_df = type_curve_df[[month_column, rate_col]].copy()
    fit_df = fit_df.rename(columns={rate_col: "oil"})
    fit_df["uwi"] = "TYPE_CURVE"
    fit_df["production_date"] = pd.date_range("2020-01-01", periods=len(fit_df), freq="MS")

    # Fit Arps model
    params = fit_arps(fit_df, uwi="TYPE_CURVE", rate_column="oil", method="auto")

    # Generate forecast
    forecast = arps_forecast(params, months=forecast_months)

    return params, forecast


def sample_type_curve(
    df: pd.DataFrame,
    n_samples: int = 1000,
    wells: list[str] | None = None,
    rate_column: str = "oil",
    month_column: str = "month",
    uwi_column: str = "uwi",
    method: Literal["bootstrap", "parametric"] = "bootstrap",
) -> pd.DataFrame:
    """Generate Monte Carlo samples from type curve distribution.

    Args:
        df: DataFrame with aligned production data
        n_samples: Number of Monte Carlo samples
        wells: Optional list of wells to include
        rate_column: Name of rate column
        month_column: Name of month column
        uwi_column: Name of UWI column
        method: Sampling method:
            - "bootstrap": Resample wells with replacement
            - "parametric": Sample from fitted distribution

    Returns:
        DataFrame with columns: sample_id, month, rate

    Example:
        >>> samples = sample_type_curve(aligned_df, n_samples=1000)
    """
    result_df = df.copy()

    if wells is not None:
        result_df = result_df[result_df[uwi_column].isin(wells)]

    unique_wells = result_df[uwi_column].unique()
    n_wells = len(unique_wells)

    all_samples = []

    if method == "bootstrap":
        for sample_id in range(n_samples):
            sampled_wells = np.random.choice(unique_wells, size=n_wells, replace=True)

            sample_data = []
            for i, uwi in enumerate(sampled_wells):
                well_data = result_df[result_df[uwi_column] == uwi].copy()
                well_data["sample_uwi"] = f"{uwi}_{i}"
                sample_data.append(well_data)

            sample_df = pd.concat(sample_data)
            monthly = sample_df.groupby(month_column)[rate_column].mean().reset_index()
            monthly["sample_id"] = sample_id
            all_samples.append(monthly)

    else:  # parametric
        for month in sorted(result_df[month_column].unique()):
            month_rates = result_df[result_df[month_column] == month][rate_column].values

            if len(month_rates) < 3:
                continue

            log_rates = np.log(month_rates[month_rates > 0])
            if len(log_rates) < 3:
                continue

            mu, sigma = np.mean(log_rates), np.std(log_rates)
            samples = np.exp(np.random.normal(mu, sigma, n_samples))

            for sample_id, rate in enumerate(samples):
                all_samples.append(
                    {
                        "sample_id": sample_id,
                        month_column: month,
                        rate_column: rate,
                    }
                )

    if method == "bootstrap":
        return pd.concat(all_samples, ignore_index=True)
    else:
        return pd.DataFrame(all_samples)


def calculate_eur_distribution(
    samples_df: pd.DataFrame,
    economic_limit: float,
    rate_column: str = "oil",
    month_column: str = "month",
) -> dict[str, float]:
    """Calculate EUR distribution from Monte Carlo samples.

    Args:
        samples_df: DataFrame from sample_type_curve
        economic_limit: Economic rate limit
        rate_column: Name of rate column
        month_column: Name of month column

    Returns:
        Dictionary with P10, P50, P90 EUR values

    Example:
        >>> samples = sample_type_curve(df, n_samples=1000)
        >>> eur_dist = calculate_eur_distribution(samples, economic_limit=10)
    """
    eur_values = []

    for sample_id in samples_df["sample_id"].unique():
        sample = samples_df[samples_df["sample_id"] == sample_id].sort_values(month_column)

        below_limit = sample[sample[rate_column] < economic_limit]

        if len(below_limit) > 0:
            cutoff_month = below_limit[month_column].iloc[0]
            sample_before_cutoff = sample[sample[month_column] < cutoff_month]
        else:
            sample_before_cutoff = sample

        eur = sample_before_cutoff[rate_column].sum()
        eur_values.append(eur)

    eur_array = np.array(eur_values)

    return {
        "p10": float(np.percentile(eur_array, 10)),
        "p50": float(np.percentile(eur_array, 50)),
        "p90": float(np.percentile(eur_array, 90)),
        "mean": float(np.mean(eur_array)),
        "std": float(np.std(eur_array)),
    }
