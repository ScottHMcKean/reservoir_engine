"""Statistical analysis functions for reservoir data.

Pure functions for calculating statistics, detecting outliers,
and summarizing production data.
"""

from typing import Literal

import numpy as np
import pandas as pd


def calculate_statistics(
    df: pd.DataFrame,
    value_column: str = "oil_bbl",
    group_by: str | list[str] | None = None,
) -> pd.DataFrame:
    """Calculate descriptive statistics for production data.

    Args:
        df: DataFrame with production data
        value_column: Column to calculate statistics for
        group_by: Optional column(s) to group by (e.g., "uwi", "field")

    Returns:
        DataFrame with statistics (count, mean, std, min, p25, p50, p75, max)

    Example:
        >>> stats = calculate_statistics(production_df, "oil_bbl", group_by="uwi")
    """
    if group_by is None:
        stats = df[value_column].describe()
        return pd.DataFrame(stats).T
    else:
        return df.groupby(group_by)[value_column].describe().reset_index()


def detect_outliers(
    df: pd.DataFrame,
    value_column: str = "oil_bbl",
    method: Literal["iqr", "zscore", "mad"] = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Detect outliers in production data.

    Args:
        df: DataFrame with production data
        value_column: Column to check for outliers
        method: Detection method - "iqr" (interquartile range),
                "zscore" (standard deviations), or "mad" (median absolute deviation)
        threshold: Threshold for outlier detection
                   - IQR: multiplier for IQR (default 1.5)
                   - Z-score: number of standard deviations (default 1.5 ~ 93%)
                   - MAD: multiplier for MAD (default 1.5)

    Returns:
        Original DataFrame with added "is_outlier" boolean column

    Example:
        >>> flagged_df = detect_outliers(production_df, "oil_bbl", method="iqr")
        >>> outliers = flagged_df[flagged_df["is_outlier"]]
    """
    df = df.copy()
    values = df[value_column].dropna()

    if method == "iqr":
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        df["is_outlier"] = (df[value_column] < lower_bound) | (df[value_column] > upper_bound)

    elif method == "zscore":
        mean = values.mean()
        std = values.std()
        if std == 0:
            df["is_outlier"] = False
        else:
            z_scores = np.abs((df[value_column] - mean) / std)
            df["is_outlier"] = z_scores > threshold

    elif method == "mad":
        median = values.median()
        mad = np.median(np.abs(values - median))
        if mad == 0:
            df["is_outlier"] = False
        else:
            modified_z = 0.6745 * np.abs(df[value_column] - median) / mad
            df["is_outlier"] = modified_z > threshold

    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr', 'zscore', or 'mad'")

    return df


def production_summary(
    df: pd.DataFrame,
    group_by: str = "uwi",
    date_column: str = "date",
    oil_column: str = "oil_bbl",
    gas_column: str = "gas_mcf",
    water_column: str = "water_bbl",
) -> pd.DataFrame:
    """Generate production summary statistics by well or group.

    Args:
        df: DataFrame with production data
        group_by: Column to group by (default: uwi)
        date_column: Name of date column
        oil_column: Name of oil production column
        gas_column: Name of gas production column
        water_column: Name of water production column

    Returns:
        DataFrame with summary statistics per group:
        - first_production: First production date
        - last_production: Last production date
        - months_on: Number of producing months
        - cum_oil: Cumulative oil (bbl)
        - cum_gas: Cumulative gas (mcf)
        - cum_water: Cumulative water (bbl)
        - peak_oil_rate: Maximum monthly oil rate
        - avg_oil_rate: Average monthly oil rate
    """
    summary = df.groupby(group_by).agg(
        first_production=(date_column, "min"),
        last_production=(date_column, "max"),
        months_on=(date_column, "count"),
        cum_oil=(oil_column, "sum"),
        cum_gas=(gas_column, "sum"),
        cum_water=(water_column, "sum"),
        peak_oil_rate=(oil_column, "max"),
        avg_oil_rate=(oil_column, "mean"),
    ).reset_index()

    return summary


def calculate_decline_rate(
    df: pd.DataFrame,
    uwi: str,
    date_column: str = "date",
    rate_column: str = "oil_bbl",
    window: int = 3,
) -> pd.DataFrame:
    """Calculate month-over-month decline rates for a well.

    Args:
        df: DataFrame with production data
        uwi: Well identifier to filter
        date_column: Name of date column
        rate_column: Name of rate column
        window: Rolling window for smoothing (months)

    Returns:
        DataFrame with date, rate, smoothed_rate, and decline_rate columns
    """
    well_df = df[df["uwi"] == uwi].copy()
    well_df = well_df.sort_values(date_column)

    # Calculate smoothed rate
    well_df["smoothed_rate"] = well_df[rate_column].rolling(window=window, min_periods=1).mean()

    # Calculate decline rate (month over month)
    well_df["decline_rate"] = well_df["smoothed_rate"].pct_change() * -1

    return well_df[[date_column, rate_column, "smoothed_rate", "decline_rate"]]

