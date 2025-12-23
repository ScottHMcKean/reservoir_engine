"""Production aggregation and normalization for type curves.

Functions for aligning, normalizing, and aggregating production
data from multiple wells for type curve generation.

Uses IHS column naming conventions (oil, gas, water in metric units).
"""

from typing import Literal

import pandas as pd


def normalize_production(
    df: pd.DataFrame,
    wells: list[str] | None = None,
    normalize_by: Literal["peak", "first_month", "lateral_length", "custom"] = "peak",
    rate_column: str = "oil",
    uwi_column: str = "uwi",
    normalization_values: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Normalize production rates for comparison across wells.

    Args:
        df: DataFrame with production data
        wells: Optional list of wells to include
        normalize_by: Normalization method:
            - "peak": Divide by peak rate
            - "first_month": Divide by first month rate
            - "lateral_length": Divide by lateral length (requires values)
            - "custom": Use provided normalization_values
        rate_column: Name of rate column to normalize
        uwi_column: Name of UWI column
        normalization_values: Dict of {uwi: value} for lateral_length or custom

    Returns:
        DataFrame with added "normalized_rate" column

    Example:
        >>> normalized = normalize_production(production_df, normalize_by="peak")
    """
    result = df.copy()

    if wells is not None:
        result = result[result[uwi_column].isin(wells)]

    norm_factors = {}

    for uwi in result[uwi_column].unique():
        well_data = result[result[uwi_column] == uwi]

        if normalize_by == "peak":
            norm_factors[uwi] = well_data[rate_column].max()
        elif normalize_by == "first_month":
            sorted_data = well_data.sort_values("production_date")
            norm_factors[uwi] = sorted_data[rate_column].iloc[0] if len(sorted_data) > 0 else 1.0
        elif normalize_by in ("lateral_length", "custom"):
            if normalization_values is None or uwi not in normalization_values:
                raise ValueError(f"Missing normalization value for {uwi}")
            norm_factors[uwi] = normalization_values[uwi]
        else:
            raise ValueError(f"Unknown normalize_by: {normalize_by}")

    def safe_normalize(row: pd.Series) -> float:
        factor = norm_factors.get(row[uwi_column], 1.0)
        if factor > 0:
            return row[rate_column] / factor
        return 0.0

    result["normalized_rate"] = result.apply(safe_normalize, axis=1)
    result["normalization_factor"] = result[uwi_column].map(norm_factors)

    return result


def align_production(
    df: pd.DataFrame,
    wells: list[str] | None = None,
    align_by: Literal["first_production", "peak_production", "completion_date"] = "first_production",
    date_column: str = "production_date",
    rate_column: str = "oil",
    uwi_column: str = "uwi",
    completion_dates: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Align production data by time zero for each well.

    Args:
        df: DataFrame with production data
        wells: Optional list of wells to include
        align_by: Alignment method:
            - "first_production": Align to first production date
            - "peak_production": Align to peak production month
            - "completion_date": Align to completion date (requires dates)
        date_column: Name of date column
        rate_column: Name of rate column
        uwi_column: Name of UWI column
        completion_dates: Dict of {uwi: date_str} for completion alignment

    Returns:
        DataFrame with added "month" column (0-indexed from alignment point)

    Example:
        >>> aligned = align_production(production_df, align_by="first_production")
    """
    result = df.copy()

    if wells is not None:
        result = result[result[uwi_column].isin(wells)]

    result[date_column] = pd.to_datetime(result[date_column])

    ref_dates = {}

    for uwi in result[uwi_column].unique():
        well_data = result[result[uwi_column] == uwi].sort_values(date_column)

        if len(well_data) == 0:
            continue

        if align_by == "first_production":
            ref_dates[uwi] = well_data[date_column].min()
        elif align_by == "peak_production":
            peak_idx = well_data[rate_column].idxmax()
            ref_dates[uwi] = well_data.loc[peak_idx, date_column]
        elif align_by == "completion_date":
            if completion_dates is None or uwi not in completion_dates:
                raise ValueError(f"Missing completion date for {uwi}")
            ref_dates[uwi] = pd.to_datetime(completion_dates[uwi])
        else:
            raise ValueError(f"Unknown align_by: {align_by}")

    def calc_month(row: pd.Series) -> int:
        ref = ref_dates.get(row[uwi_column])
        if ref is None:
            return 0
        delta = (row[date_column].year - ref.year) * 12 + (row[date_column].month - ref.month)
        return int(delta)

    result["month"] = result.apply(calc_month, axis=1)

    return result


def aggregate_by_month(
    df: pd.DataFrame,
    rate_column: str = "oil",
    month_column: str = "month",
    uwi_column: str = "uwi",
    agg_functions: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate aligned production by month.

    Args:
        df: DataFrame with aligned production data
        rate_column: Name of rate column
        month_column: Name of month column
        uwi_column: Name of UWI column
        agg_functions: List of aggregation functions (default: count, mean, std)

    Returns:
        DataFrame aggregated by month with statistics

    Example:
        >>> stats = aggregate_by_month(aligned_df)
    """
    if agg_functions is None:
        agg_functions = ["count", "mean", "std", "min", "max"]

    result = df.groupby(month_column)[rate_column].agg(agg_functions).reset_index()
    result.columns = [month_column] + [f"{rate_column}_{fn}" for fn in agg_functions]

    well_counts = df.groupby(month_column)[uwi_column].nunique().reset_index()
    well_counts.columns = [month_column, "well_count"]

    result = result.merge(well_counts, on=month_column)

    return result
