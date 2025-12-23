"""Feature engineering for production prediction models.

Provides functions for building feature matrices from production,
completion, and survey data.

Uses IHS column naming conventions.
"""

from typing import Literal

import numpy as np
import pandas as pd


def build_feature_matrix(
    wells_df: pd.DataFrame,
    production_df: pd.DataFrame | None = None,
    production_summary_df: pd.DataFrame | None = None,
    completions_df: pd.DataFrame | None = None,
    surveys_df: pd.DataFrame | None = None,
    uwi_column: str = "uwi",
) -> pd.DataFrame:
    """Build feature matrix for ML models by combining data sources.

    Joins well header, production, completion, and survey data into
    a single feature matrix suitable for machine learning.

    Args:
        wells_df: DataFrame with well header information
        production_df: Optional monthly production data
        production_summary_df: Optional production summary data
        completions_df: Optional completion data
        surveys_df: Optional survey/trajectory data
        uwi_column: Name of UWI column

    Returns:
        DataFrame with one row per well and engineered features

    Example:
        >>> features = build_feature_matrix(wells_df, production_df, completions_df)
    """
    result = wells_df[[uwi_column]].copy()
    result = result.drop_duplicates(subset=[uwi_column])

    # Add well header features
    header_features = _extract_header_features(wells_df, uwi_column)
    result = result.merge(header_features, on=uwi_column, how="left")

    # Add production features (from monthly data)
    if production_df is not None:
        prod_features = _extract_production_features(production_df, uwi_column)
        result = result.merge(prod_features, on=uwi_column, how="left")

    # Add production summary features (pre-calculated metrics)
    if production_summary_df is not None:
        summary_features = _extract_summary_features(production_summary_df, uwi_column)
        result = result.merge(summary_features, on=uwi_column, how="left")

    # Add completion features
    if completions_df is not None:
        comp_features = _extract_completion_features(completions_df, uwi_column)
        result = result.merge(comp_features, on=uwi_column, how="left")

    # Add survey/trajectory features
    if surveys_df is not None:
        survey_features = _extract_survey_features(surveys_df, uwi_column)
        result = result.merge(survey_features, on=uwi_column, how="left")

    return result


def _extract_header_features(
    wells_df: pd.DataFrame,
    uwi_column: str = "uwi",
) -> pd.DataFrame:
    """Extract features from well header data."""
    feature_cols = [
        uwi_column,
        "surface_latitude",
        "surface_longitude",
        "drill_td",
        "tvd",
        "lateral_length",
        "hole_direction",
        "play_type",
    ]

    # Only include columns that exist
    available_cols = [c for c in feature_cols if c in wells_df.columns]
    return wells_df[available_cols].drop_duplicates(subset=[uwi_column])


def _extract_production_features(
    production_df: pd.DataFrame,
    uwi_column: str = "uwi",
) -> pd.DataFrame:
    """Extract production-based features from monthly data."""
    features = production_df.groupby(uwi_column).agg(
        production_months=("production_date", "count"),
        cum_oil=("oil", "sum"),
        cum_gas=("gas", "sum"),
        cum_water=("water", "sum"),
        peak_oil=("oil", "max"),
        avg_oil=("oil", "mean"),
        first_month_oil=("oil", "first"),
    ).reset_index()

    # Derived features
    features["water_cut"] = features["cum_water"] / (
        features["cum_oil"] + features["cum_water"] + 1
    )
    features["gor"] = features["cum_gas"] / (features["cum_oil"] + 1)

    return features


def _extract_summary_features(
    summary_df: pd.DataFrame,
    uwi_column: str = "uwi",
) -> pd.DataFrame:
    """Extract features from production summary data."""
    feature_cols = [
        uwi_column,
        "cum_boe",
        "cum_production_hours",
        "first_3_month_boe",
        "first_3_month_oil",
        "first_3_month_gas",
        "max_boe",
        "max_oil",
        "max_gas",
    ]

    available_cols = [c for c in feature_cols if c in summary_df.columns]
    return summary_df[available_cols].drop_duplicates(subset=[uwi_column])


def _extract_completion_features(
    completions_df: pd.DataFrame,
    uwi_column: str = "uwi",
) -> pd.DataFrame:
    """Extract completion-based features."""
    features = completions_df.groupby(uwi_column).agg(
        completion_count=("completion_obs_no", "count"),
        total_perf_shots=("perf_shots", "sum"),
        min_top_depth=("top_depth", "min"),
        max_base_depth=("base_depth", "max"),
    ).reset_index()

    # Derived features
    features["completion_interval"] = features["max_base_depth"] - features["min_top_depth"]

    return features


def _extract_survey_features(
    surveys_df: pd.DataFrame,
    uwi_column: str = "uwi",
) -> pd.DataFrame:
    """Extract survey/trajectory-based features."""
    features = surveys_df.groupby(uwi_column).agg(
        max_md=("station_md", "max"),
        max_tvd=("station_tvd", "max"),
        max_inclination=("inclination", "max"),
        avg_inclination=("inclination", "mean"),
        survey_points=("station_md", "count"),
    ).reset_index()

    # Derived features
    features["horizontal_displacement"] = np.sqrt(
        np.maximum(0, features["max_md"] ** 2 - features["max_tvd"] ** 2)
    )

    return features


def calculate_eur_targets(
    production_df: pd.DataFrame,
    months: list[int] | None = None,
    rate_column: str = "oil",
    uwi_column: str = "uwi",
) -> pd.DataFrame:
    """Calculate EUR at various time horizons for target variables.

    Args:
        production_df: DataFrame with production data
        months: List of month horizons to calculate EUR (default: [6, 12, 24, 36, 60])
        rate_column: Name of rate column
        uwi_column: Name of UWI column

    Returns:
        DataFrame with uwi and eur_{months}m columns

    Example:
        >>> targets = calculate_eur_targets(df, months=[12, 24, 60])
    """
    if months is None:
        months = [6, 12, 24, 36, 60]

    production_df = production_df.copy()
    production_df = production_df.sort_values([uwi_column, "production_date"])
    production_df["month_num"] = production_df.groupby(uwi_column).cumcount() + 1

    results = []

    for uwi in production_df[uwi_column].unique():
        well_data = production_df[production_df[uwi_column] == uwi]
        row = {uwi_column: uwi}

        for month_horizon in months:
            eur = well_data[well_data["month_num"] <= month_horizon][rate_column].sum()
            row[f"eur_{month_horizon}m"] = eur

        results.append(row)

    return pd.DataFrame(results)


def create_spatial_features_h3(
    wells_df: pd.DataFrame,
    lat_column: str = "surface_latitude",
    lon_column: str = "surface_longitude",
    uwi_column: str = "uwi",
    resolution: int = 9,
) -> pd.DataFrame:
    """Create spatial features based on H3 indexing.

    Args:
        wells_df: DataFrame with well locations
        lat_column: Name of latitude column
        lon_column: Name of longitude column
        uwi_column: Name of UWI column
        resolution: H3 resolution for neighbor calculations

    Returns:
        DataFrame with spatial features

    Example:
        >>> spatial = create_spatial_features_h3(wells_df)
    """
    import h3

    result = wells_df[[uwi_column, lat_column, lon_column]].copy()
    result = result.dropna(subset=[lat_column, lon_column])

    # Add H3 index
    result["h3_index"] = result.apply(
        lambda row: h3.latlng_to_cell(row[lat_column], row[lon_column], resolution),
        axis=1,
    )

    # Count wells per H3 cell (density)
    cell_counts = result.groupby("h3_index").size().reset_index(name="wells_in_cell")
    result = result.merge(cell_counts, on="h3_index", how="left")

    # Get parent cell at coarser resolution for regional context
    parent_resolution = max(0, resolution - 3)
    result["h3_parent"] = result["h3_index"].apply(
        lambda x: h3.cell_to_parent(x, parent_resolution)
    )

    parent_counts = result.groupby("h3_parent").size().reset_index(name="wells_in_region")
    result = result.merge(parent_counts, on="h3_parent", how="left")

    return result[[uwi_column, "h3_index", "wells_in_cell", "h3_parent", "wells_in_region"]]


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_columns: list[str],
    method: Literal["onehot", "label", "target"] = "onehot",
    target_column: str | None = None,
) -> pd.DataFrame:
    """Encode categorical features for ML models.

    Args:
        df: DataFrame with features
        categorical_columns: List of categorical column names
        method: Encoding method:
            - "onehot": One-hot encoding
            - "label": Label encoding
            - "target": Target encoding (requires target_column)
        target_column: Target column for target encoding

    Returns:
        DataFrame with encoded features
    """
    result = df.copy()

    for col in categorical_columns:
        if col not in result.columns:
            continue

        if method == "onehot":
            dummies = pd.get_dummies(result[col], prefix=col)
            result = pd.concat([result, dummies], axis=1)
            result = result.drop(columns=[col])

        elif method == "label":
            result[f"{col}_encoded"] = pd.Categorical(result[col]).codes

        elif method == "target":
            if target_column is None:
                raise ValueError("target_column required for target encoding")
            target_means = result.groupby(col)[target_column].mean()
            result[f"{col}_encoded"] = result[col].map(target_means)

    return result
