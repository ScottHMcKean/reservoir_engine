"""Feature engineering for production prediction models.

Provides functions for building feature matrices from production,
completion, and survey data.
"""

from typing import Literal

import numpy as np
import pandas as pd


def build_feature_matrix(
    wells_df: pd.DataFrame,
    production_df: pd.DataFrame | None = None,
    completions_df: pd.DataFrame | None = None,
    surveys_df: pd.DataFrame | None = None,
    uwi_column: str = "uwi",
) -> pd.DataFrame:
    """Build feature matrix for ML models by combining data sources.

    Joins well header, production, completion, and survey data into
    a single feature matrix suitable for machine learning.

    Args:
        wells_df: DataFrame with well header information (must have uwi, lat, lon)
        production_df: Optional production data
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

    # Add production features
    if production_df is not None:
        prod_features = _extract_production_features(production_df, uwi_column)
        result = result.merge(prod_features, on=uwi_column, how="left")

    # Add completion features
    if completions_df is not None:
        comp_features = _extract_completion_features(completions_df, uwi_column)
        result = result.merge(comp_features, on=uwi_column, how="left")

    # Add survey/trajectory features
    if surveys_df is not None:
        survey_features = _extract_survey_features(surveys_df, uwi_column)
        result = result.merge(survey_features, on=uwi_column, how="left")

    # Add well header features (if present)
    header_cols = ["latitude", "longitude", "formation", "operator", "completion_date"]
    available_cols = [c for c in header_cols if c in wells_df.columns]
    if available_cols:
        result = result.merge(
            wells_df[[uwi_column] + available_cols].drop_duplicates(subset=[uwi_column]),
            on=uwi_column,
            how="left",
        )

    return result


def _extract_production_features(
    production_df: pd.DataFrame,
    uwi_column: str = "uwi",
) -> pd.DataFrame:
    """Extract production-based features."""
    features = production_df.groupby(uwi_column).agg(
        production_months=("date", "count"),
        cum_oil=("oil_bbl", "sum"),
        cum_gas=("gas_mcf", "sum"),
        cum_water=("water_bbl", "sum"),
        peak_oil=("oil_bbl", "max"),
        avg_oil=("oil_bbl", "mean"),
        first_month_oil=("oil_bbl", "first"),
    ).reset_index()

    # Derived features
    features["water_cut"] = features["cum_water"] / (
        features["cum_oil"] + features["cum_water"] + 1
    )
    features["gor"] = features["cum_gas"] / (features["cum_oil"] + 1)

    return features


def _extract_completion_features(
    completions_df: pd.DataFrame,
    uwi_column: str = "uwi",
) -> pd.DataFrame:
    """Extract completion-based features."""
    features = completions_df.groupby(uwi_column).agg(
        stage_count=("stage", "max"),
        total_proppant=("proppant_tonnes", "sum"),
        total_fluid=("fluid_m3", "sum"),
        total_clusters=("cluster_count", "sum"),
        lateral_length=("perf_btm_md", lambda x: x.max() - x.min()),
    ).reset_index()

    # Derived features
    features["proppant_per_stage"] = features["total_proppant"] / (features["stage_count"] + 1)
    features["fluid_per_stage"] = features["total_fluid"] / (features["stage_count"] + 1)
    features["proppant_intensity"] = features["total_proppant"] / (
        features["lateral_length"] + 1
    )

    return features


def _extract_survey_features(
    surveys_df: pd.DataFrame,
    uwi_column: str = "uwi",
) -> pd.DataFrame:
    """Extract survey/trajectory-based features."""
    features = surveys_df.groupby(uwi_column).agg(
        total_md=("md", "max"),
        total_tvd=("tvd", "max"),
        max_inclination=("inclination", "max"),
        avg_inclination=("inclination", "mean"),
    ).reset_index()

    # Derived features
    features["horizontal_displacement"] = np.sqrt(
        features["total_md"] ** 2 - features["total_tvd"] ** 2
    )

    return features


def calculate_eur_targets(
    production_df: pd.DataFrame,
    months: list[int] = [6, 12, 24, 36, 60],
    rate_column: str = "oil_bbl",
    uwi_column: str = "uwi",
) -> pd.DataFrame:
    """Calculate EUR at various time horizons for target variables.

    Args:
        production_df: DataFrame with production data
        months: List of month horizons to calculate EUR
        rate_column: Name of rate column
        uwi_column: Name of UWI column

    Returns:
        DataFrame with uwi and eur_{months}m columns

    Example:
        >>> targets = calculate_eur_targets(df, months=[12, 24, 60])
        >>> # Use targets["eur_12m"] as ML target
    """
    production_df = production_df.copy()
    production_df = production_df.sort_values([uwi_column, "date"])

    # Calculate month number for each well
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


def create_spatial_features(
    wells_df: pd.DataFrame,
    lat_column: str = "latitude",
    lon_column: str = "longitude",
    uwi_column: str = "uwi",
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """Create spatial features based on well locations.

    Args:
        wells_df: DataFrame with well locations
        lat_column: Name of latitude column
        lon_column: Name of longitude column
        uwi_column: Name of UWI column
        n_neighbors: Number of neighbors for density calculation

    Returns:
        DataFrame with spatial features

    Example:
        >>> spatial = create_spatial_features(wells_df)
    """
    from scipy.spatial import distance

    # Extract coordinates
    coords = wells_df[[lon_column, lat_column]].values

    # Calculate pairwise distances
    dist_matrix = distance.cdist(coords, coords, metric="euclidean")

    # Calculate features
    features = []

    for i, uwi in enumerate(wells_df[uwi_column]):
        distances = np.sort(dist_matrix[i])[1:]  # Exclude self

        row = {
            uwi_column: uwi,
            "nearest_well_dist": distances[0] if len(distances) > 0 else np.nan,
            "avg_neighbor_dist": np.mean(distances[:n_neighbors]) if len(distances) >= n_neighbors else np.nan,
            "well_density": n_neighbors / (np.sum(distances[:n_neighbors]) + 1e-6) if len(distances) >= n_neighbors else 0,
        }
        features.append(row)

    return pd.DataFrame(features)


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

