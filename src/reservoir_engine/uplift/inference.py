"""Spatial inference and prediction for production mapping.

Generate predictions on a spatial grid for visualization
and decision support.
"""

from typing import Any

import numpy as np
import pandas as pd


def create_prediction_grid(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    resolution_m: float = 500,
) -> pd.DataFrame:
    """Create a regular grid for spatial predictions.

    Args:
        min_lon: Minimum longitude (west boundary)
        min_lat: Minimum latitude (south boundary)
        max_lon: Maximum longitude (east boundary)
        max_lat: Maximum latitude (north boundary)
        resolution_m: Grid spacing in meters (approximate)

    Returns:
        DataFrame with columns: grid_id, latitude, longitude

    Example:
        >>> grid = create_prediction_grid(-110, 50, -109, 51, resolution_m=1000)
        >>> print(f"Grid points: {len(grid)}")
    """
    # Approximate conversion: 1 degree ~ 111 km at equator
    # Adjust for latitude
    lat_center = (min_lat + max_lat) / 2
    lon_degree_m = 111320 * np.cos(np.radians(lat_center))
    lat_degree_m = 110540

    # Calculate step sizes in degrees
    lon_step = resolution_m / lon_degree_m
    lat_step = resolution_m / lat_degree_m

    # Create grid
    lons = np.arange(min_lon, max_lon + lon_step, lon_step)
    lats = np.arange(min_lat, max_lat + lat_step, lat_step)

    grid_points = []
    grid_id = 0

    for lon in lons:
        for lat in lats:
            grid_points.append(
                {
                    "grid_id": grid_id,
                    "longitude": lon,
                    "latitude": lat,
                }
            )
            grid_id += 1

    return pd.DataFrame(grid_points)


def predict_on_grid(
    model: Any,
    grid_df: pd.DataFrame,
    feature_columns: list[str],
    wells_df: pd.DataFrame | None = None,
    spatial_interpolation: bool = True,
    interpolation_radius_km: float = 5.0,
) -> pd.DataFrame:
    """Generate predictions on a spatial grid.

    Args:
        model: Trained prediction model
        grid_df: DataFrame from create_prediction_grid
        feature_columns: List of feature column names (must match training)
        wells_df: Optional DataFrame with well features for interpolation
        spatial_interpolation: If True, interpolate features from nearby wells
        interpolation_radius_km: Radius for spatial interpolation

    Returns:
        Grid DataFrame with added 'prediction' column

    Example:
        >>> grid = create_prediction_grid(-110, 50, -109, 51)
        >>> predictions = predict_on_grid(model, grid, feature_columns, wells_df)
    """
    result = grid_df.copy()

    if spatial_interpolation and wells_df is not None:
        # Interpolate features from nearby wells
        result = _interpolate_features_to_grid(
            result, wells_df, feature_columns, interpolation_radius_km
        )
    else:
        # Use placeholder values (mean) for non-spatial features
        for col in feature_columns:
            if col not in result.columns:
                result[col] = 0.0

    # Make predictions
    X = result[feature_columns].fillna(0)
    result["prediction"] = model.predict(X)

    return result


def _interpolate_features_to_grid(
    grid_df: pd.DataFrame,
    wells_df: pd.DataFrame,
    feature_columns: list[str],
    radius_km: float,
) -> pd.DataFrame:
    """Interpolate well features to grid points using inverse distance weighting."""
    result = grid_df.copy()

    # Initialize feature columns
    for col in feature_columns:
        if col not in result.columns:
            result[col] = np.nan

    # Earth radius in km
    R = 6371

    for idx in result.index:
        grid_lat = np.radians(result.loc[idx, "latitude"])
        grid_lon = np.radians(result.loc[idx, "longitude"])

        # Calculate distances to all wells
        well_lats = np.radians(wells_df["latitude"].values)
        well_lons = np.radians(wells_df["longitude"].values)

        dlat = well_lats - grid_lat
        dlon = well_lons - grid_lon

        a = np.sin(dlat / 2) ** 2 + np.cos(grid_lat) * np.cos(well_lats) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distances = R * c

        # Find wells within radius
        mask = distances <= radius_km

        if not np.any(mask):
            continue

        nearby_wells = wells_df.loc[mask]
        nearby_distances = distances[mask]

        # Inverse distance weighting
        weights = 1 / (nearby_distances + 0.001)  # Add small value to avoid div by zero
        weights = weights / weights.sum()

        # Interpolate each feature
        for col in feature_columns:
            if col in nearby_wells.columns:
                values = nearby_wells[col].values
                valid_mask = ~np.isnan(values)
                if np.any(valid_mask):
                    result.loc[idx, col] = np.sum(weights[valid_mask] * values[valid_mask])

    return result


def export_to_geojson(
    grid_df: pd.DataFrame,
    value_column: str = "prediction",
    output_path: str | None = None,
) -> dict:
    """Export grid predictions to GeoJSON for visualization.

    Args:
        grid_df: DataFrame with predictions
        value_column: Column to include in properties
        output_path: Optional path to save GeoJSON file

    Returns:
        GeoJSON dictionary

    Example:
        >>> geojson = export_to_geojson(predictions, "prediction")
        >>> # Use in Leaflet or other mapping library
    """
    import json

    features = []

    for _, row in grid_df.iterrows():
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["longitude"], row["latitude"]],
            },
            "properties": {
                "grid_id": int(row.get("grid_id", 0)),
                value_column: float(row.get(value_column, 0)),
            },
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(geojson, f)

    return geojson


def bin_predictions(
    grid_df: pd.DataFrame,
    value_column: str = "prediction",
    n_bins: int = 5,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Bin predictions into categories for visualization.

    Args:
        grid_df: DataFrame with predictions
        value_column: Column to bin
        n_bins: Number of bins
        labels: Optional custom labels for bins

    Returns:
        DataFrame with added 'bin' column

    Example:
        >>> binned = bin_predictions(predictions, n_bins=5)
        >>> # Use bin column for choropleth coloring
    """
    result = grid_df.copy()

    if labels is None:
        labels = [f"Q{i+1}" for i in range(n_bins)]

    result["bin"] = pd.qcut(
        result[value_column],
        q=n_bins,
        labels=labels,
        duplicates="drop",
    )

    return result

