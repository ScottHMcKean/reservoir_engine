"""Spatial inference and prediction using H3.

Generate predictions on a spatial grid for visualization
and decision support.
"""

from typing import Any

import numpy as np
import pandas as pd


def create_prediction_grid_h3(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    resolution: int = 7,
) -> pd.DataFrame:
    """Create a prediction grid using H3 cells.

    Args:
        min_lon: Minimum longitude (west boundary)
        min_lat: Minimum latitude (south boundary)
        max_lon: Maximum longitude (east boundary)
        max_lat: Maximum latitude (north boundary)
        resolution: H3 resolution (7 = ~5km², 8 = ~0.7km², 9 = ~0.1km²)

    Returns:
        DataFrame with columns: h3_index, latitude, longitude

    Example:
        >>> grid = create_prediction_grid_h3(-117, 54, -116, 55, resolution=8)
    """
    import h3

    # Create bounding box polygon
    geojson_polygon = {
        "type": "Polygon",
        "coordinates": [
            [
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]
        ],
    }

    # Get all H3 cells in the bounding box
    cells = h3.geo_to_cells(geojson_polygon, resolution)

    # Get centroids of each cell
    grid_points = []
    for cell in cells:
        lat, lon = h3.cell_to_latlng(cell)
        grid_points.append(
            {
                "h3_index": cell,
                "latitude": lat,
                "longitude": lon,
            }
        )

    return pd.DataFrame(grid_points)


def predict_on_grid(
    model: Any,
    grid_df: pd.DataFrame,
    feature_columns: list[str],
    wells_df: pd.DataFrame | None = None,
    interpolation_k: int = 1,
) -> pd.DataFrame:
    """Generate predictions on a spatial grid using H3.

    Args:
        model: Trained prediction model
        grid_df: DataFrame from create_prediction_grid_h3
        feature_columns: List of feature column names
        wells_df: Optional DataFrame with well features for interpolation
        interpolation_k: K-ring for neighbor interpolation

    Returns:
        Grid DataFrame with added 'prediction' column

    Example:
        >>> grid = create_prediction_grid_h3(-117, 54, -116, 55)
        >>> predictions = predict_on_grid(model, grid, feature_columns, wells_df)
    """

    result = grid_df.copy()

    if wells_df is not None:
        # Interpolate features from nearby wells using H3 neighbors
        result = _interpolate_features_h3(
            result, wells_df, feature_columns, interpolation_k
        )
    else:
        # Initialize missing features to 0
        for col in feature_columns:
            if col not in result.columns:
                result[col] = 0.0

    # Make predictions
    X = result[feature_columns].fillna(0)
    result["prediction"] = model.predict(X)

    return result


def _interpolate_features_h3(
    grid_df: pd.DataFrame,
    wells_df: pd.DataFrame,
    feature_columns: list[str],
    k: int = 1,
) -> pd.DataFrame:
    """Interpolate well features to grid points using H3 neighbors."""
    import h3

    result = grid_df.copy()

    # Ensure wells have H3 index
    if "h3_index" not in wells_df.columns:
        wells_with_h3 = wells_df.copy()
        wells_with_h3["h3_index"] = wells_with_h3.apply(
            lambda row: h3.latlng_to_cell(
                row["surface_latitude"], row["surface_longitude"], 9
            )
            if pd.notna(row.get("surface_latitude"))
            else None,
            axis=1,
        )
    else:
        wells_with_h3 = wells_df

    # Initialize feature columns
    for col in feature_columns:
        if col not in result.columns:
            result[col] = np.nan

    # For each grid cell, find wells in k-ring and average their features
    for idx in result.index:
        grid_cell = result.loc[idx, "h3_index"]

        # Get cells in k-ring
        neighbor_cells = h3.grid_disk(grid_cell, k)

        # Find wells in those cells
        nearby_wells = wells_with_h3[wells_with_h3["h3_index"].isin(neighbor_cells)]

        if len(nearby_wells) > 0:
            for col in feature_columns:
                if col in nearby_wells.columns:
                    result.loc[idx, col] = nearby_wells[col].mean()

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
                "h3_index": row.get("h3_index", ""),
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


def export_h3_hexagons_geojson(
    grid_df: pd.DataFrame,
    value_column: str = "prediction",
    output_path: str | None = None,
) -> dict:
    """Export H3 hexagons as GeoJSON polygons for choropleth visualization.

    Args:
        grid_df: DataFrame with H3 indices and predictions
        value_column: Column to include in properties
        output_path: Optional path to save GeoJSON file

    Returns:
        GeoJSON dictionary with hexagon polygons
    """
    import json

    import h3

    features = []

    for _, row in grid_df.iterrows():
        h3_index = row.get("h3_index")
        if h3_index is None:
            continue

        # Get hexagon boundary
        boundary = h3.cell_to_boundary(h3_index)
        # Convert to GeoJSON format (lon, lat) and close the ring
        coords = [[lon, lat] for lat, lon in boundary]
        coords.append(coords[0])  # Close the polygon

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
            "properties": {
                "h3_index": h3_index,
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
