"""Spatial selection utilities for well filtering.

Provides functions to select wells by polygon, bounding box,
or proximity to a point.
"""

from typing import Any

import pandas as pd


def wells_in_polygon(
    wells_df: pd.DataFrame,
    polygon: dict[str, Any] | list[tuple[float, float]],
    lat_column: str = "latitude",
    lon_column: str = "longitude",
    uwi_column: str = "uwi",
) -> list[str]:
    """Select wells within a polygon.

    Args:
        wells_df: DataFrame with well locations
        polygon: GeoJSON polygon dict or list of (lon, lat) tuples
        lat_column: Name of latitude column
        lon_column: Name of longitude column
        uwi_column: Name of UWI column

    Returns:
        List of UWIs for wells within the polygon

    Example:
        >>> polygon = [(-110, 50), (-110, 51), (-109, 51), (-109, 50), (-110, 50)]
        >>> wells = wells_in_polygon(df, polygon)
    """
    try:
        from shapely.geometry import Point, Polygon, shape
    except ImportError:
        raise ImportError("shapely required for spatial operations. Install with: uv add shapely")

    # Convert polygon to shapely geometry
    if isinstance(polygon, dict):
        # GeoJSON format
        poly = shape(polygon)
    else:
        # List of coordinates
        poly = Polygon(polygon)

    # Check each well
    selected = []
    for _, row in wells_df.iterrows():
        point = Point(row[lon_column], row[lat_column])
        if poly.contains(point):
            selected.append(row[uwi_column])

    return selected


def wells_in_bbox(
    wells_df: pd.DataFrame,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    lat_column: str = "latitude",
    lon_column: str = "longitude",
    uwi_column: str = "uwi",
) -> list[str]:
    """Select wells within a bounding box.

    Args:
        wells_df: DataFrame with well locations
        min_lon: Minimum longitude (west)
        min_lat: Minimum latitude (south)
        max_lon: Maximum longitude (east)
        max_lat: Maximum latitude (north)
        lat_column: Name of latitude column
        lon_column: Name of longitude column
        uwi_column: Name of UWI column

    Returns:
        List of UWIs for wells within the bounding box

    Example:
        >>> wells = wells_in_bbox(df, -110, 50, -109, 51)
    """
    mask = (
        (wells_df[lon_column] >= min_lon)
        & (wells_df[lon_column] <= max_lon)
        & (wells_df[lat_column] >= min_lat)
        & (wells_df[lat_column] <= max_lat)
    )

    return wells_df.loc[mask, uwi_column].tolist()


def wells_near_point(
    wells_df: pd.DataFrame,
    center_lon: float,
    center_lat: float,
    radius_km: float,
    lat_column: str = "latitude",
    lon_column: str = "longitude",
    uwi_column: str = "uwi",
) -> list[str]:
    """Select wells within a radius of a point.

    Uses Haversine formula for accurate distance calculation.

    Args:
        wells_df: DataFrame with well locations
        center_lon: Center point longitude
        center_lat: Center point latitude
        radius_km: Search radius in kilometers
        lat_column: Name of latitude column
        lon_column: Name of longitude column
        uwi_column: Name of UWI column

    Returns:
        List of UWIs for wells within the radius

    Example:
        >>> wells = wells_near_point(df, -110.5, 50.5, radius_km=10)
    """
    import numpy as np

    # Haversine distance calculation
    R = 6371  # Earth radius in km

    lat1 = np.radians(center_lat)
    lat2 = np.radians(wells_df[lat_column].values)
    dlat = np.radians(wells_df[lat_column].values - center_lat)
    dlon = np.radians(wells_df[lon_column].values - center_lon)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distances = R * c

    mask = distances <= radius_km
    return wells_df.loc[mask, uwi_column].tolist()


def filter_wells_by_attributes(
    wells_df: pd.DataFrame,
    uwis: list[str] | None = None,
    min_production_months: int | None = None,
    formation: str | None = None,
    completion_year_min: int | None = None,
    completion_year_max: int | None = None,
    uwi_column: str = "uwi",
) -> list[str]:
    """Filter wells by various attributes.

    Args:
        wells_df: DataFrame with well attributes
        uwis: Optional list of UWIs to filter from
        min_production_months: Minimum months of production
        formation: Target formation name
        completion_year_min: Minimum completion year
        completion_year_max: Maximum completion year
        uwi_column: Name of UWI column

    Returns:
        Filtered list of UWIs
    """
    df = wells_df.copy()

    if uwis is not None:
        df = df[df[uwi_column].isin(uwis)]

    if min_production_months is not None and "production_months" in df.columns:
        df = df[df["production_months"] >= min_production_months]

    if formation is not None and "formation" in df.columns:
        df = df[df["formation"] == formation]

    if completion_year_min is not None and "completion_year" in df.columns:
        df = df[df["completion_year"] >= completion_year_min]

    if completion_year_max is not None and "completion_year" in df.columns:
        df = df[df["completion_year"] <= completion_year_max]

    return df[uwi_column].tolist()

