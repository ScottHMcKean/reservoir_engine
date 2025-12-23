"""Spatial selection utilities using H3 hexagonal grid.

H3 provides a consistent, hierarchical spatial index that is:
- Resolution-independent (can zoom in/out)
- Efficient for spatial queries
- Native to Databricks and compatible with Spark

Key H3 resolutions:
- Resolution 4: ~1,770 km² (regional analysis)
- Resolution 7: ~5.16 km² (field-level)
- Resolution 9: ~0.1 km² (well-level, ~105m edge)
- Resolution 10: ~0.015 km² (precise, ~66m edge)
"""

import pandas as pd

# Default H3 resolution for well-level analysis
DEFAULT_RESOLUTION = 9


def get_h3_index(lat: float, lon: float, resolution: int = DEFAULT_RESOLUTION) -> str:
    """Get H3 index for a lat/lon point.

    Args:
        lat: Latitude
        lon: Longitude
        resolution: H3 resolution (0-15, higher = finer)

    Returns:
        H3 index string

    Example:
        >>> h3_idx = get_h3_index(54.5, -116.5, resolution=9)
    """
    import h3

    return h3.latlng_to_cell(lat, lon, resolution)


def add_h3_index(
    df: pd.DataFrame,
    lat_column: str = "surface_latitude",
    lon_column: str = "surface_longitude",
    resolution: int = DEFAULT_RESOLUTION,
    h3_column: str = "h3_index",
) -> pd.DataFrame:
    """Add H3 index column to a DataFrame with lat/lon.

    Args:
        df: DataFrame with lat/lon columns
        lat_column: Name of latitude column
        lon_column: Name of longitude column
        resolution: H3 resolution
        h3_column: Name for new H3 index column

    Returns:
        DataFrame with added H3 index column

    Example:
        >>> wells_with_h3 = add_h3_index(wells_df)
    """
    import h3

    result = df.copy()
    result[h3_column] = result.apply(
        lambda row: h3.latlng_to_cell(row[lat_column], row[lon_column], resolution)
        if pd.notna(row[lat_column]) and pd.notna(row[lon_column])
        else None,
        axis=1,
    )
    return result


def wells_in_h3_cells(
    wells_df: pd.DataFrame,
    h3_cells: list[str],
    h3_column: str = "h3_index",
    uwi_column: str = "uwi",
) -> list[str]:
    """Select wells that fall within specified H3 cells.

    Args:
        wells_df: DataFrame with H3 index column
        h3_cells: List of H3 cell indices to filter by
        h3_column: Name of H3 index column
        uwi_column: Name of UWI column

    Returns:
        List of UWIs within the specified cells

    Example:
        >>> cells = get_h3_ring(center_h3, k=2)  # 2-ring around center
        >>> wells = wells_in_h3_cells(wells_df, cells)
    """
    mask = wells_df[h3_column].isin(h3_cells)
    return wells_df.loc[mask, uwi_column].tolist()


def wells_in_h3_ring(
    wells_df: pd.DataFrame,
    center_lat: float,
    center_lon: float,
    k_rings: int = 1,
    resolution: int = DEFAULT_RESOLUTION,
    lat_column: str = "surface_latitude",
    lon_column: str = "surface_longitude",
    uwi_column: str = "uwi",
) -> list[str]:
    """Select wells within k H3 rings of a center point.

    Args:
        wells_df: DataFrame with well locations
        center_lat: Center point latitude
        center_lon: Center point longitude
        k_rings: Number of rings around center (1 = immediate neighbors)
        resolution: H3 resolution
        lat_column: Name of latitude column
        lon_column: Name of longitude column
        uwi_column: Name of UWI column

    Returns:
        List of UWIs within the ring

    Example:
        >>> wells = wells_in_h3_ring(wells_df, 54.5, -116.5, k_rings=3)
    """
    import h3

    # Get center cell and its k-ring
    center_cell = h3.latlng_to_cell(center_lat, center_lon, resolution)
    ring_cells = h3.grid_disk(center_cell, k_rings)

    # Add H3 index to wells if not present
    wells_with_h3 = add_h3_index(wells_df, lat_column, lon_column, resolution)

    return wells_in_h3_cells(wells_with_h3, list(ring_cells), uwi_column=uwi_column)


def wells_in_polygon_h3(
    wells_df: pd.DataFrame,
    polygon_coords: list[tuple[float, float]],
    resolution: int = DEFAULT_RESOLUTION,
    lat_column: str = "surface_latitude",
    lon_column: str = "surface_longitude",
    uwi_column: str = "uwi",
) -> list[str]:
    """Select wells within a polygon using H3 polyfill.

    Args:
        wells_df: DataFrame with well locations
        polygon_coords: List of (lat, lon) tuples defining polygon boundary
        resolution: H3 resolution for polyfill
        lat_column: Name of latitude column
        lon_column: Name of longitude column
        uwi_column: Name of UWI column

    Returns:
        List of UWIs within the polygon

    Example:
        >>> polygon = [(54.0, -117.0), (54.0, -116.0), (55.0, -116.0), (55.0, -117.0)]
        >>> wells = wells_in_polygon_h3(wells_df, polygon)
    """
    import h3

    # H3 expects GeoJSON format: outer ring as list of [lng, lat] pairs
    geojson_coords = [[lon, lat] for lat, lon in polygon_coords]
    # Close the polygon if needed
    if geojson_coords[0] != geojson_coords[-1]:
        geojson_coords.append(geojson_coords[0])

    geojson_polygon = {"type": "Polygon", "coordinates": [geojson_coords]}

    # Get all H3 cells that cover the polygon
    polygon_cells = h3.geo_to_cells(geojson_polygon, resolution)

    # Add H3 index to wells and filter
    wells_with_h3 = add_h3_index(wells_df, lat_column, lon_column, resolution)

    return wells_in_h3_cells(wells_with_h3, list(polygon_cells), uwi_column=uwi_column)


def wells_in_bbox(
    wells_df: pd.DataFrame,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    lat_column: str = "surface_latitude",
    lon_column: str = "surface_longitude",
    uwi_column: str = "uwi",
) -> list[str]:
    """Select wells within a bounding box.

    Simple lat/lon filtering without H3 (faster for bbox queries).

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
        List of UWIs within the bounding box

    Example:
        >>> wells = wells_in_bbox(wells_df, -117, 54, -116, 55)
    """
    mask = (
        (wells_df[lon_column] >= min_lon)
        & (wells_df[lon_column] <= max_lon)
        & (wells_df[lat_column] >= min_lat)
        & (wells_df[lat_column] <= max_lat)
    )
    return wells_df.loc[mask, uwi_column].tolist()


def get_h3_neighbors(h3_index: str, k: int = 1) -> list[str]:
    """Get k-ring neighbors of an H3 cell.

    Args:
        h3_index: H3 cell index
        k: Number of rings (1 = immediate neighbors)

    Returns:
        List of H3 indices including center and all neighbors
    """
    import h3

    return list(h3.grid_disk(h3_index, k))


def get_h3_resolution_for_radius(radius_km: float) -> int:
    """Get appropriate H3 resolution for a given radius.

    Args:
        radius_km: Target radius in kilometers

    Returns:
        H3 resolution that best matches the radius

    Example:
        >>> resolution = get_h3_resolution_for_radius(5.0)  # ~5km radius
    """
    # H3 edge lengths by resolution (approximate, in km)
    # Resolution: edge_length_km
    edge_lengths = {
        0: 1107.712,
        1: 418.676,
        2: 158.244,
        3: 59.810,
        4: 22.606,
        5: 8.544,
        6: 3.229,
        7: 1.220,
        8: 0.461,
        9: 0.174,
        10: 0.066,
        11: 0.025,
        12: 0.009,
        13: 0.003,
        14: 0.001,
        15: 0.0005,
    }

    # Find resolution where edge length is closest to radius
    for res, edge in edge_lengths.items():
        if edge <= radius_km:
            return res
    return 15


def filter_wells_by_attributes(
    wells_df: pd.DataFrame,
    uwis: list[str] | None = None,
    hole_direction: str | None = None,
    play_type: str | None = None,
    formation: str | None = None,
    min_lateral_length: float | None = None,
    uwi_column: str = "uwi",
) -> list[str]:
    """Filter wells by various attributes.

    Args:
        wells_df: DataFrame with well attributes
        uwis: Optional list of UWIs to filter from
        hole_direction: Well type filter (HORIZONTAL, VERTICAL, DIRECTIONAL)
        play_type: Play type filter (SHALE, TIGHT, etc.)
        formation: Target formation filter
        min_lateral_length: Minimum lateral length (m)
        uwi_column: Name of UWI column

    Returns:
        Filtered list of UWIs
    """
    df = wells_df.copy()

    if uwis is not None:
        df = df[df[uwi_column].isin(uwis)]

    if hole_direction is not None and "hole_direction" in df.columns:
        df = df[df["hole_direction"] == hole_direction]

    if play_type is not None and "play_type" in df.columns:
        df = df[df["play_type"] == play_type]

    if formation is not None and "target_formation" in df.columns:
        df = df[df["target_formation"] == formation]

    if min_lateral_length is not None and "lateral_length" in df.columns:
        df = df[df["lateral_length"] >= min_lateral_length]

    return df[uwi_column].tolist()
