"""Data loading utilities for various sources.

Provides a unified interface for loading reservoir data from
Delta Lake, Postgres, CSV, and IHS sources.

Column mappings handle IHS-specific naming conventions.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql import SparkSession


# ============================================================================
# Column mappings from IHS to canonical names
# ============================================================================

WELL_HEADER_COLUMNS = {
    "uwi": "uwi",
    "surface_latitude": "surface_latitude",
    "surface_longitude": "surface_longitude",
    "bh_latitude": "bh_latitude",
    "bh_longitude": "bh_longitude",
    "hole_direction": "hole_direction",
    "current_status": "current_status",
    "current_operator": "current_operator",
    "drill_td_metric": "drill_td",
    "drill_td": "drill_td",
    "tvd_metric": "tvd",
    "tvd": "tvd",
    "lateral_length_metric": "lateral_length",
    "lateral_length": "lateral_length",
    "spud_date": "spud_date",
    "completion_date": "completion_date",
    "target_formation": "target_formation",
    "target_formation_code": "target_formation_code",
    "producing_formation": "producing_formation",
    "province_state": "province_state",
    "basin": "basin",
    "field": "field",
    "play": "play",
    "play_type": "play_type",
}

PRODUCTION_MONTHLY_COLUMNS = {
    "entity": "uwi",
    "production_date": "production_date",
    "year": "year",
    "month_num": "month_num",
    "oil": "oil",
    "gas": "gas",
    "water": "water",
    "condensate": "condensate",
    "oil_actual_daily": "oil_actual_daily",
    "gas_actual_daily": "gas_actual_daily",
    "hours": "hours",
    "boe": "boe",
    "cum_oil": "cum_oil",
    "cum_gas": "cum_gas",
    "cum_water": "cum_water",
    "cum_boe": "cum_boe",
}

PRODUCTION_SUMMARY_COLUMNS = {
    "entity": "uwi",
    "cum_oil": "cum_oil",
    "cum_gas": "cum_gas",
    "cum_water": "cum_water",
    "cum_boe": "cum_boe",
    "cum_condensate": "cum_condensate",
    "cum_production_hours": "cum_production_hours",
    "first_3_month_boe": "first_3_month_boe",
    "first_3_month_gas": "first_3_month_gas",
    "first_3_month_oil": "first_3_month_oil",
    "first_3_month_water": "first_3_month_water",
    "first_3_month_production_hours": "first_3_month_production_hours",
    "last_3_month_boe": "last_3_month_boe",
    "last_3_month_gas": "last_3_month_gas",
    "last_3_month_oil": "last_3_month_oil",
    "last_3_month_production_hours": "last_3_month_production_hours",
    "max_boe": "max_boe",
    "max_boe_date": "max_boe_date",
    "max_gas": "max_gas",
    "max_gas_date": "max_gas_date",
    "max_oil": "max_oil",
    "max_oil_date": "max_oil_date",
    "on_production_date": "on_production_date",
    "last_production_date": "last_production_date",
    "pden_status_type": "pden_status_type",
    "production_ind": "production_ind",
    "formation": "formation",
}

SURVEY_STATION_COLUMNS = {
    "uwi": "uwi",
    "survey_id": "survey_id",
    "depth_obs_no": "depth_obs_no",
    "station_md_metric": "station_md",
    "station_md": "station_md",
    "station_tvd_metric": "station_tvd",
    "station_tvd": "station_tvd",
    "inclination": "inclination",
    "azimuth": "azimuth",
    "dog_leg_severity": "dog_leg_severity",
    "latitude": "latitude",
    "longitude": "longitude",
    "x_offset_metric": "x_offset",
    "x_offset": "x_offset",
    "y_offset_metric": "y_offset",
    "y_offset": "y_offset",
    "ss_tvd_metric": "ss_tvd",
    "ss_tvd": "ss_tvd",
}

COMPLETION_COLUMNS = {
    "uwi": "uwi",
    "completion_obs_no": "completion_obs_no",
    "completion_date": "completion_date",
    "top_depth_metric": "top_depth",
    "top_depth": "top_depth",
    "base_depth_metric": "base_depth",
    "base_depth": "base_depth",
    "completion_type": "completion_type",
    "completion_method": "completion_method",
    "perf_shots": "perf_shots",
    "perf_status": "perf_status",
    "top_formation": "top_formation",
    "base_formation": "base_formation",
    "play": "play",
    "play_type": "play_type",
}


def _apply_column_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Apply column mapping to DataFrame, keeping only mapped columns that exist.
    
    Handles duplicate target names by preferring the first available source column.
    For example, if both 'drill_td' and 'drill_td_metric' map to 'drill_td',
    only one will be kept.
    """
    result_df = pd.DataFrame()
    
    # Group source columns by target name
    target_to_sources: dict[str, list[str]] = {}
    for source, target in mapping.items():
        if target not in target_to_sources:
            target_to_sources[target] = []
        target_to_sources[target].append(source)
    
    # For each target, find the first available source column
    for target, sources in target_to_sources.items():
        for source in sources:
            if source in df.columns:
                result_df[target] = df[source]
                break  # Use first available source
    
    return result_df


def _parse_dates(df: pd.DataFrame, date_columns: list[str]) -> pd.DataFrame:
    """Parse date columns to datetime.date objects."""
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df


# ============================================================================
# CSV Loaders (for local development)
# ============================================================================


def load_well_header_csv(path: str | Path) -> pd.DataFrame:
    """Load well header data from CSV.

    Args:
        path: Path to well_header.csv

    Returns:
        DataFrame with normalized column names
    """
    df = pd.read_csv(path, low_memory=False)
    df = _apply_column_mapping(df, WELL_HEADER_COLUMNS)
    df = _parse_dates(df, ["spud_date", "completion_date"])
    return df


def load_production_monthly_csv(path: str | Path) -> pd.DataFrame:
    """Load monthly production data from CSV.

    Args:
        path: Path to well_production_monthly.csv

    Returns:
        DataFrame with normalized column names
    """
    df = pd.read_csv(path, low_memory=False)
    df = _apply_column_mapping(df, PRODUCTION_MONTHLY_COLUMNS)
    df = _parse_dates(df, ["production_date"])
    return df


def load_production_summary_csv(path: str | Path) -> pd.DataFrame:
    """Load production summary data from CSV.

    Args:
        path: Path to well_production_summary.csv

    Returns:
        DataFrame with normalized column names
    """
    df = pd.read_csv(path, low_memory=False)
    df = _apply_column_mapping(df, PRODUCTION_SUMMARY_COLUMNS)
    df = _parse_dates(
        df,
        [
            "max_boe_date",
            "max_gas_date",
            "max_oil_date",
            "on_production_date",
            "last_production_date",
        ],
    )
    return df


def load_survey_stations_csv(path: str | Path) -> pd.DataFrame:
    """Load directional survey stations from CSV.

    Args:
        path: Path to well_directional_survey_stations.csv

    Returns:
        DataFrame with normalized column names
    """
    df = pd.read_csv(path, low_memory=False)
    df = _apply_column_mapping(df, SURVEY_STATION_COLUMNS)
    return df


def load_completions_csv(path: str | Path) -> pd.DataFrame:
    """Load completion data from CSV.

    Args:
        path: Path to well_completions.csv

    Returns:
        DataFrame with normalized column names
    """
    df = pd.read_csv(path, low_memory=False)
    df = _apply_column_mapping(df, COMPLETION_COLUMNS)
    df = _parse_dates(df, ["completion_date"])
    return df


def load_uwis_csv(path: str | Path) -> list[str]:
    """Load UWI list from CSV.

    Args:
        path: Path to uwis_area_of_interest.csv

    Returns:
        List of UWI strings
    """
    df = pd.read_csv(path)
    return df["uwi"].tolist()


# ============================================================================
# Delta Lake Loaders (for Databricks)
# ============================================================================


def load_from_delta(
    table_path: str,
    spark: "SparkSession | None" = None,
    filters: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load data from a Delta Lake table.

    Args:
        table_path: Full path to table (catalog.schema.table) or path
        spark: Optional SparkSession (uses active session if not provided)
        filters: Optional dictionary of column filters to apply

    Returns:
        pandas DataFrame with the loaded data

    Example:
        >>> df = load_from_delta("main.reservoir.well_header")
    """
    if spark is None:
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError("No active SparkSession found")

    sdf: SparkDataFrame = spark.table(table_path)

    if filters:
        for col, value in filters.items():
            sdf = sdf.filter(f"{col} = '{value}'")

    return sdf.toPandas()


def load_well_header_delta(
    catalog: str,
    schema: str,
    table: str = "well_header",
    spark: "SparkSession | None" = None,
) -> pd.DataFrame:
    """Load well header data from Delta Lake."""
    table_path = f"{catalog}.{schema}.{table}"
    df = load_from_delta(table_path, spark)
    df = _apply_column_mapping(df, WELL_HEADER_COLUMNS)
    df = _parse_dates(df, ["spud_date", "completion_date"])
    return df


def load_production_monthly_delta(
    catalog: str,
    schema: str,
    table: str = "well_production_monthly",
    spark: "SparkSession | None" = None,
) -> pd.DataFrame:
    """Load monthly production data from Delta Lake."""
    table_path = f"{catalog}.{schema}.{table}"
    df = load_from_delta(table_path, spark)
    df = _apply_column_mapping(df, PRODUCTION_MONTHLY_COLUMNS)
    df = _parse_dates(df, ["production_date"])
    return df


# ============================================================================
# Postgres Loaders (for Lakebase)
# ============================================================================


def load_from_postgres(
    table: str,
    connection_string: str,
    schema: str = "public",
    filters: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load data from a Postgres/Lakebase table.

    Args:
        table: Table name
        connection_string: Postgres connection string
        schema: Schema name (default: public)
        filters: Optional dictionary of column filters

    Returns:
        pandas DataFrame with the loaded data
    """
    import sqlalchemy

    engine = sqlalchemy.create_engine(connection_string)

    query = f"SELECT * FROM {schema}.{table}"
    if filters:
        conditions = " AND ".join(f"{k} = '{v}'" for k, v in filters.items())
        query += f" WHERE {conditions}"

    return pd.read_sql(query, engine)
