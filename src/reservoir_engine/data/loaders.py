"""Data loading utilities for various sources.

Provides a unified interface for loading reservoir data from
Delta Lake, Postgres, and IHS sources.
"""

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql import SparkSession


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
        >>> df = load_from_delta("main.reservoir.production")
        >>> df = load_from_delta("main.reservoir.production", filters={"uwi": "WELL-001"})
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

    # Build query with optional filters
    query = f"SELECT * FROM {schema}.{table}"
    if filters:
        conditions = " AND ".join(f"{k} = '{v}'" for k, v in filters.items())
        query += f" WHERE {conditions}"

    return pd.read_sql(query, engine)


def load_ihs_production(
    catalog: str,
    schema: str,
    table: str = "production",
    spark: "SparkSession | None" = None,
) -> pd.DataFrame:
    """Load IHS production data from Delta Lake.

    This function handles IHS-specific column mappings and normalization.

    Args:
        catalog: Unity Catalog name
        schema: Schema name
        table: Table name (default: production)
        spark: Optional SparkSession

    Returns:
        pandas DataFrame with normalized production data
    """
    table_path = f"{catalog}.{schema}.{table}"
    df = load_from_delta(table_path, spark)

    # IHS column mapping (adjust based on actual IHS schema)
    column_mapping = {
        "API_UWI": "uwi",
        "PROD_DATE": "date",
        "OIL_BBL": "oil_bbl",
        "GAS_MCF": "gas_mcf",
        "WATER_BBL": "water_bbl",
        "DAYS_ON_PROD": "days_on",
    }

    # Only rename columns that exist
    rename_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
    if rename_cols:
        df = df.rename(columns=rename_cols)

    return df


def load_ihs_surveys(
    catalog: str,
    schema: str,
    table: str = "surveys",
    spark: "SparkSession | None" = None,
) -> pd.DataFrame:
    """Load IHS directional survey data.

    Args:
        catalog: Unity Catalog name
        schema: Schema name
        table: Table name (default: surveys)
        spark: Optional SparkSession

    Returns:
        pandas DataFrame with normalized survey data
    """
    table_path = f"{catalog}.{schema}.{table}"
    df = load_from_delta(table_path, spark)

    column_mapping = {
        "API_UWI": "uwi",
        "MEASURED_DEPTH": "md",
        "TRUE_VERTICAL_DEPTH": "tvd",
        "INCLINATION": "inclination",
        "AZIMUTH": "azimuth",
    }

    rename_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
    if rename_cols:
        df = df.rename(columns=rename_cols)

    return df


def load_ihs_completions(
    catalog: str,
    schema: str,
    table: str = "completions",
    spark: "SparkSession | None" = None,
) -> pd.DataFrame:
    """Load IHS completion data.

    Args:
        catalog: Unity Catalog name
        schema: Schema name
        table: Table name (default: completions)
        spark: Optional SparkSession

    Returns:
        pandas DataFrame with normalized completion data
    """
    table_path = f"{catalog}.{schema}.{table}"
    df = load_from_delta(table_path, spark)

    column_mapping = {
        "API_UWI": "uwi",
        "STAGE_NUM": "stage",
        "PERF_TOP_MD": "perf_top_md",
        "PERF_BTM_MD": "perf_btm_md",
        "PROPPANT_LBS": "proppant_tonnes",  # Will need conversion
        "FLUID_GAL": "fluid_m3",  # Will need conversion
        "CLUSTER_COUNT": "cluster_count",
    }

    rename_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
    if rename_cols:
        df = df.rename(columns=rename_cols)

    # Unit conversions
    if "proppant_tonnes" in df.columns:
        df["proppant_tonnes"] = df["proppant_tonnes"] * 0.000453592  # lbs to tonnes
    if "fluid_m3" in df.columns:
        df["fluid_m3"] = df["fluid_m3"] * 0.00378541  # gallons to m3

    return df


def load_ihs_steam(
    catalog: str,
    schema: str,
    table: str = "steam",
    spark: "SparkSession | None" = None,
) -> pd.DataFrame:
    """Load IHS steam injection data (for thermal assets).

    Args:
        catalog: Unity Catalog name
        schema: Schema name
        table: Table name (default: steam)
        spark: Optional SparkSession

    Returns:
        pandas DataFrame with normalized steam data
    """
    table_path = f"{catalog}.{schema}.{table}"
    df = load_from_delta(table_path, spark)

    column_mapping = {
        "API_UWI": "uwi",
        "INJ_DATE": "date",
        "STEAM_TONNES": "steam_tonnes",
        "INJ_PRESSURE_KPA": "pressure_kpa",
        "STEAM_TEMP_C": "temperature_c",
    }

    rename_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
    if rename_cols:
        df = df.rename(columns=rename_cols)

    return df

