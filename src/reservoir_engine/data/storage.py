"""Storage utilities for Delta Lake and Postgres.

Provides functions for persisting and retrieving data from
both Delta Lake and Postgres/Lakebase.
"""

from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def to_delta(
    df: pd.DataFrame,
    table_path: str,
    mode: Literal["append", "overwrite", "merge"] = "append",
    spark: "SparkSession | None" = None,
    partition_by: list[str] | None = None,
) -> None:
    """Write a DataFrame to Delta Lake.

    Args:
        df: pandas DataFrame to write
        table_path: Full path to table (catalog.schema.table)
        mode: Write mode - append, overwrite, or merge
        spark: Optional SparkSession (uses active session if not provided)
        partition_by: Optional list of columns to partition by

    Example:
        >>> to_delta(df, "main.reservoir.production", mode="append")
    """
    if spark is None:
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError("No active SparkSession found")

    sdf = spark.createDataFrame(df)
    writer = sdf.write.format("delta").mode(mode)

    if partition_by:
        writer = writer.partitionBy(*partition_by)

    writer.saveAsTable(table_path)


def from_delta(
    table_path: str,
    spark: "SparkSession | None" = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read a DataFrame from Delta Lake.

    Args:
        table_path: Full path to table (catalog.schema.table)
        spark: Optional SparkSession
        columns: Optional list of columns to select

    Returns:
        pandas DataFrame
    """
    if spark is None:
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError("No active SparkSession found")

    sdf = spark.table(table_path)

    if columns:
        sdf = sdf.select(*columns)

    return sdf.toPandas()


def to_postgres(
    df: pd.DataFrame,
    table: str,
    connection_string: str,
    schema: str = "public",
    if_exists: Literal["fail", "replace", "append"] = "append",
) -> None:
    """Write a DataFrame to Postgres/Lakebase.

    Args:
        df: pandas DataFrame to write
        table: Table name
        connection_string: Postgres connection string
        schema: Schema name (default: public)
        if_exists: Behavior if table exists (fail, replace, append)

    Example:
        >>> to_postgres(df, "production", "postgresql://user:pass@host/db")
    """
    import sqlalchemy

    engine = sqlalchemy.create_engine(connection_string)
    df.to_sql(table, engine, schema=schema, if_exists=if_exists, index=False)


def from_postgres(
    table: str,
    connection_string: str,
    schema: str = "public",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read a DataFrame from Postgres/Lakebase.

    Args:
        table: Table name
        connection_string: Postgres connection string
        schema: Schema name (default: public)
        columns: Optional list of columns to select

    Returns:
        pandas DataFrame
    """
    import sqlalchemy

    engine = sqlalchemy.create_engine(connection_string)

    if columns:
        cols = ", ".join(columns)
        query = f"SELECT {cols} FROM {schema}.{table}"
    else:
        query = f"SELECT * FROM {schema}.{table}"

    return pd.read_sql(query, engine)


def sync_delta_to_postgres(
    delta_table_path: str,
    postgres_table: str,
    connection_string: str,
    spark: "SparkSession | None" = None,
    schema: str = "public",
    mode: Literal["replace", "append"] = "replace",
) -> int:
    """Sync data from Delta Lake to Postgres.

    Useful for making Delta data available in Lakebase for
    application queries.

    Args:
        delta_table_path: Source Delta table path
        postgres_table: Target Postgres table name
        connection_string: Postgres connection string
        spark: Optional SparkSession
        schema: Postgres schema name
        mode: Sync mode (replace or append)

    Returns:
        Number of rows synced
    """
    df = from_delta(delta_table_path, spark)
    to_postgres(df, postgres_table, connection_string, schema, if_exists=mode)
    return len(df)

