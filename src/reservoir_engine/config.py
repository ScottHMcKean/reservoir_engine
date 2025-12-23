"""Configuration management for Reservoir Engine.

Provides a simple, environment-aware configuration system.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class SparkConfig:
    """Spark session configuration."""

    app_name: str = "reservoir_engine"
    # Additional Spark configs can be added here


@dataclass(frozen=True)
class StorageConfig:
    """Storage layer configuration."""

    # Delta Lake settings
    delta_catalog: str = "main"
    delta_schema: str = "reservoir"

    # Postgres/Lakebase settings
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "reservoir"
    postgres_schema: str = "public"


@dataclass(frozen=True)
class Config:
    """Main configuration container."""

    environment: Literal["dev", "staging", "prod"] = "dev"
    spark: SparkConfig = field(default_factory=SparkConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)


def get_config() -> Config:
    """Get configuration from environment.

    In a full implementation, this would read from environment variables,
    Databricks secrets, or a configuration file.
    """
    return Config()

