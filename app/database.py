"""Database backend abstraction for test and prod modes.

- test: CSVBackend loading from local CSV files
- prod: LakebaseBackend with PostgreSQL on Databricks
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional
import uuid

import pandas as pd

if TYPE_CHECKING:
    from .config import AppConfig

logger = logging.getLogger(__name__)


class DatabaseBackend(ABC):
    """Abstract base class for database backends."""

    @abstractmethod
    def get_well_header(self) -> pd.DataFrame:
        """Get well header data. Pre-filtered to FLOWING SHALE GAS wells for performance."""
        pass

    @abstractmethod
    def get_production_monthly(self, uwis: list[str] | None = None) -> pd.DataFrame:
        """Get monthly production data."""
        pass

    @abstractmethod
    def get_production_summary(self) -> pd.DataFrame:
        """Get production summary data."""
        pass

    @abstractmethod
    def get_completions(self, uwis: list[str] | None = None) -> pd.DataFrame:
        """Get completion data."""
        pass

    @abstractmethod
    def get_surveys(self, uwis: list[str] | None = None) -> pd.DataFrame:
        """Get survey data."""
        pass

    @abstractmethod
    def get_uwis(self) -> list[str]:
        """Get list of UWIs in area of interest."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if backend is connected."""
        pass


class CSVBackend(DatabaseBackend):
    """CSV backend for test mode using local files."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self._cache: Dict[str, pd.DataFrame] = {}

        # Apply column mappings from reservoir_engine
        from reservoir_engine.data.loaders import (
            WELL_HEADER_COLUMNS,
            PRODUCTION_MONTHLY_COLUMNS,
            PRODUCTION_SUMMARY_COLUMNS,
            SURVEY_STATION_COLUMNS,
            COMPLETION_COLUMNS,
        )

        self._column_mappings = {
            "well_header": WELL_HEADER_COLUMNS,
            "production_monthly": PRODUCTION_MONTHLY_COLUMNS,
            "production_summary": PRODUCTION_SUMMARY_COLUMNS,
            "surveys": SURVEY_STATION_COLUMNS,
            "completions": COMPLETION_COLUMNS,
        }

    def _load_csv(self, filename: str, mapping_key: str | None = None) -> pd.DataFrame:
        """Load CSV with caching and optional column mapping."""
        if filename in self._cache:
            return self._cache[filename].copy()

        path = self.data_dir / filename
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return pd.DataFrame()

        df = pd.read_csv(path, low_memory=False)

        # Apply column mapping if specified, handling duplicate targets
        if mapping_key and mapping_key in self._column_mappings:
            mapping = self._column_mappings[mapping_key]
            df = self._apply_column_mapping(df, mapping)

        self._cache[filename] = df
        return df.copy()

    def _apply_column_mapping(self, df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
        """Apply column mapping, handling duplicate target names.

        When multiple source columns map to the same target, prefer the first available.
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
                    result_df[target] = df[source].values
                    break  # Use first available source
        
        return result_df

    def get_well_header(self) -> pd.DataFrame:
        df = self._load_csv("well_header.csv", "well_header")
        # Parse dates
        for col in ["spud_date", "completion_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
        
        # Filter to FLOWING SHALE GAS wells for performance
        if "current_status" in df.columns:
            df = df[df["current_status"] == "FLOWING SHALE GAS"]
        
        return df

    def get_production_monthly(self, uwis: list[str] | None = None) -> pd.DataFrame:
        df = self._load_csv("well_production_monthly.csv", "production_monthly")
        if "production_date" in df.columns:
            df["production_date"] = pd.to_datetime(df["production_date"], errors="coerce").dt.date
        if uwis is not None:
            df = df[df["uwi"].isin(uwis)]
        return df

    def get_production_summary(self) -> pd.DataFrame:
        df = self._load_csv("well_production_summary.csv", "production_summary")
        return df

    def get_completions(self, uwis: list[str] | None = None) -> pd.DataFrame:
        df = self._load_csv("well_completions.csv", "completions")
        if uwis is not None and "uwi" in df.columns:
            df = df[df["uwi"].isin(uwis)]
        return df

    def get_surveys(self, uwis: list[str] | None = None) -> pd.DataFrame:
        df = self._load_csv("well_directional_survey_stations.csv", "surveys")
        if uwis is not None and "uwi" in df.columns:
            df = df[df["uwi"].isin(uwis)]
            return df

    def get_uwis(self) -> list[str]:
        df = self._load_csv("uwis_area_of_interest.csv")
        if "uwi" in df.columns:
            return df["uwi"].tolist()
        return []

    def is_connected(self) -> bool:
        return self.data_dir.exists()


class LakebaseBackend(DatabaseBackend):
    """PostgreSQL backend using Databricks Lakebase."""

    def __init__(self, config: "AppConfig"):
        self.config = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from databricks.sdk import WorkspaceClient
            self._client = WorkspaceClient()
        return self._client

    @contextmanager
    def get_connection(self):
        """Get a database connection using psycopg2."""
        import psycopg2

        try:
            db_instance = self.client.database.get_database_instance(
                name=self.config.lakebase_instance
            )
            cred = self.client.database.generate_database_credential(
                request_id=str(uuid.uuid4()),
                instance_names=[self.config.lakebase_instance],
            )

            user = self.client.current_user.me()
            if hasattr(user, "emails") and user.emails:
                user_email = user.emails[0].value
            elif hasattr(user, "user_name") and user.user_name:
                user_email = user.user_name
            else:
                raise ValueError("Could not determine user email")

            conn = psycopg2.connect(
                host=db_instance.read_write_dns,
                dbname=self.config.lakebase_dbname,
                user=user_email,
                password=cred.token,
                sslmode="require",
            )

            try:
                yield conn
            finally:
                conn.close()

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise

    def _execute_query(self, query: str) -> pd.DataFrame:
        with self.get_connection() as conn:
            return pd.read_sql(query, conn)

    def get_well_header(self) -> pd.DataFrame:
        query = f"SELECT * FROM {self.config.well_header_table}"
        query += " WHERE current_status = 'FLOWING SHALE GAS'"
        return self._execute_query(query)

    def get_production_monthly(self, uwis: list[str] | None = None) -> pd.DataFrame:
        query = f"SELECT * FROM {self.config.production_monthly_table}"
        if uwis:
            uwi_list = ", ".join(f"'{u}'" for u in uwis)
            query += f" WHERE entity IN ({uwi_list})"
        return self._execute_query(query)

    def get_production_summary(self) -> pd.DataFrame:
        query = f"SELECT * FROM {self.config.production_summary_table}"
        return self._execute_query(query)

    def get_completions(self, uwis: list[str] | None = None) -> pd.DataFrame:
        query = f"SELECT * FROM {self.config.completions_table}"
        if uwis:
            uwi_list = ", ".join(f"'{u}'" for u in uwis)
            query += f" WHERE uwi IN ({uwi_list})"
        return self._execute_query(query)

    def get_surveys(self, uwis: list[str] | None = None) -> pd.DataFrame:
        query = f"SELECT * FROM {self.config.surveys_table}"
        if uwis:
            uwi_list = ", ".join(f"'{u}'" for u in uwis)
            query += f" WHERE uwi IN ({uwi_list})"
        return self._execute_query(query)

    def get_uwis(self) -> list[str]:
        query = f"SELECT uwi FROM {self.config.uwis_table}"
        df = self._execute_query(query)
        return df["uwi"].tolist()

    def is_connected(self) -> bool:
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    return True
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False


def create_backend(config: "AppConfig") -> DatabaseBackend:
    """Factory function to create the appropriate backend."""
    if config.is_test_mode:
        logger.info("Creating CSVBackend for test mode")
        return CSVBackend(config.data_dir)

    if config.is_prod_mode:
        if not config.lakebase_instance:
            raise ValueError("lakebase_instance required for prod mode")
        logger.info(f"Creating LakebaseBackend (instance: {config.lakebase_instance})")
        return LakebaseBackend(config)

    raise ValueError(f"Unknown mode: {config.mode}")


# Global backend instance
_backend: Optional[DatabaseBackend] = None


def get_backend() -> DatabaseBackend:
    """Get the global database backend."""
    global _backend
    if _backend is None:
        raise RuntimeError("Backend not initialized. Call init_backend() first.")
    return _backend


def init_backend(config: "AppConfig") -> DatabaseBackend:
    """Initialize the global database backend."""
    global _backend
    _backend = create_backend(config)
    return _backend


def reset_backend() -> None:
    """Reset the global backend."""
    global _backend
    _backend = None
