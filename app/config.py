"""App configuration for the Reservoir Engine Streamlit application.

Supports both test mode (CSV files) and prod mode (Lakebase).
"""

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict


class AppConfig(BaseModel):
    """Configuration for the Reservoir Engine app."""

    model_config = ConfigDict(extra="ignore")

    # App settings
    mode: Literal["test", "prod"] = "test"
    default_user: str = "engineer"
    lakebase_instance: str = ""
    lakebase_dbname: str = "databricks_postgres"

    # Data paths (for test mode)
    data_dir: str = "data"

    # Lakebase table names (for prod mode)
    well_header_table: str = "well_header"
    production_monthly_table: str = "well_production_monthly"
    production_summary_table: str = "well_production_summary"
    completions_table: str = "well_completions"
    surveys_table: str = "well_directional_survey_stations"
    uwis_table: str = "uwis_area_of_interest"

    # UI
    ui_title: str = "Reservoir Engine"
    ui_icon: str = ""

    # Analysis defaults
    default_h3_resolution: int = 9
    default_forecast_months: int = 360
    default_economic_limit: float = 5.0  # M3/month

    @property
    def is_test_mode(self) -> bool:
        return self.mode == "test"

    @property
    def is_prod_mode(self) -> bool:
        return self.mode == "prod"

    @classmethod
    def from_yaml(cls, config_path: Optional[str] = None) -> "AppConfig":
        """Load from config.yaml app section."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            # Return default config if no file
            return cls()

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        app = data.get("app", {})

        flat = {
            "mode": app.get("mode", "test"),
            "default_user": app.get("default_user", "engineer"),
            "lakebase_instance": app.get("lakebase_instance", ""),
            "lakebase_dbname": app.get("lakebase_dbname", "databricks_postgres"),
            "data_dir": app.get("data_dir", "data"),
        }

        if "tables" in app:
            tables = app["tables"]
            flat["well_header_table"] = tables.get("well_header", "well_header")
            flat["production_monthly_table"] = tables.get(
                "production_monthly", "well_production_monthly"
            )
            flat["production_summary_table"] = tables.get(
                "production_summary", "well_production_summary"
            )

        if "ui" in app:
            ui = app["ui"]
            flat["ui_title"] = ui.get("title", "Reservoir Engine")
            flat["ui_icon"] = ui.get("icon", "")

        if "analysis" in app:
            analysis = app["analysis"]
            flat["default_h3_resolution"] = analysis.get("h3_resolution", 9)
            flat["default_forecast_months"] = analysis.get("forecast_months", 360)
            flat["default_economic_limit"] = analysis.get("economic_limit", 5.0)

        return cls.model_validate(flat)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load app configuration."""
    return AppConfig.from_yaml(config_path)
