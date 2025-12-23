"""Data schemas for reservoir engineering data.

Uses Pydantic for validation and type safety. These models define
the canonical data structures used throughout the framework.

Schema mappings are based on IHS/Enverus data formats.
"""

from datetime import date, datetime
from typing import Annotated

from pydantic import BaseModel, Field


class WellHeader(BaseModel):
    """Well header record - master well information.

    Based on IHS well_header schema. Contains static well attributes
    including location, completion info, and well classification.
    """

    uwi: str = Field(..., description="Unique Well Identifier")

    # Location - Surface
    surface_latitude: float | None = Field(None, description="Surface latitude (WGS84)")
    surface_longitude: float | None = Field(None, description="Surface longitude (WGS84)")

    # Location - Bottom Hole
    bh_latitude: float | None = Field(None, description="Bottom hole latitude")
    bh_longitude: float | None = Field(None, description="Bottom hole longitude")

    # Well Classification
    hole_direction: str | None = Field(None, description="VERTICAL, HORIZONTAL, DIRECTIONAL")
    current_status: str | None = Field(None, description="Current well status")
    current_operator: str | None = Field(None, description="Current operator name")

    # Depths (metric)
    drill_td: float | None = Field(None, description="Drill total depth (m)")
    tvd: float | None = Field(None, description="True vertical depth (m)")
    lateral_length: float | None = Field(None, description="Lateral length (m)")

    # Dates
    spud_date: date | None = Field(None, description="Spud date")
    completion_date: date | None = Field(None, description="Completion date")

    # Formation
    target_formation: str | None = Field(None, description="Target formation name")
    target_formation_code: str | None = Field(None, description="Target formation code")
    producing_formation: str | None = Field(None, description="Producing formation")

    # Geography
    province_state: str | None = Field(None, description="Province or state")
    basin: str | None = Field(None, description="Basin name")
    field: str | None = Field(None, description="Field name")
    play: str | None = Field(None, description="Play name")
    play_type: str | None = Field(None, description="Play type (SHALE, TIGHT, etc.)")

    @property
    def is_horizontal(self) -> bool:
        """Check if well is horizontal."""
        return self.hole_direction == "HORIZONTAL"


class ProductionMonthly(BaseModel):
    """Monthly production record.

    Based on IHS well_production_monthly schema.
    Note: 'entity' in IHS data maps to 'uwi' here.
    """

    uwi: str = Field(..., description="Unique Well Identifier (entity in IHS)")
    production_date: date = Field(..., description="Production month (first day)")
    year: int = Field(..., description="Production year")
    month_num: int = Field(..., description="Month number (1-12)")

    # Production volumes (metric)
    oil: Annotated[float, Field(ge=0)] = Field(0.0, description="Oil production (M3)")
    gas: Annotated[float, Field(ge=0)] = Field(0.0, description="Gas production (E3M3)")
    water: Annotated[float, Field(ge=0)] = Field(0.0, description="Water production (M3)")
    condensate: Annotated[float, Field(ge=0)] = Field(0.0, description="Condensate (M3)")

    # Rates (calculated daily)
    oil_actual_daily: float | None = Field(None, description="Oil rate (M3/D)")
    gas_actual_daily: float | None = Field(None, description="Gas rate (E3M3/D)")

    # Operating
    hours: Annotated[float, Field(ge=0)] = Field(0.0, description="Production hours")
    boe: float = Field(0.0, description="Barrels of oil equivalent")

    # Cumulative
    cum_oil: float = Field(0.0, description="Cumulative oil (M3)")
    cum_gas: float = Field(0.0, description="Cumulative gas (E3M3)")
    cum_water: float = Field(0.0, description="Cumulative water (M3)")
    cum_boe: float = Field(0.0, description="Cumulative BOE")

    @property
    def days_on(self) -> float:
        """Calculate producing days from hours."""
        return self.hours / 24.0 if self.hours > 0 else 0.0


class ProductionSummary(BaseModel):
    """Production summary metrics per well.

    Based on IHS well_production_summary schema.
    Contains EUR-type metrics and production statistics.
    """

    uwi: str = Field(..., description="Unique Well Identifier (entity in IHS)")

    # Cumulative production
    cum_oil: float = Field(0.0, description="Cumulative oil (M3)")
    cum_gas: float = Field(0.0, description="Cumulative gas (E3M3)")
    cum_water: float = Field(0.0, description="Cumulative water (M3)")
    cum_boe: float = Field(0.0, description="Cumulative BOE")
    cum_condensate: float = Field(0.0, description="Cumulative condensate (M3)")
    cum_production_hours: float = Field(0.0, description="Total production hours")

    # First 3 month metrics (IP)
    first_3_month_boe: float | None = None
    first_3_month_gas: float | None = None
    first_3_month_oil: float | None = None
    first_3_month_water: float | None = None
    first_3_month_production_hours: float | None = None

    # Last 3 month metrics
    last_3_month_boe: float | None = None
    last_3_month_gas: float | None = None
    last_3_month_oil: float | None = None
    last_3_month_production_hours: float | None = None

    # Peak production
    max_boe: float | None = None
    max_boe_date: date | None = None
    max_gas: float | None = None
    max_gas_date: date | None = None
    max_oil: float | None = None
    max_oil_date: date | None = None

    # Date ranges
    on_production_date: date | None = None
    last_production_date: date | None = None

    # Status
    pden_status_type: str | None = None
    production_ind: str | None = None  # Y/N
    formation: str | None = None


class SurveyStation(BaseModel):
    """Directional survey station record.

    Based on IHS well_directional_survey_stations schema.
    One record per survey station (depth point).
    """

    uwi: str = Field(..., description="Unique Well Identifier")
    survey_id: str | None = Field(None, description="Survey ID")
    depth_obs_no: int | None = Field(None, description="Depth observation number")

    # Measured depth and TVD
    station_md: Annotated[float, Field(ge=0)] = Field(..., description="Measured depth (m)")
    station_tvd: Annotated[float, Field(ge=0)] = Field(..., description="True vertical depth (m)")

    # Orientation
    inclination: Annotated[float, Field(ge=0, le=180)] = Field(
        0.0, description="Inclination (degrees, 0=vertical)"
    )
    azimuth: Annotated[float, Field(ge=0, lt=360)] = Field(
        0.0, description="Azimuth (degrees, 0=north)"
    )
    dog_leg_severity: float | None = Field(None, description="Dog leg severity (deg/30m)")

    # Position
    latitude: float | None = Field(None, description="Latitude at station")
    longitude: float | None = Field(None, description="Longitude at station")
    x_offset: float | None = Field(None, description="X offset from surface (m)")
    y_offset: float | None = Field(None, description="Y offset from surface (m)")

    # Subsea TVD
    ss_tvd: float | None = Field(None, description="Subsea TVD (m)")


class Completion(BaseModel):
    """Well completion record.

    Based on IHS well_completions schema.
    One record per completion interval/event.
    """

    uwi: str = Field(..., description="Unique Well Identifier")
    completion_obs_no: int | None = Field(None, description="Completion observation number")
    completion_date: date | None = Field(None, description="Completion date")

    # Depths (metric)
    top_depth: Annotated[float, Field(ge=0)] | None = Field(
        None, description="Top depth (m)", alias="top_depth_metric"
    )
    base_depth: Annotated[float, Field(ge=0)] | None = Field(
        None, description="Base depth (m)", alias="base_depth_metric"
    )

    # Completion details
    completion_type: str | None = Field(None, description="JET PERFORATION, FRACTURED, etc.")
    completion_method: str | None = Field(None, description="Completion method")
    perf_shots: float | None = Field(None, description="Perforation shot count")
    perf_status: str | None = Field(None, description="OPEN, SQUEEZED, etc.")

    # Formation
    top_formation: str | None = Field(None, description="Top formation name")
    base_formation: str | None = Field(None, description="Base formation name")

    # Location
    latitude: float | None = None
    longitude: float | None = None
    play: str | None = None
    play_type: str | None = None

    @property
    def interval_length(self) -> float | None:
        """Calculate completion interval length."""
        if self.top_depth is not None and self.base_depth is not None:
            return self.base_depth - self.top_depth
        return None


class AreaOfInterest(BaseModel):
    """Simple UWI list for area of interest selection."""

    uwi: str = Field(..., description="Unique Well Identifier")


# Type aliases for collections
WellHeaders = list[WellHeader]
ProductionRecords = list[ProductionMonthly]
SurveyStations = list[SurveyStation]
Completions = list[Completion]


# Constants for unit conversion (metric to imperial)
class UnitConversion:
    """Unit conversion constants."""

    M3_TO_BBL = 6.28981  # Cubic meters to barrels
    E3M3_TO_MCF = 35.3147  # Thousand cubic meters to MCF
    M_TO_FT = 3.28084  # Meters to feet
