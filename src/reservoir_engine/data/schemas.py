"""Data schemas for reservoir engineering data.

Uses Pydantic for validation and type safety. These models define
the canonical data structures used throughout the framework.
"""

from datetime import date
from typing import Annotated

from pydantic import BaseModel, Field


class ProductionRecord(BaseModel):
    """Monthly production record for a well.

    Attributes:
        uwi: Unique Well Identifier (API number or equivalent)
        date: Production month (first day of month)
        oil_bbl: Oil production in barrels
        gas_mcf: Gas production in MCF (thousand cubic feet)
        water_bbl: Water production in barrels
        days_on: Number of producing days in the month
    """

    uwi: str = Field(..., description="Unique Well Identifier")
    date: date = Field(..., description="Production month")
    oil_bbl: Annotated[float, Field(ge=0)] = 0.0
    gas_mcf: Annotated[float, Field(ge=0)] = 0.0
    water_bbl: Annotated[float, Field(ge=0)] = 0.0
    days_on: Annotated[int, Field(ge=0, le=31)] = 0

    @property
    def oil_rate_bpd(self) -> float:
        """Calculate daily oil rate."""
        return self.oil_bbl / self.days_on if self.days_on > 0 else 0.0

    @property
    def gas_rate_mcfd(self) -> float:
        """Calculate daily gas rate."""
        return self.gas_mcf / self.days_on if self.days_on > 0 else 0.0


class SurveyRecord(BaseModel):
    """Directional survey point for a wellbore.

    Attributes:
        uwi: Unique Well Identifier
        md: Measured depth in meters
        tvd: True vertical depth in meters
        inclination: Wellbore inclination in degrees (0=vertical)
        azimuth: Wellbore azimuth in degrees (0=north)
    """

    uwi: str = Field(..., description="Unique Well Identifier")
    md: Annotated[float, Field(ge=0)] = Field(..., description="Measured depth (m)")
    tvd: Annotated[float, Field(ge=0)] = Field(..., description="True vertical depth (m)")
    inclination: Annotated[float, Field(ge=0, le=180)] = Field(
        ..., description="Inclination (degrees)"
    )
    azimuth: Annotated[float, Field(ge=0, lt=360)] = Field(..., description="Azimuth (degrees)")


class CompletionRecord(BaseModel):
    """Completion record for a well stage.

    Attributes:
        uwi: Unique Well Identifier
        stage: Stage number (1-indexed)
        perf_top_md: Top of perforations (measured depth, m)
        perf_btm_md: Bottom of perforations (measured depth, m)
        proppant_tonnes: Proppant mass in tonnes
        fluid_m3: Fluid volume in cubic meters
        cluster_count: Number of perforation clusters
    """

    uwi: str = Field(..., description="Unique Well Identifier")
    stage: Annotated[int, Field(ge=1)] = Field(..., description="Stage number")
    perf_top_md: Annotated[float, Field(ge=0)] = Field(..., description="Perf top MD (m)")
    perf_btm_md: Annotated[float, Field(ge=0)] = Field(..., description="Perf bottom MD (m)")
    proppant_tonnes: Annotated[float, Field(ge=0)] = 0.0
    fluid_m3: Annotated[float, Field(ge=0)] = 0.0
    cluster_count: Annotated[int, Field(ge=0)] = 0

    @property
    def stage_length_m(self) -> float:
        """Calculate stage length."""
        return self.perf_btm_md - self.perf_top_md


class SteamRecord(BaseModel):
    """Steam injection record for thermal operations.

    Attributes:
        uwi: Unique Well Identifier (injector well)
        date: Injection month
        steam_tonnes: Steam injected in tonnes
        pressure_kpa: Injection pressure in kPa
        temperature_c: Steam temperature in Celsius
    """

    uwi: str = Field(..., description="Unique Well Identifier")
    date: date = Field(..., description="Injection month")
    steam_tonnes: Annotated[float, Field(ge=0)] = 0.0
    pressure_kpa: Annotated[float, Field(ge=0)] = 0.0
    temperature_c: Annotated[float, Field(ge=0)] = 0.0


# Type aliases for collections
ProductionData = list[ProductionRecord]
SurveyData = list[SurveyRecord]
CompletionData = list[CompletionRecord]
SteamData = list[SteamRecord]

