"""Shared test fixtures for Reservoir Engine.

Provides reusable fixtures based on IHS data schemas.
All fixtures generate synthetic data matching the actual
IHS column naming conventions.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# Well Header Fixtures
# ============================================================================


@pytest.fixture
def sample_well_header_df() -> pd.DataFrame:
    """Create sample well header data matching IHS schema."""
    np.random.seed(42)

    return pd.DataFrame(
        [
            {
                "uwi": "100010106021W500",
                "surface_latitude": 54.087308906,
                "surface_longitude": -116.338507275,
                "bh_latitude": 54.087308906,
                "bh_longitude": -116.338507275,
                "hole_direction": "VERTICAL",
                "current_status": "OPERATING",
                "current_operator": "OPERATOR A",
                "drill_td": 2930.0,
                "tvd": 2930.0,
                "lateral_length": None,
                "spud_date": date(2020, 1, 15),
                "completion_date": date(2020, 2, 20),
                "target_formation": "DUVERNAY",
                "target_formation_code": "DUVERN",
                "producing_formation": "DUVERNAY",
                "province_state": "ALBERTA",
                "basin": "ALBERTA BASIN",
                "field": "KAYBOB",
                "play": "DUVERNAY",
                "play_type": "SHALE",
            },
            {
                "uwi": "102133506219W500",
                "surface_latitude": 54.383286180,
                "surface_longitude": -116.697578684,
                "bh_latitude": 54.381893693,
                "bh_longitude": -116.698522242,
                "hole_direction": "HORIZONTAL",
                "current_status": "OPERATING",
                "current_operator": "OPERATOR B",
                "drill_td": 5540.0,
                "tvd": 3340.12,
                "lateral_length": 2061.9,
                "spud_date": date(2019, 6, 1),
                "completion_date": date(2019, 8, 15),
                "target_formation": "DUVERNAY",
                "target_formation_code": "DUVERN",
                "producing_formation": "DUVERNAY",
                "province_state": "ALBERTA",
                "basin": "ALBERTA BASIN",
                "field": "KAYBOB SOUTH",
                "play": "DUVERNAY",
                "play_type": "SHALE",
            },
            {
                "uwi": "100021506320W500",
                "surface_latitude": 54.443918206,
                "surface_longitude": -116.933549346,
                "bh_latitude": 54.444161188,
                "bh_longitude": -116.933375354,
                "hole_direction": "DIRECTIONAL",
                "current_status": "OPERATING",
                "current_operator": "OPERATOR C",
                "drill_td": 3164.0,
                "tvd": 3163.28,
                "lateral_length": None,
                "spud_date": date(2018, 11, 15),
                "completion_date": date(2018, 12, 20),
                "target_formation": "DUVERNAY",
                "target_formation_code": "DUVERN",
                "producing_formation": "DUVERNAY",
                "province_state": "ALBERTA",
                "basin": "ALBERTA BASIN",
                "field": "KAYBOB",
                "play": "DUVERNAY",
                "play_type": "SHALE",
            },
        ]
    )


# ============================================================================
# Production Fixtures
# ============================================================================


@pytest.fixture
def sample_production_monthly_df() -> pd.DataFrame:
    """Create sample monthly production data matching IHS schema.

    Generates 24 months of production for 3 wells with realistic decline.
    """
    np.random.seed(42)

    wells = ["100010106021W500", "102133506219W500", "100021506320W500"]
    months = 24
    records = []

    for uwi in wells:
        # Different initial rates for each well
        if "102133" in uwi:  # Horizontal well - higher production
            qi_oil = np.random.uniform(100, 200)  # M3/month
            qi_gas = np.random.uniform(500, 1000)  # E3M3/month
        else:
            qi_oil = np.random.uniform(30, 80)
            qi_gas = np.random.uniform(200, 400)

        di = np.random.uniform(0.08, 0.15)
        b = np.random.uniform(0.5, 1.2)

        cum_oil = 0.0
        cum_gas = 0.0
        cum_water = 0.0

        for month in range(months):
            # Arps hyperbolic decline with noise
            decline_factor = 1 / (1 + b * di * month) ** (1 / b)
            oil = qi_oil * decline_factor * np.random.uniform(0.85, 1.15)
            gas = qi_gas * decline_factor * np.random.uniform(0.85, 1.15)
            water = oil * 0.3 * (1 + month * 0.05)  # Increasing water cut

            prod_date = date(2020, 1, 1) + timedelta(days=30 * month)
            hours = np.random.uniform(680, 744)  # Hours producing

            cum_oil += oil
            cum_gas += gas
            cum_water += water

            # Calculate BOE (6:1 gas conversion)
            boe = oil + gas * 1000 / 6  # E3M3 to M3 equivalent

            records.append(
                {
                    "uwi": uwi,
                    "production_date": prod_date,
                    "year": prod_date.year,
                    "month_num": prod_date.month,
                    "oil": max(0, oil),
                    "gas": max(0, gas),
                    "water": max(0, water),
                    "condensate": 0.0,
                    "hours": hours,
                    "boe": boe,
                    "cum_oil": cum_oil,
                    "cum_gas": cum_gas,
                    "cum_water": cum_water,
                    "cum_boe": cum_oil + cum_gas * 1000 / 6,
                }
            )

    return pd.DataFrame(records)


@pytest.fixture
def sample_production_summary_df() -> pd.DataFrame:
    """Create sample production summary data matching IHS schema."""
    return pd.DataFrame(
        [
            {
                "uwi": "100010106021W500",
                "cum_oil": 1500.0,
                "cum_gas": 8000.0,
                "cum_water": 500.0,
                "cum_boe": 2833333.0,
                "cum_condensate": 10.0,
                "cum_production_hours": 15000.0,
                "first_3_month_boe": 500000.0,
                "first_3_month_gas": 1500.0,
                "first_3_month_oil": 250.0,
                "first_3_month_water": 50.0,
                "first_3_month_production_hours": 2000.0,
                "last_3_month_boe": 100000.0,
                "last_3_month_gas": 300.0,
                "last_3_month_oil": 50.0,
                "last_3_month_production_hours": 2000.0,
                "max_boe": 300000.0,
                "max_boe_date": date(2020, 2, 1),
                "max_gas": 600.0,
                "max_gas_date": date(2020, 2, 1),
                "max_oil": 100.0,
                "max_oil_date": date(2020, 2, 1),
                "on_production_date": date(2020, 1, 1),
                "last_production_date": date(2021, 12, 31),
                "pden_status_type": "OPERATING",
                "production_ind": "Y",
                "formation": "DUVERNAY",
            },
            {
                "uwi": "102133506219W500",
                "cum_oil": 3500.0,
                "cum_gas": 18000.0,
                "cum_water": 1200.0,
                "cum_boe": 6500000.0,
                "cum_condensate": 25.0,
                "cum_production_hours": 17000.0,
                "first_3_month_boe": 1200000.0,
                "first_3_month_gas": 3500.0,
                "first_3_month_oil": 600.0,
                "first_3_month_water": 100.0,
                "first_3_month_production_hours": 2100.0,
                "last_3_month_boe": 200000.0,
                "last_3_month_gas": 600.0,
                "last_3_month_oil": 100.0,
                "last_3_month_production_hours": 2100.0,
                "max_boe": 700000.0,
                "max_boe_date": date(2019, 10, 1),
                "max_gas": 1400.0,
                "max_gas_date": date(2019, 10, 1),
                "max_oil": 200.0,
                "max_oil_date": date(2019, 10, 1),
                "on_production_date": date(2019, 9, 1),
                "last_production_date": date(2021, 12, 31),
                "pden_status_type": "OPERATING",
                "production_ind": "Y",
                "formation": "DUVERNAY",
            },
            {
                "uwi": "100021506320W500",
                "cum_oil": 1200.0,
                "cum_gas": 6000.0,
                "cum_water": 400.0,
                "cum_boe": 2200000.0,
                "cum_condensate": 8.0,
                "cum_production_hours": 14000.0,
                "first_3_month_boe": 400000.0,
                "first_3_month_gas": 1200.0,
                "first_3_month_oil": 200.0,
                "first_3_month_water": 40.0,
                "first_3_month_production_hours": 1900.0,
                "last_3_month_boe": 80000.0,
                "last_3_month_gas": 240.0,
                "last_3_month_oil": 40.0,
                "last_3_month_production_hours": 1900.0,
                "max_boe": 250000.0,
                "max_boe_date": date(2019, 2, 1),
                "max_gas": 500.0,
                "max_gas_date": date(2019, 2, 1),
                "max_oil": 80.0,
                "max_oil_date": date(2019, 2, 1),
                "on_production_date": date(2019, 1, 1),
                "last_production_date": date(2021, 12, 31),
                "pden_status_type": "OPERATING",
                "production_ind": "Y",
                "formation": "DUVERNAY",
            },
        ]
    )


# ============================================================================
# Survey Fixtures
# ============================================================================


@pytest.fixture
def sample_survey_stations_df() -> pd.DataFrame:
    """Create sample directional survey data matching IHS schema."""
    records = []

    # Horizontal well survey
    uwi = "102133506219W500"
    survey_id = "001"

    # Vertical section (0-1000m)
    for i, md in enumerate(range(0, 1000, 100)):
        records.append(
            {
                "uwi": uwi,
                "survey_id": survey_id,
                "depth_obs_no": i + 1,
                "station_md": float(md),
                "station_tvd": float(md),
                "inclination": 0.0,
                "azimuth": 0.0,
                "dog_leg_severity": 0.0,
                "latitude": 54.383286180,
                "longitude": -116.697578684,
                "x_offset": 0.0,
                "y_offset": 0.0,
                "ss_tvd": 867.20 - md,
            }
        )

    # Build section (1000-2000m, building to horizontal)
    for i, md in enumerate(range(1000, 2000, 100)):
        inc = min(90, 10 + i * 9)
        tvd = 1000 + (md - 1000) * np.cos(np.radians(inc / 2))
        records.append(
            {
                "uwi": uwi,
                "survey_id": survey_id,
                "depth_obs_no": 10 + i + 1,
                "station_md": float(md),
                "station_tvd": float(tvd),
                "inclination": float(inc),
                "azimuth": 314.0,
                "dog_leg_severity": 6.0,
                "latitude": 54.383286180 - 0.0001 * i,
                "longitude": -116.697578684 - 0.0001 * i,
                "x_offset": float(i * 10),
                "y_offset": float(i * -15),
                "ss_tvd": 867.20 - tvd,
            }
        )

    # Horizontal section (2000-5000m)
    for i, md in enumerate(range(2000, 5500, 100)):
        records.append(
            {
                "uwi": uwi,
                "survey_id": survey_id,
                "depth_obs_no": 20 + i + 1,
                "station_md": float(md),
                "station_tvd": 3000.0,  # Constant TVD
                "inclination": 90.0,
                "azimuth": 314.0,
                "dog_leg_severity": 0.0,
                "latitude": 54.381893693 + 0.00005 * i,
                "longitude": -116.698522242 - 0.00003 * i,
                "x_offset": float(100 + i * 30),
                "y_offset": float(-150 - i * 20),
                "ss_tvd": 867.20 - 3000.0,
            }
        )

    return pd.DataFrame(records)


# ============================================================================
# Completion Fixtures
# ============================================================================


@pytest.fixture
def sample_completions_df() -> pd.DataFrame:
    """Create sample completion data matching IHS schema."""
    records = []

    # Horizontal well completions (multi-stage frac)
    uwi = "102133506219W500"
    for stage in range(1, 31):
        records.append(
            {
                "uwi": uwi,
                "completion_obs_no": stage,
                "completion_date": date(2019, 8, 15),
                "top_depth": 2000.0 + stage * 100,
                "base_depth": 2050.0 + stage * 100,
                "completion_type": "FRACTURED",
                "completion_method": "PLUG AND PERF",
                "perf_shots": float(np.random.randint(15, 25)),
                "perf_status": "OPEN",
                "top_formation": "DUVERNAY",
                "base_formation": "DUVERNAY",
                "play": "DUVERNAY",
                "play_type": "SHALE",
            }
        )

    # Vertical well completions (single zone)
    uwi = "100010106021W500"
    records.append(
        {
            "uwi": uwi,
            "completion_obs_no": 1,
            "completion_date": date(2020, 2, 20),
            "top_depth": 2800.0,
            "base_depth": 2900.0,
            "completion_type": "JET PERFORATION",
            "completion_method": None,
            "perf_shots": 50.0,
            "perf_status": "OPEN",
            "top_formation": "DUVERNAY",
            "base_formation": "DUVERNAY",
            "play": "DUVERNAY",
            "play_type": "SHALE",
        }
    )

    return pd.DataFrame(records)


# ============================================================================
# UWI List Fixtures
# ============================================================================


@pytest.fixture
def sample_uwis() -> list[str]:
    """Sample UWI list for testing."""
    return [
        "100010106021W500",
        "102133506219W500",
        "100021506320W500",
    ]


# ============================================================================
# DCA Parameter Fixtures
# ============================================================================


@pytest.fixture
def arps_params():
    """Create sample Arps parameters for testing."""
    from reservoir_engine.dca.models import ArpsParams

    return ArpsParams(qi=100.0, di=0.12, b=0.8)


@pytest.fixture
def butler_params():
    """Create sample Butler parameters for testing."""
    from reservoir_engine.dca.models import ButlerParams

    return ButlerParams(q_peak=50.0, m=0.1, tau=12, sor=3.0)


# ============================================================================
# Thermal Production Fixtures (for Butler model testing)
# ============================================================================


@pytest.fixture
def sample_thermal_production_df() -> pd.DataFrame:
    """Create sample SAGD production data with ramp-up and decline."""
    np.random.seed(42)

    records = []
    uwi = "PAD-01-SAGD"

    q_peak = 50.0  # M3/month peak
    tau = 12  # Months to peak
    m = 0.08  # Decline rate

    cum_oil = 0.0

    for month in range(36):
        prod_date = date(2020, 1, 1) + timedelta(days=30 * month)

        # Butler-style ramp-up and decline
        if month <= tau:
            rate = q_peak * (month / tau) * np.exp(1 - month / tau)
        else:
            rate = q_peak * np.exp(-m * (month - tau) / tau)

        rate *= np.random.uniform(0.9, 1.1)
        hours = np.random.uniform(680, 744)

        cum_oil += rate

        records.append(
            {
                "uwi": uwi,
                "production_date": prod_date,
                "year": prod_date.year,
                "month_num": prod_date.month,
                "oil": max(0, rate),
                "gas": max(0, rate * 0.1),  # Low GOR for SAGD
                "water": max(0, rate * 2.0),  # High water cut
                "hours": hours,
                "boe": rate,
                "cum_oil": cum_oil,
            }
        )

    return pd.DataFrame(records)


@pytest.fixture
def sample_steam_df() -> pd.DataFrame:
    """Create sample steam injection data for thermal testing."""
    np.random.seed(42)

    records = []
    uwi = "PAD-01-SAGD"

    for month in range(36):
        prod_date = date(2020, 1, 1) + timedelta(days=30 * month)

        records.append(
            {
                "uwi": uwi,
                "date": prod_date,
                "steam_tonnes": np.random.uniform(8000, 12000),
                "pressure_kpa": np.random.uniform(2000, 2500),
                "temperature_c": np.random.uniform(200, 220),
            }
        )

    return pd.DataFrame(records)
