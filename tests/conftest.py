"""Shared test fixtures for Reservoir Engine.

Provides reusable fixtures for production, completion, survey,
and well header data.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_production_df() -> pd.DataFrame:
    """Create sample production data for testing.

    Generates 24 months of synthetic production data for 3 wells
    with realistic decline behavior.
    """
    np.random.seed(42)

    wells = ["WELL-001", "WELL-002", "WELL-003"]
    months = 24
    records = []

    for uwi in wells:
        # Different initial rates and decline for each well
        qi = np.random.uniform(500, 1500)
        di = np.random.uniform(0.08, 0.15)
        b = np.random.uniform(0.5, 1.2)

        for month in range(months):
            # Arps hyperbolic decline with noise
            rate = qi / (1 + b * di * month) ** (1 / b)
            rate *= np.random.uniform(0.9, 1.1)  # Add noise

            prod_date = date(2022, 1, 1) + timedelta(days=30 * month)
            days_on = np.random.randint(25, 31)

            records.append(
                {
                    "uwi": uwi,
                    "date": prod_date,
                    "oil_bbl": max(0, rate * days_on),
                    "gas_mcf": max(0, rate * 1.5 * days_on),
                    "water_bbl": max(0, rate * 0.3 * days_on),
                    "days_on": days_on,
                }
            )

    return pd.DataFrame(records)


@pytest.fixture
def sample_wells_df() -> pd.DataFrame:
    """Create sample well header data for testing."""
    return pd.DataFrame(
        [
            {
                "uwi": "WELL-001",
                "latitude": 50.5,
                "longitude": -110.5,
                "formation": "Montney",
                "operator": "Operator A",
                "completion_date": "2021-12-15",
            },
            {
                "uwi": "WELL-002",
                "latitude": 50.52,
                "longitude": -110.48,
                "formation": "Montney",
                "operator": "Operator A",
                "completion_date": "2021-12-20",
            },
            {
                "uwi": "WELL-003",
                "latitude": 50.48,
                "longitude": -110.52,
                "formation": "Duvernay",
                "operator": "Operator B",
                "completion_date": "2022-01-05",
            },
        ]
    )


@pytest.fixture
def sample_completions_df() -> pd.DataFrame:
    """Create sample completion data for testing."""
    records = []

    # WELL-001: 30 stages
    for stage in range(1, 31):
        records.append(
            {
                "uwi": "WELL-001",
                "stage": stage,
                "perf_top_md": 2000 + stage * 100,
                "perf_btm_md": 2050 + stage * 100,
                "proppant_tonnes": np.random.uniform(80, 120),
                "fluid_m3": np.random.uniform(400, 600),
                "cluster_count": np.random.randint(3, 6),
            }
        )

    # WELL-002: 25 stages
    for stage in range(1, 26):
        records.append(
            {
                "uwi": "WELL-002",
                "stage": stage,
                "perf_top_md": 2100 + stage * 100,
                "perf_btm_md": 2150 + stage * 100,
                "proppant_tonnes": np.random.uniform(70, 110),
                "fluid_m3": np.random.uniform(350, 550),
                "cluster_count": np.random.randint(3, 5),
            }
        )

    # WELL-003: 35 stages
    for stage in range(1, 36):
        records.append(
            {
                "uwi": "WELL-003",
                "stage": stage,
                "perf_top_md": 1900 + stage * 90,
                "perf_btm_md": 1940 + stage * 90,
                "proppant_tonnes": np.random.uniform(90, 130),
                "fluid_m3": np.random.uniform(450, 650),
                "cluster_count": np.random.randint(4, 7),
            }
        )

    return pd.DataFrame(records)


@pytest.fixture
def sample_surveys_df() -> pd.DataFrame:
    """Create sample survey data for testing."""
    records = []

    for uwi in ["WELL-001", "WELL-002", "WELL-003"]:
        # Vertical section
        for md in range(0, 1000, 100):
            records.append(
                {
                    "uwi": uwi,
                    "md": md,
                    "tvd": md,
                    "inclination": 0,
                    "azimuth": 0,
                }
            )

        # Build section
        for i, md in enumerate(range(1000, 2000, 100)):
            inc = min(90, 5 + i * 9)  # Build to horizontal
            records.append(
                {
                    "uwi": uwi,
                    "md": md,
                    "tvd": 1000 + (md - 1000) * np.cos(np.radians(inc / 2)),
                    "inclination": inc,
                    "azimuth": 270,
                }
            )

        # Horizontal section
        for md in range(2000, 5000, 100):
            records.append(
                {
                    "uwi": uwi,
                    "md": md,
                    "tvd": 1500,  # Constant TVD in horizontal
                    "inclination": 90,
                    "azimuth": 270,
                }
            )

    return pd.DataFrame(records)


@pytest.fixture
def sample_steam_df() -> pd.DataFrame:
    """Create sample steam injection data for thermal testing."""
    np.random.seed(42)

    records = []
    for month in range(36):
        prod_date = date(2022, 1, 1) + timedelta(days=30 * month)

        records.append(
            {
                "uwi": "PAD-01",
                "date": prod_date,
                "steam_tonnes": np.random.uniform(8000, 12000),
                "pressure_kpa": np.random.uniform(2000, 2500),
                "temperature_c": np.random.uniform(200, 220),
            }
        )

    return pd.DataFrame(records)


@pytest.fixture
def sample_thermal_production_df() -> pd.DataFrame:
    """Create sample thermal (SAGD) production data."""
    np.random.seed(42)

    records = []

    # SAGD ramp-up and decline profile
    q_peak = 500
    tau = 12  # Months to peak
    m = 0.08  # Decline rate

    for month in range(36):
        prod_date = date(2022, 1, 1) + timedelta(days=30 * month)

        # Butler-style production profile
        if month <= tau:
            rate = q_peak * (month / tau) * np.exp(1 - month / tau)
        else:
            rate = q_peak * np.exp(-m * (month - tau) / tau)

        rate *= np.random.uniform(0.9, 1.1)  # Add noise
        days_on = np.random.randint(28, 31)

        records.append(
            {
                "uwi": "PAD-01",
                "date": prod_date,
                "oil_bbl": max(0, rate * days_on),
                "gas_mcf": max(0, rate * 0.1 * days_on),  # Low GOR for SAGD
                "water_bbl": max(0, rate * 2.0 * days_on),  # High water cut
                "days_on": days_on,
            }
        )

    return pd.DataFrame(records)


@pytest.fixture
def arps_params():
    """Create sample Arps parameters for testing."""
    from reservoir_engine.dca.models import ArpsParams

    return ArpsParams(qi=1000, di=0.12, b=0.8)


@pytest.fixture
def butler_params():
    """Create sample Butler parameters for testing."""
    from reservoir_engine.dca.models import ButlerParams

    return ButlerParams(q_peak=500, m=0.1, tau=12, sor=3.0)

