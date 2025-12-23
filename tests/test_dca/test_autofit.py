"""Tests for decline curve autofit functions."""

import numpy as np
import pytest

from reservoir_engine.dca.autofit import fit_arps, fit_butler
from reservoir_engine.dca.models import arps_rate


class TestFitArps:
    """Tests for fit_arps function."""

    def test_fit_recovers_parameters(self, sample_production_monthly_df):
        """Test that fitting recovers reasonable parameters."""
        uwi = sample_production_monthly_df["uwi"].iloc[0]
        params = fit_arps(sample_production_monthly_df, uwi=uwi, method="auto")

        # Check parameters are in reasonable range
        assert params.qi > 0
        assert 0 < params.di < 1
        assert 0 <= params.b <= 2

    def test_fit_different_methods(self, sample_production_monthly_df):
        """Test different fitting methods work."""
        uwi = sample_production_monthly_df["uwi"].iloc[0]

        for method in ["hyperbolic", "harmonic", "exponential", "auto"]:
            params = fit_arps(sample_production_monthly_df, uwi=uwi, method=method)
            assert params.qi > 0

    def test_insufficient_data_raises(self, sample_production_monthly_df):
        """Test that insufficient data raises ValueError."""
        from datetime import date

        short_df = sample_production_monthly_df[
            sample_production_monthly_df["production_date"] < date(2020, 3, 1)
        ]
        uwi = short_df["uwi"].iloc[0]

        with pytest.raises(ValueError, match="Insufficient"):
            fit_arps(short_df, uwi=uwi, min_months=12)

    def test_fit_quality(self, sample_production_monthly_df):
        """Test that fitted curve matches data reasonably well."""
        uwi = sample_production_monthly_df["uwi"].iloc[0]
        params = fit_arps(sample_production_monthly_df, uwi=uwi, method="hyperbolic")

        # Get actual data
        well_df = sample_production_monthly_df[
            sample_production_monthly_df["uwi"] == uwi
        ]
        well_df = well_df.sort_values("production_date")
        t = np.arange(len(well_df))
        actual = well_df["oil"].values

        # Predict
        predicted = arps_rate(t, params)

        # Check correlation is reasonable (synthetic data may have noise)
        correlation = np.corrcoef(actual, predicted)[0, 1]
        assert correlation > 0.7


class TestFitButler:
    """Tests for fit_butler function."""

    def test_fit_thermal_data(self, sample_thermal_production_df, sample_steam_df):
        """Test fitting Butler model to thermal data."""
        uwi = sample_thermal_production_df["uwi"].iloc[0]
        params = fit_butler(
            sample_thermal_production_df,
            sample_steam_df,
            uwi=uwi,
        )

        # Check parameters are reasonable
        assert params.q_peak > 0
        assert params.m > 0
        assert params.tau > 0

    def test_fit_without_steam(self, sample_thermal_production_df):
        """Test fitting works without steam data."""
        uwi = sample_thermal_production_df["uwi"].iloc[0]
        params = fit_butler(
            sample_thermal_production_df,
            steam_df=None,
            uwi=uwi,
        )

        assert params.q_peak > 0
        assert params.sor == 3.0  # Default SOR
