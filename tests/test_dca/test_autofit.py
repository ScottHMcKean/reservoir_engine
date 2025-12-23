"""Tests for decline curve autofit functions."""

import numpy as np
import pytest

from reservoir_engine.dca.autofit import fit_arps, fit_butler
from reservoir_engine.dca.models import arps_rate


class TestFitArps:
    """Tests for fit_arps function."""

    def test_fit_recovers_parameters(self, sample_production_df):
        """Test that fitting recovers reasonable parameters."""
        params = fit_arps(sample_production_df, uwi="WELL-001", method="auto")

        # Check parameters are in reasonable range
        assert params.qi > 0
        assert 0 < params.di < 1
        assert 0 <= params.b <= 2

    def test_fit_different_methods(self, sample_production_df):
        """Test different fitting methods work."""
        for method in ["hyperbolic", "harmonic", "exponential", "auto"]:
            params = fit_arps(sample_production_df, uwi="WELL-001", method=method)
            assert params.qi > 0

    def test_insufficient_data_raises(self, sample_production_df):
        """Test that insufficient data raises ValueError."""
        from datetime import date

        short_df = sample_production_df[sample_production_df["date"] < date(2022, 3, 1)]
        with pytest.raises(ValueError, match="Insufficient"):
            fit_arps(short_df, uwi="WELL-001", min_months=12)

    def test_fit_quality(self, sample_production_df):
        """Test that fitted curve matches data reasonably well."""
        params = fit_arps(sample_production_df, uwi="WELL-001", method="hyperbolic")

        # Get actual data
        well_df = sample_production_df[sample_production_df["uwi"] == "WELL-001"]
        well_df = well_df.sort_values("date")
        t = np.arange(len(well_df))
        actual = well_df["oil_bbl"].values

        # Predict
        predicted = arps_rate(t, params)

        # Check correlation is high
        correlation = np.corrcoef(actual, predicted)[0, 1]
        assert correlation > 0.8


class TestFitButler:
    """Tests for fit_butler function."""

    def test_fit_thermal_data(self, sample_thermal_production_df, sample_steam_df):
        """Test fitting Butler model to thermal data."""
        params = fit_butler(
            sample_thermal_production_df,
            sample_steam_df,
            uwi="PAD-01",
        )

        # Check parameters are reasonable
        assert params.q_peak > 0
        assert params.m > 0
        assert params.tau > 0

    def test_fit_without_steam(self, sample_thermal_production_df):
        """Test fitting works without steam data."""
        params = fit_butler(
            sample_thermal_production_df,
            steam_df=None,
            uwi="PAD-01",
        )

        assert params.q_peak > 0
        assert params.sor == 3.0  # Default SOR

