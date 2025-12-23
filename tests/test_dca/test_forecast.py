"""Tests for forecast functions."""

import pytest

from reservoir_engine.dca.forecast import (
    arps_forecast,
    butler_forecast,
    calculate_eur,
    combine_forecasts,
)


class TestArpsForecast:
    """Tests for arps_forecast function."""

    def test_forecast_length(self, arps_params):
        forecast = arps_forecast(arps_params, months=120)
        assert len(forecast) == 121  # 0 to 120 inclusive

    def test_forecast_columns(self, arps_params):
        forecast = arps_forecast(arps_params, months=60)
        assert "date" in forecast.columns
        assert "month" in forecast.columns
        assert "rate" in forecast.columns
        assert "cumulative" in forecast.columns

    def test_cumulative_increases(self, arps_params):
        forecast = arps_forecast(arps_params, months=60)
        assert forecast["cumulative"].is_monotonic_increasing

    def test_economic_limit(self, arps_params):
        forecast = arps_forecast(arps_params, months=600, economic_limit=50)
        # Should stop before 600 months due to economic limit
        assert len(forecast) < 600


class TestButlerForecast:
    """Tests for butler_forecast function."""

    def test_forecast_length(self, butler_params):
        forecast = butler_forecast(butler_params, months=120)
        assert len(forecast) == 121

    def test_includes_steam(self, butler_params):
        forecast = butler_forecast(butler_params, months=60, include_sor=True)
        assert "steam_required" in forecast.columns


class TestCalculateEur:
    """Tests for calculate_eur function."""

    def test_eur_positive(self, arps_params):
        eur = calculate_eur(arps_params, economic_limit=10)
        assert eur > 0

    def test_higher_limit_lower_eur(self, arps_params):
        eur_low = calculate_eur(arps_params, economic_limit=10)
        eur_high = calculate_eur(arps_params, economic_limit=50)
        assert eur_high < eur_low

    def test_eur_butler(self, butler_params):
        eur = calculate_eur(butler_params, economic_limit=10)
        assert eur > 0


class TestCombineForecasts:
    """Tests for combine_forecasts function."""

    def test_combine_sum(self, arps_params):
        fc1 = arps_forecast(arps_params, months=60)
        fc2 = arps_forecast(arps_params, months=60)

        combined = combine_forecasts([("W1", fc1), ("W2", fc2)], agg_method="sum")

        # Combined rate should be ~2x individual
        assert combined["rate"].iloc[0] == pytest.approx(fc1["rate"].iloc[0] * 2, rel=0.01)

    def test_combine_empty(self):
        result = combine_forecasts([])
        assert len(result) == 0

