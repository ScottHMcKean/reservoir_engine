"""Tests for decline curve models."""

import numpy as np
import pytest

from reservoir_engine.dca.models import (
    ArpsParams,
    ButlerParams,
    arps_cumulative,
    arps_rate,
    butler_rate,
)


class TestArpsParams:
    """Tests for ArpsParams dataclass."""

    def test_valid_params(self):
        params = ArpsParams(qi=1000, di=0.1, b=0.8)
        assert params.qi == 1000
        assert params.di == 0.1
        assert params.b == 0.8

    def test_invalid_qi_raises(self):
        with pytest.raises(ValueError, match="qi must be positive"):
            ArpsParams(qi=-100, di=0.1, b=0.8)

    def test_invalid_b_raises(self):
        with pytest.raises(ValueError, match="b must be between"):
            ArpsParams(qi=1000, di=0.1, b=3.0)

    def test_t_switch_calculation(self):
        params = ArpsParams(qi=1000, di=0.12, b=0.8, d_min=0.05)
        # t_switch should be finite for b > 0
        assert params.t_switch > 0
        assert params.t_switch < float("inf")


class TestArpsRate:
    """Tests for arps_rate function."""

    def test_initial_rate(self, arps_params):
        rate = arps_rate(0, arps_params)
        assert rate == pytest.approx(arps_params.qi, rel=0.01)

    def test_rate_declines(self, arps_params):
        rate_0 = arps_rate(0, arps_params)
        rate_12 = arps_rate(12, arps_params)
        rate_24 = arps_rate(24, arps_params)

        assert rate_12 < rate_0
        assert rate_24 < rate_12

    def test_exponential_decline(self):
        params = ArpsParams(qi=1000, di=0.1, b=0)
        rate = arps_rate(10, params)
        expected = 1000 * np.exp(-0.1 * 10)
        assert rate == pytest.approx(expected, rel=0.01)

    def test_array_input(self, arps_params):
        t = np.array([0, 6, 12, 24])
        rates = arps_rate(t, arps_params)
        assert len(rates) == 4
        assert rates[0] > rates[-1]


class TestArpsCumulative:
    """Tests for arps_cumulative function."""

    def test_cumulative_increases(self, arps_params):
        cum_12 = arps_cumulative(12, arps_params)
        cum_24 = arps_cumulative(24, arps_params)
        assert cum_24 > cum_12

    def test_cumulative_at_zero(self, arps_params):
        cum = arps_cumulative(0, arps_params)
        assert cum == pytest.approx(0, abs=1e-6)


class TestButlerRate:
    """Tests for butler_rate function."""

    def test_peak_production(self, butler_params):
        # Rate should peak around tau
        t = np.arange(0, 48)
        rates = butler_rate(t, butler_params)
        peak_month = np.argmax(rates)

        # Peak should be close to tau
        assert abs(peak_month - butler_params.tau) <= 2

    def test_rate_at_zero(self, butler_params):
        rate = butler_rate(0, butler_params)
        # Rate at t=0 should be close to 0 (still ramping up)
        assert rate < butler_params.q_peak * 0.5

    def test_decline_after_peak(self, butler_params):
        rate_peak = butler_rate(butler_params.tau, butler_params)
        rate_later = butler_rate(butler_params.tau + 12, butler_params)
        assert rate_later < rate_peak

