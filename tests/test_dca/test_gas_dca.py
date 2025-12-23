"""Tests for gas-first DCA engine."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from reservoir_engine.dca.gas_dca import (
    NORMALIZED_MONTH_DAYS,
    CGRConfig,
    CGRParams,
    DCAConfig,
    GasParams,
    TimeSeries,
    calendar_to_t_norm,
    fit_cgr,
    fit_gas,
    forecast_all,
    forecast_condensate,
    forecast_gas,
    prepare_time_series,
    t_norm_to_calendar,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synthetic_gas_data() -> pd.DataFrame:
    """Generate synthetic gas production data with known decline parameters."""
    # Known parameters: qi=1000, di=0.08, b=0.5
    qi = 1000.0
    di = 0.08
    b = 0.5

    months = 36
    dates = pd.date_range("2020-01-01", periods=months, freq="MS")

    # Generate decline curve
    t = np.arange(months)
    qg = qi / np.power(1 + b * di * t, 1 / b)

    # Add some noise
    np.random.seed(42)
    qg = qg * (1 + 0.05 * np.random.randn(months))
    qg = np.maximum(qg, 1)

    # Condensate as ~10% of gas with slight decline in CGR
    cgr = 0.10 * (1 - 0.002 * t)
    qc = qg * cgr

    # Convert rates to volumes (multiply by days in month)
    days_in_month = dates.to_series().dt.days_in_month.values

    df = pd.DataFrame({
        "production_date": dates,
        "gas": qg * days_in_month / NORMALIZED_MONTH_DAYS,
        "condensate": qc * days_in_month / NORMALIZED_MONTH_DAYS,
    })

    return df


@pytest.fixture
def gas_params() -> GasParams:
    """Sample gas parameters for testing."""
    return GasParams(
        qi=[1000.0],
        di=[0.08],
        b=[0.5],
        terminal_decline_month=0.005,
        segment_boundaries=[0],
    )


@pytest.fixture
def cgr_params() -> CGRParams:
    """Sample CGR parameters for testing."""
    return CGRParams(
        model_type="log_linear",
        alpha=-2.3,  # ~10% base CGR
        beta=0.0,
    )


# =============================================================================
# Time Handling Tests
# =============================================================================


class TestPrepareTimeSeries:
    """Tests for time series preparation."""

    def test_creates_normalized_rates(self, synthetic_gas_data):
        """Should create normalized rates from calendar data."""
        ts = prepare_time_series(synthetic_gas_data)

        assert isinstance(ts, TimeSeries)
        assert len(ts.t_norm) > 0
        assert len(ts.qg) == len(ts.t_norm)
        assert len(ts.qc) == len(ts.t_norm)

    def test_t_norm_is_integer_sequence(self, synthetic_gas_data):
        """t_norm should be 0, 1, 2, ..."""
        config = DCAConfig(drop_first_month=False)
        ts = prepare_time_series(synthetic_gas_data, config=config)

        expected = np.arange(len(ts.t_norm))
        np.testing.assert_array_equal(ts.t_norm, expected)

    def test_drops_first_month_by_default(self, synthetic_gas_data):
        """Should drop first month by default."""
        ts_with_drop = prepare_time_series(synthetic_gas_data)
        ts_no_drop = prepare_time_series(
            synthetic_gas_data, config=DCAConfig(drop_first_month=False)
        )

        assert len(ts_with_drop.t_norm) == len(ts_no_drop.t_norm) - 1

    def test_preserves_calendar_dates(self, synthetic_gas_data):
        """Should preserve original calendar dates."""
        ts = prepare_time_series(synthetic_gas_data)

        assert len(ts.calendar_dates) == len(ts.t_norm)
        assert isinstance(ts.calendar_start, date)


class TestTimeMapping:
    """Tests for calendar ↔ normalized time mapping."""

    def test_t_norm_to_calendar(self):
        """Should map normalized months to calendar dates."""
        start = date(2020, 1, 1)
        t_norm = np.array([0, 1, 2, 12, 24])

        dates = t_norm_to_calendar(t_norm, start)

        assert dates[0] == date(2020, 1, 1)
        assert dates[1] == date(2020, 2, 1)
        assert dates[2] == date(2020, 3, 1)
        assert dates[3] == date(2021, 1, 1)
        assert dates[4] == date(2022, 1, 1)

    def test_calendar_to_t_norm(self):
        """Should map calendar dates to normalized months."""
        start = date(2020, 1, 1)
        dates = [
            date(2020, 1, 1),
            date(2020, 6, 1),
            date(2021, 1, 1),
        ]

        t_norm = calendar_to_t_norm(dates, start)

        np.testing.assert_array_equal(t_norm, [0, 5, 12])

    def test_roundtrip_mapping(self):
        """Calendar -> t_norm -> calendar should roundtrip."""
        start = date(2020, 1, 1)
        original_dates = [date(2020, 1, 1), date(2020, 6, 1), date(2021, 3, 1)]

        t_norm = calendar_to_t_norm(original_dates, start)
        recovered_dates = t_norm_to_calendar(t_norm, start)

        assert recovered_dates == original_dates


# =============================================================================
# Gas Model Tests
# =============================================================================


class TestGasRate:
    """Tests for gas rate calculations."""

    def test_initial_rate_equals_qi(self, gas_params):
        """Gas rate at t=0 should equal qi."""
        qg = forecast_gas(np.array([0]), gas_params)
        assert abs(qg[0] - gas_params.qi[0]) < 1e-6

    def test_rate_declines(self, gas_params):
        """Gas rate should decline over time."""
        t = np.array([0, 6, 12, 24])
        qg = forecast_gas(t, gas_params)

        for i in range(len(qg) - 1):
            assert qg[i] > qg[i + 1], f"Rate should decline: {qg[i]} > {qg[i+1]}"

    def test_terminal_decline_applied(self):
        """Should switch to terminal exponential decline."""
        # High initial decline, low terminal
        params = GasParams(
            qi=[1000.0],
            di=[0.30],  # 30% monthly - very high
            b=[0.5],
            terminal_decline_month=0.01,  # 1% terminal
            segment_boundaries=[0],
        )

        t = np.arange(0, 120)
        qg = forecast_gas(t, params)

        # Rate should still be declining at end, but slowly
        late_decline = (qg[-12] - qg[-1]) / qg[-12]
        assert late_decline < 0.20  # Less than 20% over 12 months (terminal)


class TestFitGas:
    """Tests for gas model fitting."""

    def test_fit_recovers_approximate_parameters(self, synthetic_gas_data):
        """Should recover approximate decline parameters from synthetic data."""
        ts = prepare_time_series(synthetic_gas_data)
        params = fit_gas(ts)

        # Original: qi=1000, di=0.08, b=0.5
        assert 800 < params.qi[0] < 1200, f"qi should be ~1000, got {params.qi[0]}"
        assert 0.03 < params.di[0] < 0.15, f"di should be ~0.08, got {params.di[0]}"
        assert 0.2 < params.b[0] < 0.8, f"b should be ~0.5, got {params.b[0]}"

    def test_fit_requires_minimum_data(self):
        """Should raise error with insufficient data."""
        df = pd.DataFrame({
            "production_date": pd.date_range("2020-01-01", periods=3, freq="MS"),
            "gas": [100, 90, 80],
        })

        ts = prepare_time_series(df, config=DCAConfig(drop_first_month=False))

        with pytest.raises(ValueError, match="Insufficient data"):
            fit_gas(ts, DCAConfig(min_months_for_fit=10))

    def test_fit_handles_zeros(self, synthetic_gas_data):
        """Should handle zero production months."""
        df = synthetic_gas_data.copy()
        df.loc[5, "gas"] = 0  # Add a zero month

        ts = prepare_time_series(df)
        params = fit_gas(ts)

        assert params.qi[0] > 0


# =============================================================================
# CGR Model Tests
# =============================================================================


class TestCGRParams:
    """Tests for CGR parameter class."""

    def test_log_linear_cgr(self):
        """Log-linear CGR should compute correctly."""
        cgr = CGRParams(model_type="log_linear", alpha=-2.3, beta=0.0)

        qg = np.array([1000, 500, 100])
        result = cgr.cgr(qg)

        # alpha = -2.3 => CGR = exp(-2.3) ≈ 0.10
        expected = np.exp(-2.3)
        np.testing.assert_allclose(result, expected, rtol=0.01)

    def test_cgr_with_rate_dependence(self):
        """CGR with beta != 0 should vary with rate."""
        cgr = CGRParams(model_type="log_linear", alpha=-2.0, beta=0.1)

        qg_high = np.array([1000])
        qg_low = np.array([100])

        cgr_high = cgr.cgr(qg_high)[0]
        cgr_low = cgr.cgr(qg_low)[0]

        # Positive beta means higher CGR at higher rates
        assert cgr_high > cgr_low


class TestFitCGR:
    """Tests for CGR model fitting."""

    def test_fit_cgr_from_data(self, synthetic_gas_data):
        """Should fit CGR from production data."""
        ts = prepare_time_series(synthetic_gas_data)
        gas_params = fit_gas(ts)
        cgr_params = fit_cgr(ts, gas_params)

        assert isinstance(cgr_params, CGRParams)
        assert cgr_params.alpha != 0  # Should have fitted

    def test_fit_cgr_gives_reasonable_values(self, synthetic_gas_data):
        """Fitted CGR should give reasonable condensate estimates."""
        ts = prepare_time_series(synthetic_gas_data)
        gas_params = fit_gas(ts)
        cgr_params = fit_cgr(ts, gas_params)

        # CGR should be around 10% (0.10) based on synthetic data
        qg_ref = np.array([500])
        cgr_value = cgr_params.cgr(qg_ref)[0]

        assert 0.05 < cgr_value < 0.20, f"CGR should be ~0.10, got {cgr_value}"


# =============================================================================
# Forecast Tests
# =============================================================================


class TestForecastGas:
    """Tests for gas forecasting."""

    def test_forecast_returns_array(self, gas_params):
        """Should return numpy array of rates."""
        t = np.arange(0, 60)
        qg = forecast_gas(t, gas_params)

        assert isinstance(qg, np.ndarray)
        assert len(qg) == 60

    def test_forecast_all_positive(self, gas_params):
        """All forecasted rates should be positive."""
        t = np.arange(0, 360)
        qg = forecast_gas(t, gas_params)

        assert np.all(qg > 0)


class TestForecastCondensate:
    """Tests for condensate forecasting."""

    def test_forecast_condensate_from_gas(self, gas_params, cgr_params):
        """Should compute condensate from gas + CGR."""
        t = np.arange(0, 60)
        qc = forecast_condensate(t, gas_params, cgr_params)

        assert isinstance(qc, np.ndarray)
        assert len(qc) == 60
        assert np.all(qc > 0)

    def test_condensate_proportional_to_gas(self, gas_params, cgr_params):
        """Condensate should be proportional to gas (with constant CGR)."""
        t = np.arange(0, 60)
        qg = forecast_gas(t, gas_params)
        qc = forecast_condensate(t, gas_params, cgr_params)

        ratio = qc / qg
        # With beta=0, ratio should be constant
        np.testing.assert_allclose(ratio, ratio[0], rtol=0.01)


class TestForecastAll:
    """Tests for complete forecast generation."""

    def test_forecast_all_returns_dataframe(self, gas_params, cgr_params):
        """Should return complete forecast DataFrame."""
        t = np.arange(0, 120)
        df = forecast_all(t, gas_params, cgr_params)

        assert isinstance(df, pd.DataFrame)
        assert "t_norm" in df.columns
        assert "gas_rate" in df.columns
        assert "condensate_rate" in df.columns
        assert "cgr" in df.columns
        assert "cum_gas" in df.columns
        assert "cum_condensate" in df.columns

    def test_forecast_all_with_calendar(self, gas_params, cgr_params):
        """Should include calendar dates when start date provided."""
        t = np.arange(0, 12)
        df = forecast_all(t, gas_params, cgr_params, calendar_start=date(2020, 1, 1))

        assert "calendar_date" in df.columns
        assert df["calendar_date"].iloc[0] == date(2020, 1, 1)

    def test_cumulative_increases(self, gas_params, cgr_params):
        """Cumulative production should be monotonically increasing."""
        t = np.arange(0, 60)
        df = forecast_all(t, gas_params, cgr_params)

        assert np.all(np.diff(df["cum_gas"]) >= 0)
        assert np.all(np.diff(df["cum_condensate"]) >= 0)


# =============================================================================
# Multi-Segment Tests
# =============================================================================


class TestMultiSegment:
    """Tests for multi-segment decline."""

    def test_two_segment_continuity(self):
        """Two-segment model should be continuous at boundary."""
        config = DCAConfig(
            num_segments=2,
            segment_boundaries=[0, 12],  # Switch at month 12
        )

        # Create synthetic data with different decline rates
        t1 = np.arange(0, 12)
        t2 = np.arange(12, 36)

        qi = 1000
        di1, b1 = 0.15, 0.6  # Steep initial
        di2, b2 = 0.05, 0.3  # Shallower later

        q1 = qi / np.power(1 + b1 * di1 * t1, 1 / b1)
        q_at_12 = q1[-1]
        q2 = q_at_12 / np.power(1 + b2 * di2 * (t2 - 12), 1 / b2)

        qg = np.concatenate([q1, q2])
        t = np.concatenate([t1, t2])

        dates = pd.date_range("2020-01-01", periods=len(t), freq="MS")
        df = pd.DataFrame({
            "production_date": dates,
            "gas": qg * 30.4,  # Convert rate to volume
        })

        ts = prepare_time_series(df, config=DCAConfig(drop_first_month=False))
        params = fit_gas(ts, config)

        # Check continuity at boundary
        t_check = np.array([11.99, 12.01])
        q_check = forecast_gas(t_check, params)

        # Should be approximately equal (within 5%)
        assert abs(q_check[0] - q_check[1]) / q_check[0] < 0.05

