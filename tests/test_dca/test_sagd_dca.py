"""Tests for Butler-based SAGD DCA engine."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from reservoir_engine.dca.sagd_dca import (
    SAGDConfig,
    SAGDParams2Stage,
    SAGDParamsMultiSeg,
    SAGDTimeSeries,
    calculate_sagd_sor,
    fit_sagd_2stage,
    fit_sagd_butler,
    fit_sagd_multiseg,
    forecast_sagd,
    prepare_sagd_series,
    sagd_rate,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synthetic_sagd_data() -> pd.DataFrame:
    """Generate synthetic SAGD production data with known Butler parameters."""
    # Two-stage Butler: rising then declining
    # Rising: q = A_r * t^0.5
    # Spreading: q = A_s * t^(-0.5)
    # Transition at t_tr = 12

    a_r = 50.0
    alpha_r = 0.5
    t_tr = 12.0
    alpha_s = 0.5
    a_s = a_r * (t_tr ** (alpha_r + alpha_s))

    months = 60
    dates = pd.date_range("2020-01-01", periods=months, freq="MS")
    t = np.arange(1, months + 1)  # Start at 1 to avoid t=0

    # Compute Butler rates
    qo = np.where(
        t <= t_tr,
        a_r * np.power(t, alpha_r),
        a_s * np.power(t, -alpha_s),
    )

    # Add some noise
    np.random.seed(42)
    qo = qo * (1 + 0.08 * np.random.randn(months))
    qo = np.maximum(qo, 1)

    # Steam: SOR around 3-4
    sor = 3.5 + 0.5 * np.random.randn(months)
    sor = np.maximum(sor, 2.0)
    steam = qo * sor

    # Convert rates to volumes
    days_in_month = dates.to_series().dt.days_in_month.values
    from reservoir_engine.dca.gas_dca import NORMALIZED_MONTH_DAYS

    df = pd.DataFrame({
        "production_date": dates,
        "oil": qo * days_in_month / NORMALIZED_MONTH_DAYS,
        "steam_volume": steam * days_in_month / NORMALIZED_MONTH_DAYS,
    })

    return df


@pytest.fixture
def sagd_params_2stage() -> SAGDParams2Stage:
    """Sample two-stage SAGD parameters."""
    return SAGDParams2Stage(
        a_r=50.0,
        alpha_r=0.5,
        alpha_s=0.5,
        t_tr=12.0,
        terminal_decline_month=0.005,
        t_terminal=48.0,
    )


@pytest.fixture
def sagd_params_multiseg() -> SAGDParamsMultiSeg:
    """Sample multi-segment SAGD parameters."""
    return SAGDParamsMultiSeg(
        a=[50.0, 200.0],
        beta=[0.5, -0.4],
        segment_boundaries=[0, 12],
        terminal_decline_month=0.005,
    )


# =============================================================================
# Time Series Tests
# =============================================================================


class TestPrepareSAGDSeries:
    """Tests for SAGD time series preparation."""

    def test_creates_normalized_rates(self, synthetic_sagd_data):
        """Should create normalized rates from calendar data."""
        ts = prepare_sagd_series(synthetic_sagd_data)

        assert isinstance(ts, SAGDTimeSeries)
        assert len(ts.t_norm) > 0
        assert len(ts.qo) == len(ts.t_norm)
        assert len(ts.steam) == len(ts.t_norm)

    def test_drops_heatup_months(self, synthetic_sagd_data):
        """Should drop early heat-up months."""
        config_with_drop = SAGDConfig(drop_heatup_months=3)
        config_no_drop = SAGDConfig(drop_heatup_months=0)

        ts_with_drop = prepare_sagd_series(synthetic_sagd_data, config=config_with_drop)
        ts_no_drop = prepare_sagd_series(synthetic_sagd_data, config=config_no_drop)

        assert len(ts_with_drop.t_norm) == len(ts_no_drop.t_norm) - 3

    def test_computes_cumulative(self, synthetic_sagd_data):
        """Should compute cumulative oil and steam."""
        ts = prepare_sagd_series(synthetic_sagd_data)

        assert len(ts.cum_oil) == len(ts.t_norm)
        assert len(ts.cum_steam) == len(ts.t_norm)
        assert ts.cum_oil[-1] > ts.cum_oil[0]
        assert np.all(np.diff(ts.cum_oil) >= 0)


# =============================================================================
# Two-Stage Butler Model Tests
# =============================================================================


class TestButler2StageRate:
    """Tests for two-stage Butler rate calculations."""

    def test_rising_phase(self, sagd_params_2stage):
        """Rate should increase in rising phase."""
        t = np.array([1, 3, 6, 9, 12])  # Before transition
        qo = sagd_rate(t, sagd_params_2stage)

        for i in range(len(qo) - 1):
            assert qo[i] < qo[i + 1], f"Rate should rise: {qo[i]} < {qo[i+1]}"

    def test_spreading_phase(self, sagd_params_2stage):
        """Rate should decline in spreading phase."""
        t = np.array([15, 20, 30, 40])  # After transition
        qo = sagd_rate(t, sagd_params_2stage)

        for i in range(len(qo) - 1):
            assert qo[i] > qo[i + 1], f"Rate should decline: {qo[i]} > {qo[i+1]}"

    def test_peak_at_transition(self, sagd_params_2stage):
        """Peak rate should occur near transition time."""
        t = np.arange(1, 50)
        qo = sagd_rate(t, sagd_params_2stage)

        peak_idx = np.argmax(qo)
        peak_time = t[peak_idx]

        # Peak should be within 2 months of t_tr
        assert abs(peak_time - sagd_params_2stage.t_tr) <= 2

    def test_continuity_at_transition(self, sagd_params_2stage):
        """Rate should be continuous at transition."""
        t_before = sagd_params_2stage.t_tr - 0.01
        t_after = sagd_params_2stage.t_tr + 0.01

        q_before = sagd_rate(np.array([t_before]), sagd_params_2stage)[0]
        q_after = sagd_rate(np.array([t_after]), sagd_params_2stage)[0]

        # Should be within 1% of each other
        assert abs(q_before - q_after) / q_before < 0.01

    def test_terminal_decline_applied(self):
        """Should apply terminal exponential decline."""
        params = SAGDParams2Stage(
            a_r=50.0,
            alpha_r=0.5,
            alpha_s=0.5,
            t_tr=12.0,
            terminal_decline_month=0.02,  # 2% monthly
            t_terminal=36.0,
        )

        t = np.array([35, 36, 48, 60])
        qo = sagd_rate(t, params)

        # Rate should decline faster after terminal kicks in
        decline_before = (qo[0] - qo[1]) / qo[0]
        decline_after = (qo[2] - qo[3]) / qo[2]

        # Terminal decline should be faster
        assert decline_after > decline_before * 0.8


class TestFitSAGD2Stage:
    """Tests for two-stage Butler fitting."""

    def test_fit_produces_valid_parameters(self, synthetic_sagd_data):
        """Fitted model should produce valid parameters."""
        config = SAGDConfig(drop_heatup_months=0, use_terminal_decline=False)
        ts = prepare_sagd_series(synthetic_sagd_data, config=config)
        params = fit_sagd_2stage(ts, config)

        # Parameters should be positive
        assert params.a_r > 0, "a_r should be positive"
        assert params.alpha_r > 0, "alpha_r should be positive"
        assert params.alpha_s > 0, "alpha_s should be positive"
        assert params.t_tr > 0, "t_tr should be positive"

        # Exponents should be within bounds
        assert 0.3 <= params.alpha_r <= 0.7, f"alpha_r={params.alpha_r} out of bounds"
        assert 0.2 <= params.alpha_s <= 0.8, f"alpha_s={params.alpha_s} out of bounds"

        # Should be able to generate a forecast
        t_forecast = np.arange(1, 100)
        qo_forecast = sagd_rate(t_forecast, params)
        assert np.all(qo_forecast > 0), "Forecast should be all positive"

    def test_fit_requires_minimum_data(self):
        """Should raise error with insufficient data."""
        df = pd.DataFrame({
            "production_date": pd.date_range("2020-01-01", periods=5, freq="MS"),
            "oil": [100, 120, 130, 125, 115],
        })

        config = SAGDConfig(drop_heatup_months=0, min_months_for_fit=20)
        ts = prepare_sagd_series(df, config=config)

        with pytest.raises(ValueError, match="Insufficient data"):
            fit_sagd_2stage(ts, config)


# =============================================================================
# Multi-Segment Butler Model Tests
# =============================================================================


class TestButlerMultiSegRate:
    """Tests for multi-segment Butler rate calculations."""

    def test_computes_rate(self, sagd_params_multiseg):
        """Should compute rate for multi-segment model."""
        t = np.arange(1, 50)
        qo = sagd_rate(t, sagd_params_multiseg)

        assert len(qo) == len(t)
        assert np.all(qo > 0)

    def test_segment_transition(self, sagd_params_multiseg):
        """Rate behavior should change at segment boundary."""
        # First segment: rising (beta=0.5)
        t_rising = np.array([2, 5, 10])
        qo_rising = sagd_rate(t_rising, sagd_params_multiseg)

        # Rate should increase in rising segment
        assert qo_rising[0] < qo_rising[1] < qo_rising[2]

        # Second segment: declining (beta=-0.4)
        t_decline = np.array([15, 25, 40])
        qo_decline = sagd_rate(t_decline, sagd_params_multiseg)

        # Rate should decrease in declining segment
        assert qo_decline[0] > qo_decline[1] > qo_decline[2]


class TestFitSAGDMultiSeg:
    """Tests for multi-segment Butler fitting."""

    def test_fit_multiseg(self, synthetic_sagd_data):
        """Should fit multi-segment model."""
        config = SAGDConfig(
            model="butler_multiseg",
            num_segments=2,
            drop_heatup_months=0,
            use_terminal_decline=False,
        )
        ts = prepare_sagd_series(synthetic_sagd_data, config=config)
        params = fit_sagd_multiseg(ts, config)

        assert isinstance(params, SAGDParamsMultiSeg)
        assert len(params.a) == 2
        assert len(params.beta) == 2

    def test_first_segment_rising(self, synthetic_sagd_data):
        """First segment should have positive exponent (rising)."""
        config = SAGDConfig(
            model="butler_multiseg",
            num_segments=2,
            drop_heatup_months=0,
        )
        ts = prepare_sagd_series(synthetic_sagd_data, config=config)
        params = fit_sagd_multiseg(ts, config)

        # First segment should have positive beta (rising)
        assert params.beta[0] > 0, f"First segment should be rising, got beta={params.beta[0]}"


# =============================================================================
# Unified Interface Tests
# =============================================================================


class TestFitSAGDButler:
    """Tests for unified fitting interface."""

    def test_selects_2stage_model(self, synthetic_sagd_data):
        """Should use 2-stage model when configured."""
        config = SAGDConfig(model="butler_2stage", drop_heatup_months=0)
        ts = prepare_sagd_series(synthetic_sagd_data, config=config)
        params = fit_sagd_butler(ts, config)

        assert isinstance(params, SAGDParams2Stage)

    def test_selects_multiseg_model(self, synthetic_sagd_data):
        """Should use multi-segment model when configured."""
        config = SAGDConfig(model="butler_multiseg", num_segments=2, drop_heatup_months=0)
        ts = prepare_sagd_series(synthetic_sagd_data, config=config)
        params = fit_sagd_butler(ts, config)

        assert isinstance(params, SAGDParamsMultiSeg)


# =============================================================================
# Forecast Tests
# =============================================================================


class TestForecastSAGD:
    """Tests for SAGD forecasting."""

    def test_forecast_returns_dataframe(self, sagd_params_2stage):
        """Should return complete forecast DataFrame."""
        t = np.arange(0, 120)
        df = forecast_sagd(t, sagd_params_2stage)

        assert isinstance(df, pd.DataFrame)
        assert "t_norm" in df.columns
        assert "oil_rate" in df.columns
        assert "cum_oil" in df.columns

    def test_forecast_with_calendar(self, sagd_params_2stage):
        """Should include calendar dates when start date provided."""
        t = np.arange(0, 12)
        df = forecast_sagd(t, sagd_params_2stage, calendar_start=date(2020, 1, 1))

        assert "calendar_date" in df.columns
        assert df["calendar_date"].iloc[0] == date(2020, 1, 1)

    def test_cumulative_increases(self, sagd_params_2stage):
        """Cumulative production should be monotonically increasing."""
        t = np.arange(1, 60)  # Start at 1 to avoid t=0
        df = forecast_sagd(t, sagd_params_2stage)

        assert np.all(np.diff(df["cum_oil"]) >= 0)


# =============================================================================
# SOR Calculation Tests
# =============================================================================


class TestCalculateSAGDSOR:
    """Tests for Steam-Oil Ratio calculations."""

    def test_calculates_sor_metrics(self, synthetic_sagd_data):
        """Should calculate SOR metrics."""
        config = SAGDConfig(drop_heatup_months=0)
        ts = prepare_sagd_series(synthetic_sagd_data, config=config)
        params = fit_sagd_2stage(ts, config)

        sor_metrics = calculate_sagd_sor(ts, params)

        assert "isor_avg" in sor_metrics
        assert "csor" in sor_metrics
        assert sor_metrics["isor_avg"] > 0
        assert sor_metrics["csor"] > 0

    def test_sor_reasonable_range(self, synthetic_sagd_data):
        """SOR should be in reasonable range (2-6 for SAGD)."""
        config = SAGDConfig(drop_heatup_months=0)
        ts = prepare_sagd_series(synthetic_sagd_data, config=config)
        params = fit_sagd_2stage(ts, config)

        sor_metrics = calculate_sagd_sor(ts, params)

        # Synthetic data has SOR ~3.5
        assert 2.0 < sor_metrics["isor_avg"] < 6.0
        assert 2.0 < sor_metrics["csor"] < 6.0

