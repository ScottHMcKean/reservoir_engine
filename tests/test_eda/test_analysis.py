"""Tests for EDA analysis functions."""

import pandas as pd
import pytest

from reservoir_engine.eda.analysis import (
    calculate_decline_rate,
    calculate_statistics,
    detect_outliers,
    production_summary,
)


class TestCalculateStatistics:
    """Tests for calculate_statistics function."""

    def test_basic_stats(self, sample_production_monthly_df):
        stats = calculate_statistics(sample_production_monthly_df, value_column="oil")
        assert "mean" in stats.columns or "mean" in stats.index

    def test_grouped_stats(self, sample_production_monthly_df):
        stats = calculate_statistics(
            sample_production_monthly_df, value_column="oil", group_by="uwi"
        )
        assert len(stats) == 3  # 3 wells


class TestDetectOutliers:
    """Tests for detect_outliers function."""

    def test_iqr_method(self, sample_production_monthly_df):
        result = detect_outliers(sample_production_monthly_df, "oil", method="iqr")
        assert "is_outlier" in result.columns

    def test_zscore_method(self, sample_production_monthly_df):
        result = detect_outliers(sample_production_monthly_df, "oil", method="zscore")
        assert "is_outlier" in result.columns

    def test_detects_obvious_outlier(self):
        df = pd.DataFrame(
            {
                "uwi": ["W1"] * 10,
                "oil": [100, 100, 100, 100, 100, 100, 100, 100, 100, 10000],
            }
        )
        result = detect_outliers(df, "oil", method="iqr")
        assert result["is_outlier"].sum() >= 1


class TestProductionSummary:
    """Tests for production_summary function."""

    def test_summary_columns(self, sample_production_monthly_df):
        summary = production_summary(sample_production_monthly_df)
        assert "cum_oil" in summary.columns
        assert "peak_oil_rate" in summary.columns
        assert "months_on" in summary.columns

    def test_summary_per_well(self, sample_production_monthly_df):
        summary = production_summary(sample_production_monthly_df)
        assert len(summary) == 3  # 3 wells


class TestCalculateDeclineRate:
    """Tests for calculate_decline_rate function."""

    def test_returns_decline_column(self, sample_production_monthly_df):
        uwi = sample_production_monthly_df["uwi"].iloc[0]
        result = calculate_decline_rate(sample_production_monthly_df, uwi=uwi)
        assert "decline_rate" in result.columns
        assert "smoothed_rate" in result.columns
