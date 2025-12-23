"""Tests for type curve well selection functions."""

import pytest

from reservoir_engine.type_curves.selection import (
    wells_in_bbox,
    wells_near_point,
)


class TestWellsInBbox:
    """Tests for wells_in_bbox function."""

    def test_all_wells_in_large_bbox(self, sample_wells_df):
        wells = wells_in_bbox(
            sample_wells_df,
            min_lon=-111,
            min_lat=50,
            max_lon=-110,
            max_lat=51,
        )
        assert len(wells) == 3

    def test_no_wells_outside_bbox(self, sample_wells_df):
        wells = wells_in_bbox(
            sample_wells_df,
            min_lon=-120,
            min_lat=60,
            max_lon=-119,
            max_lat=61,
        )
        assert len(wells) == 0


class TestWellsNearPoint:
    """Tests for wells_near_point function."""

    def test_finds_nearby_wells(self, sample_wells_df):
        wells = wells_near_point(
            sample_wells_df,
            center_lon=-110.5,
            center_lat=50.5,
            radius_km=10,
        )
        assert len(wells) >= 1

    def test_no_wells_far_away(self, sample_wells_df):
        wells = wells_near_point(
            sample_wells_df,
            center_lon=-100,
            center_lat=40,
            radius_km=1,
        )
        assert len(wells) == 0

