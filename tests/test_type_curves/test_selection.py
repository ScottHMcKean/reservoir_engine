"""Tests for type curve well selection functions (H3-based)."""

import pytest


class TestWellsInBbox:
    """Tests for wells_in_bbox function."""

    def test_all_wells_in_large_bbox(self, sample_well_header_df):
        from reservoir_engine.type_curves.selection import wells_in_bbox

        wells = wells_in_bbox(
            sample_well_header_df,
            min_lon=-117,
            min_lat=54,
            max_lon=-116,
            max_lat=55,
        )
        assert len(wells) == 3

    def test_no_wells_outside_bbox(self, sample_well_header_df):
        from reservoir_engine.type_curves.selection import wells_in_bbox

        wells = wells_in_bbox(
            sample_well_header_df,
            min_lon=-120,
            min_lat=60,
            max_lon=-119,
            max_lat=61,
        )
        assert len(wells) == 0


class TestH3Functions:
    """Tests for H3-based spatial functions."""

    def test_get_h3_index(self):
        from reservoir_engine.type_curves.selection import get_h3_index

        h3_idx = get_h3_index(54.5, -116.5, resolution=9)
        assert isinstance(h3_idx, str)
        assert len(h3_idx) > 0

    def test_add_h3_index(self, sample_well_header_df):
        from reservoir_engine.type_curves.selection import add_h3_index

        result = add_h3_index(sample_well_header_df)
        assert "h3_index" in result.columns
        assert result["h3_index"].notna().all()

    def test_wells_in_h3_ring(self, sample_well_header_df):
        from reservoir_engine.type_curves.selection import wells_in_h3_ring

        # Use center of data
        wells = wells_in_h3_ring(
            sample_well_header_df,
            center_lat=54.3,
            center_lon=-116.7,
            k_rings=5,
            resolution=7,
        )
        # Should find at least one well
        assert len(wells) >= 1

    def test_get_h3_resolution_for_radius(self):
        from reservoir_engine.type_curves.selection import get_h3_resolution_for_radius

        # 5km radius should give resolution ~7
        res = get_h3_resolution_for_radius(5.0)
        assert 6 <= res <= 8


class TestFilterWellsByAttributes:
    """Tests for filter_wells_by_attributes function."""

    def test_filter_by_hole_direction(self, sample_well_header_df):
        from reservoir_engine.type_curves.selection import filter_wells_by_attributes

        horizontal = filter_wells_by_attributes(
            sample_well_header_df, hole_direction="HORIZONTAL"
        )
        assert len(horizontal) == 1
        assert "102133506219W500" in horizontal

    def test_filter_by_play_type(self, sample_well_header_df):
        from reservoir_engine.type_curves.selection import filter_wells_by_attributes

        shale = filter_wells_by_attributes(sample_well_header_df, play_type="SHALE")
        assert len(shale) == 3  # All wells are SHALE
