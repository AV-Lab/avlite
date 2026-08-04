"""Unit tests for HDMap OpenDRIVE parsing (avlite.c10_perception.c18_hdmap_parser).

Tests verify:
- is_loadable recognizes .xodr fixture files.
- Parsing builds roads, lanes, and a lane network.
- reference_point is extracted from geoReference metadata.
- sample_OpenDrive_geometry produces expected line geometry.
"""

import numpy as np
import pytest
import xml.etree.ElementTree as ET

from avlite.c10_perception.c11_perception_model import HDMap
from avlite.c10_perception.c18_hdmap_parser import (
    _get_lane_offset_at_s,
    parse_geo_reference_from_root,
    sample_OpenDrive_geometry,
)


class TestHDMapLoadable:
    def test_is_loadable_for_fixture(self, minimal_opendrive_path):
        assert HDMap.is_loadable(minimal_opendrive_path) is True

    def test_is_loadable_rejects_non_xodr(self, tmp_path):
        path = tmp_path / "map.json"
        path.write_text("{}")
        assert HDMap.is_loadable(path) is False


class TestHDMapParse:
    def test_parses_road_and_driving_lanes(self, minimal_opendrive_path):
        hdmap = HDMap.from_path(minimal_opendrive_path)
        assert len(hdmap.roads) == 1
        driving = [lane for lane in hdmap.lanes if lane.type == "driving"]
        assert len(driving) >= 2
        assert all(len(lane.center_line) > 0 for lane in driving)

    def test_reference_point_from_geo_reference(self, minimal_opendrive_path):
        hdmap = HDMap.from_path(minimal_opendrive_path)
        ref = hdmap.reference_point
        assert ref is not None
        assert ref[0] == pytest.approx(45.0, abs=1e-6)
        assert ref[1] == pytest.approx(55.0, abs=1e-6)


class TestGeoReferenceParsing:
    def test_parse_geo_reference_converts_radians(self):
        root = ET.fromstring(
            "<OpenDRIVE><header>"
            "<geoReference>+proj=tmerc +lat_0=0.5 +lon_0=-0.001 +units=m +no_defs</geoReference>"
            "</header></OpenDRIVE>"
        )
        ref = parse_geo_reference_from_root(root)
        assert ref is not None
        assert ref[0] == pytest.approx(np.degrees(0.5), abs=1e-6)
        assert ref[1] == pytest.approx(np.degrees(-0.001), abs=1e-6)


class TestOpenDriveGeometry:
    def test_line_geometry_endpoints(self):
        x_vals, y_vals = sample_OpenDrive_geometry(0.0, 0.0, 0.0, 10.0, "line", n_pts=11)
        assert len(x_vals) == 11
        assert x_vals[0] == pytest.approx(0.0)
        assert x_vals[-1] == pytest.approx(10.0, rel=0.01)
        assert np.allclose(y_vals, 0.0)

    def test_arc_geometry_quarter_circle_endpoint(self):
        """Positive curvature arc of π/2 with radius 10 ends near (10, 10)."""
        curvature = 0.1
        radius = 1.0 / curvature
        length = (np.pi / 2.0) * radius
        x_vals, y_vals = sample_OpenDrive_geometry(
            0.0,
            0.0,
            0.0,
            length,
            "arc",
            attributes={"curvature": str(curvature)},
            n_pts=25,
        )
        assert x_vals[0] == pytest.approx(0.0, abs=1e-9)
        assert y_vals[0] == pytest.approx(0.0, abs=1e-9)
        assert x_vals[-1] == pytest.approx(radius, abs=1e-6)
        assert y_vals[-1] == pytest.approx(radius, abs=1e-6)

    def test_zero_curvature_spiral_falls_back_to_line(self):
        x_vals, y_vals = sample_OpenDrive_geometry(
            0.0,
            0.0,
            0.0,
            8.0,
            "spiral",
            attributes={"curvStart": "0", "curvEnd": "0"},
            n_pts=5,
        )
        assert x_vals[-1] == pytest.approx(8.0, abs=1e-9)
        assert np.allclose(y_vals, 0.0)


class TestLaneOffsetAtS:
    def test_empty_offsets_are_zero(self):
        assert _get_lane_offset_at_s([], 5.0) == 0.0

    def test_uses_latest_applicable_polynomial_segment(self):
        offsets = [
            {"s": "0.0", "a": "1.0", "b": "0.0", "c": "0.0", "d": "0.0"},
            {"s": "10.0", "a": "2.0", "b": "0.5", "c": "0.0", "d": "0.0"},
        ]
        # Before second segment: constant a=1.
        assert _get_lane_offset_at_s(offsets, 5.0) == pytest.approx(1.0)
        # Inside second segment: a + b*(s-10) = 2 + 0.5*4.
        assert _get_lane_offset_at_s(offsets, 14.0) == pytest.approx(4.0)

    def test_before_first_offset_s_is_zero(self):
        offsets = [{"s": "5.0", "a": "3.0", "b": "0.0", "c": "0.0", "d": "0.0"}]
        assert _get_lane_offset_at_s(offsets, 1.0) == 0.0
