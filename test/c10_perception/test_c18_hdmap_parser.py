"""Unit tests for HDMap OpenDRIVE parsing (avlite.c10_perception.c18_hdmap_parser).

Tests verify:
- is_loadable recognizes .xodr fixture files.
- Parsing builds roads, lanes, and a lane network.
- reference_point is extracted from geoReference metadata.
- sample_OpenDrive_geometry produces expected line geometry.
"""

import numpy as np
import pytest

from avlite.c10_perception.c11_perception_model import HDMap
from avlite.c10_perception.c18_hdmap_parser import sample_OpenDrive_geometry


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


class TestOpenDriveGeometry:
    def test_line_geometry_endpoints(self):
        x_vals, y_vals = sample_OpenDrive_geometry(0.0, 0.0, 0.0, 10.0, "line", n_pts=11)
        assert len(x_vals) == 11
        assert x_vals[0] == pytest.approx(0.0)
        assert x_vals[-1] == pytest.approx(10.0, rel=0.01)
        assert np.allclose(y_vals, 0.0)
