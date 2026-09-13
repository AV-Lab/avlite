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
from avlite.c10_perception.c18_hdmap_parser import parse_geo_reference_from_root, sample_OpenDrive_geometry


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

    def test_short_centerline_access_check_does_not_index_error(self):
        """CARLA junction stubs often have only 2 sampled points."""
        hdmap = HDMap()
        lane_a = HDMap.Lane(
            id=-1,
            uid="a_-1",
            lane_element=ET.Element("lane"),
            center_line=np.array([[0.0, 1.0], [0.0, 0.0]]),
        )
        lane_b = HDMap.Lane(
            id=-1,
            uid="b_-1",
            lane_element=ET.Element("lane"),
            center_line=np.array([[1.0, 2.0], [0.0, 0.0]]),
        )
        lane_a.neighbors.add(lane_b)
        hdmap.can_laneA_access_laneB(lane_a, lane_b)
        empty = HDMap.Lane(id=-1, uid="empty", lane_element=ET.Element("lane"))
        assert hdmap.can_laneA_access_laneB(lane_a, empty) is False

    def test_unresolved_lane_link_does_not_crash(self):
        """Town03-style sidewalk/missing predecessor must not None-deref neighbors."""
        from pathlib import Path

        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "opendrive_unresolved_lane_link.xodr"
        hdmap = HDMap.from_path(fixture)
        driving = [lane for lane in hdmap.lanes if lane.type == "driving"]
        assert len(driving) == 1
        assert None not in driving[0].neighbors

    def test_bundled_town03_loads(self):
        from pathlib import Path

        town03 = Path(__file__).resolve().parents[2] / "avlite" / "data" / "Town03_Opt.xodr"
        hdmap = HDMap.from_path(town03)
        assert len(hdmap.roads) > 0
        assert all(lane is not None for lane in hdmap.lanes)
        for lane in hdmap.lanes:
            assert None not in lane.neighbors

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
