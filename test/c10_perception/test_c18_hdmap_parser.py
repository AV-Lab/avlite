"""Unit tests for HDMap OpenDRIVE parsing (avlite.c10_perception.c18_hdmap_parser).

Tests verify:
- is_loadable recognizes .xodr fixture files.
- Parsing builds roads, lanes, and a lane network.
- reference_point is extracted from geoReference metadata.
- Lane sections, lane width and offset, sharp curves and north roads parse correctly.
- sample_OpenDrive_geometry produces expected line geometry.
- HDMapGlobalPlanner plans a route through lane sections.
"""

import logging

import numpy as np
import pytest
import xml.etree.ElementTree as ET

from avlite.c10_perception.c11_perception_model import HDMap
from avlite.c10_perception.c18_hdmap_parser import parse_geo_reference_from_root, sample_OpenDrive_geometry
from avlite.c20_planning.c24_global_hdmap_planners import HDMapGlobalPlanner


@pytest.fixture
def lane_map(lane_geometry_opendrive_path):
    return HDMap.from_path(lane_geometry_opendrive_path)


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


class TestHDMapLaneGeometry:
    # The map's header describes each road
    def test_each_section_keeps_its_lanes(self, lane_map):
        uids = sorted(uid for uid in lane_map.lane_by_uid if uid[0] in "12")
        assert uids == ["1_0_-1", "1_0_1", "1_1_-1", "1_1_-2", "1_1_1", "2_-1", "2_1"]

    def test_section_lanes_cover_only_their_section(self, lane_map):
        for uid, (x_start, x_end) in {"1_0_-1": (0.0, 50.0), "1_1_-1": (50.0, 100.0), "2_-1": (100.0, 150.0)}.items():
            x = lane_map.lane_by_uid[uid].center_line[0]
            assert x.min() == pytest.approx(x_start, abs=0.1)
            assert x.max() == pytest.approx(x_end, abs=0.1)

    def test_sections_link_in_driving_direction(self, lane_map):
        edges = {(a, b) for a, b, change in lane_map.lane_network.edges(data="lane_change") if not change and a[0] in "12"}
        assert edges == {("1_0_-1", "1_1_-1"), ("1_1_-1", "2_-1"), ("2_1", "1_1_1"), ("1_1_1", "1_0_1")}

    def test_single_section_road_keeps_plain_uid(self, minimal_opendrive_path):
        assert sorted(HDMap.from_path(minimal_opendrive_path).lane_by_uid) == ["1_-1", "1_1"]

    def test_lane_follows_its_width_along_the_road(self, lane_map):
        line = lane_map.lane_by_uid["3_-1"].center_line
        for x, y in [(0, -101.0), (10, -101.25), (30, -102.0)]:
            assert line[1][np.argmin(np.abs(line[0] - x))] == pytest.approx(y, abs=0.01)

    def test_lane_follows_lane_offset_along_the_road(self, lane_map):
        y = lane_map.lane_by_uid["4_-1"].center_line[1]
        assert (y[0], y[-1]) == pytest.approx((-201.0, -199.0), abs=0.01)

    def test_short_sharp_curve_keeps_its_turn(self, lane_map):
        assert lane_map.lane_network.has_edge("5_-1", "6_-1")  # 42

    def test_road_heading_exactly_north_loads(self, lane_map):
        line = lane_map.lane_by_uid["7_-1"].center_line  # 41
        assert np.allclose(line[0], 4.0)
        assert line[1][-1] == pytest.approx(-440.0, abs=0.01)

    def test_offset_jump_between_sections_keeps_the_link(self, lane_map):
        assert lane_map.lane_network.has_edge("8_0_-2", "8_1_-1")
        assert np.allclose(lane_map.lane_by_uid["8_0_-2"].center_line[1], -503.5)

    def test_tiny_sections_stay_valid(self, lane_map):
        lane = lane_map.lane_by_uid["9_0_-1"]
        assert lane.width == pytest.approx(3.5)
        assert np.allclose(lane.center_line[1], -601.75, atol=1.0)
        assert lane_map.lane_by_uid["9_2_-1"].center_line.shape[1] == 3

    def test_route_through_lane_sections(self, lane_map, caplog):
        planner = HDMapGlobalPlanner(lane_map)
        planner.set_start_goal((10.0, -1.75), (140.0, -1.75))
        with caplog.at_level(logging.WARNING):
            plan = planner.plan()
        assert [lane.uid for lane in plan.lane_path] == ["1_0_-1", "1_1_-1", "2_-1"]
        assert not caplog.records
        path = np.array(plan.path)
        assert np.allclose(path[[0, -1], 0], [10.0, 140.0], atol=0.5)
        assert np.all(np.diff(path[:, 0]) > 0)
        assert np.allclose(path[:, 1], -1.75, atol=0.1)


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
