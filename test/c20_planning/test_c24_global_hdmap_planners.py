"""Tests for HDMapGlobalPlanner routes, on the loop of roads 30 to 33 in the shared test map."""
import numpy as np
import pytest

from avlite.c10_perception.c11_perception_model import EgoState, HDMap, PerceptionModel
from avlite.c20_planning.c24_global_hdmap_planners import HDMapGlobalPlanner

STRAIGHT_ON = ["31_0_-1", "31_1_-1", "31_2_-1"]


def _plan(map_path, start, goal, heading=None):
    planner = HDMapGlobalPlanner(HDMap.from_path(map_path))
    planner.set_start_goal(start, goal)
    if heading is None:
        return planner.plan()
    return planner.plan(PerceptionModel(ego_vehicle=EgoState(x=start[0], y=start[1], theta=heading)))


class TestHDMapGlobalPlannerRoutes:
    def test_route_is_shortest_in_metres_not_in_lanes(self, lane_geometry_opendrive_path):
        # Straight on: 3 lanes of one road, 10 m in all. Detour: 1 lane, about 16 m
        plan = _plan(lane_geometry_opendrive_path, (2.0, -701.0), (10.0, -689.0))
        assert [lane.uid for lane in plan.lane_path] == ["30_-1", *STRAIGHT_ON, "33_1"]

    def test_route_does_not_hop_lanes_to_save_a_few_metres(self, lane_geometry_opendrive_path):
        # Hopping to the inner lane of road 33 and back would save about 13 m
        plan = _plan(lane_geometry_opendrive_path, (10.0, -687.0), (5.0, -703.0))
        assert [lane.uid for lane in plan.lane_path] == ["33_2", "30_-2"]

    @pytest.mark.parametrize("start, goal, route", [
        ((7.0, -701.0), (2.0, -701.0), ["30_-1", *STRAIGHT_ON, "33_1", "30_-1"]),  # Right lane, driven along its road
        ((5.0, -689.0), (15.0, -689.0), ["33_1", "30_-1", *STRAIGHT_ON, "33_1"]),  # Left lane, driven against its road
    ])
    def test_goal_behind_on_the_same_lane_goes_round(self, lane_geometry_opendrive_path, start, goal, route):
        plan = _plan(lane_geometry_opendrive_path, start, goal)
        assert [lane.uid for lane in plan.lane_path] == route
        assert np.allclose(np.array(plan.path)[[0, -1]], [start, goal], atol=1.0)

    def test_goal_behind_skips_a_next_lane_with_no_way_back(self, lane_geometry_opendrive_path):
        planner = HDMapGlobalPlanner(HDMap.from_path(lane_geometry_opendrive_path))
        planner.hdmap.lane_network.add_edge("30_-1", "2_-1", weight=0.0, lane_change=False)
        planner.set_start_goal((7.0, -701.0), (2.0, -701.0))
        assert [lane.uid for lane in planner.plan().lane_path] == ["30_-1", *STRAIGHT_ON, "33_1", "30_-1"]

    @pytest.mark.parametrize("start, goal", [
        ((30.0, -1.75), (10.0, -1.75)),  # Same lane
        ((90.0, -1.75), (60.0, -5.25)),  # Other lane of the same section
    ])
    def test_goal_behind_with_no_way_round_returns_no_plan(self, lane_geometry_opendrive_path, start, goal):
        assert _plan(lane_geometry_opendrive_path, start, goal) is None

    def test_no_drivable_lane_returns_no_plan(self, tmp_path, lane_geometry_opendrive_path):
        no_lanes = tmp_path / "no_driving.xodr"
        no_lanes.write_text(lane_geometry_opendrive_path.read_text().replace('type="driving"', 'type="sidewalk"'))
        assert _plan(no_lanes, (7.0, -701.0), (2.0, -701.0)) is None

    @pytest.mark.parametrize("goal_x", [3.0, 3.5, 5.0])
    def test_short_routes_plan(self, lane_geometry_opendrive_path, goal_x):
        # Smoothing used to crash on routes of 3, 4 or 6 points
        plan = _plan(lane_geometry_opendrive_path, (2.0, -701.0), (goal_x, -701.0))
        assert abs(plan.path[-1][0] - goal_x) < 1.0


class TestHDMapGlobalPlannerTwoWayLanes:
    @pytest.mark.parametrize("start, goal, heading, route", [
        ((2.0, -701.0), (8.0, -701.0), None, ["30_-1"]),
        ((8.0, -701.0), (2.0, -701.0), None, ["30_-1_reverse"]),
        ((2.0, -701.0), (15.0, -689.0), None, ["30_-1_reverse", "33_1_reverse"]),
        ((9.0, -701.0), (15.0, -689.0), None, ["30_-1", *STRAIGHT_ON, "33_1"]),
        ((2.0, -701.0), (15.0, -689.0), 0.0, ["30_-1", *STRAIGHT_ON, "33_1"]),
        ((9.0, -701.0), (15.0, -689.0), np.pi, ["30_-1_reverse", "33_1_reverse"]),
        ((5.0, -689.0), (15.0, -689.0), 0.0, ["33_1_reverse"]),
        ((7.0, -701.0), (2.0, -701.0), 0.0, ["30_-1", *STRAIGHT_ON, "33_1", "30_-1"]),
        ((7.0, -701.0), (2.0, -701.0), np.pi, ["30_-1_reverse"]),
    ])
    def test_two_way_routes(self, two_way_opendrive_path, start, goal, heading, route):
        plan = _plan(two_way_opendrive_path, start, goal, heading)
        lanes, ends = [lane.uid for lane in plan.lane_path], np.array(plan.path)[[0, -1]]
        assert lanes == route
        assert np.allclose(ends, [start, goal], atol=1.0)

    def test_no_plan_when_no_route_goes_the_way_the_vehicle_faces(self, two_way_opendrive_path):
        start, goal = (120.0, -1.75), (20.0, -1.75)
        no_plan = _plan(two_way_opendrive_path, start, goal, heading=0.0) is None
        lanes = [lane.uid for lane in _plan(two_way_opendrive_path, start, goal, heading=np.pi).lane_path]
        assert no_plan and lanes == ["2_-1_reverse", "1_1_-1_reverse", "1_0_-1_reverse"]

    def test_one_way_lanes_ignore_the_heading(self, lane_geometry_opendrive_path):
        plan = _plan(lane_geometry_opendrive_path, (7.0, -701.0), (2.0, -701.0), heading=np.pi)
        assert [lane.uid for lane in plan.lane_path] == ["30_-1", *STRAIGHT_ON, "33_1", "30_-1"]
