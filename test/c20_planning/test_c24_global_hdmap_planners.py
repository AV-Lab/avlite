"""Tests for HD-map global planner geometry helpers and plan() edge cases.

chop_path / chop_path_from_two_sides encode OpenDRIVE lane-id sign (negative
lanes travel increasing s; positive lanes are reversed). A regression here
emits a reversed or truncated global path. _inset_d and smoothen_path_savgol
keep corridor boundaries and velocity samples aligned with the reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from avlite.c10_perception.c11_perception_model import HDMap
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c20_planning.c24_global_hdmap_planners import (
    HDMapGlobalPlanner,
    _inset_d,
    chop_path,
    chop_path_from_two_sides,
    smoothen_path_savgol,
)
from avlite.c20_planning.c29_settings import PlanningSettings


def _xy_path(n: int = 10) -> np.ndarray:
    return np.column_stack([np.arange(n, dtype=float), np.zeros(n)])


class TestInsetD:
    def test_margin_shrinks_corridor_inward(self):
        left, right = _inset_d(1.75, -1.75, 0.25)
        assert left == pytest.approx(1.5)
        assert right == pytest.approx(-1.5)


class TestChopPath:
    def test_start_of_negative_lane_keeps_tail(self):
        path = _xy_path(10)
        chopped = chop_path(path, lane_id=-1, idx=3, start=True)
        np.testing.assert_array_equal(chopped, path[3:])

    def test_end_of_negative_lane_keeps_head(self):
        path = _xy_path(10)
        chopped = chop_path(path, lane_id=-1, idx=3, start=False)
        np.testing.assert_array_equal(chopped, path[:4])

    def test_start_of_positive_lane_reverses_after_chop(self):
        path = _xy_path(10)
        chopped = chop_path(path, lane_id=1, idx=3, start=True)
        # Positive lanes travel decreasing s, so the driving-direction prefix
        # is path[:idx+1] reversed.
        np.testing.assert_array_equal(chopped, path[:4][::-1])

    def test_end_of_positive_lane_reverses_after_chop(self):
        path = _xy_path(10)
        chopped = chop_path(path, lane_id=1, idx=3, start=False)
        np.testing.assert_array_equal(chopped, path[3:][::-1])


class TestChopPathFromTwoSides:
    def test_negative_lane_slices_inclusive(self):
        path = _xy_path(10)
        chopped = chop_path_from_two_sides(path, lane_id=-1, s_idx=2, g_idx=6)
        np.testing.assert_array_equal(chopped, path[2:7])

    def test_positive_lane_slices_then_reverses(self):
        path = _xy_path(10)
        chopped = chop_path_from_two_sides(path, lane_id=1, s_idx=6, g_idx=2)
        np.testing.assert_array_equal(chopped, path[2:7][::-1])


class TestSmoothenPathSavgol:
    def test_short_path_is_unchanged(self):
        plan = GlobalPlan(
            path=[(0.0, 0.0)],
            velocity=[5.0],
            left_boundary_d=[1.0],
            right_boundary_d=[-1.0],
        )
        out = smoothen_path_savgol(plan)
        assert out.path == [(0.0, 0.0)]
        assert out.velocity == [5.0]

    def test_near_duplicates_drop_and_keep_arrays_aligned(self):
        plan = GlobalPlan(
            path=[(0.0, 0.0), (0.1, 0.0), (2.0, 0.0), (4.0, 0.0), (6.0, 0.0)],
            velocity=[3.0, 4.0, 5.0, 6.0, 7.0],
            left_boundary_d=[1.5, 1.5, 1.5, 1.5, 1.5],
            right_boundary_d=[-1.5, -1.5, -1.5, -1.5, -1.5],
        )
        out = smoothen_path_savgol(plan, min_spacing=0.5, window_length=3, polyorder=1)
        assert len(out.path) == len(out.velocity) == len(out.left_boundary_d) == len(out.right_boundary_d)
        assert len(out.path) == 4  # (0.1, 0) is within min_spacing of origin
        assert out.velocity[0] == pytest.approx(3.0)
        assert out.velocity[-1] == pytest.approx(7.0)


class TestHDMapGlobalPlanner:
    def test_plan_without_start_goal_returns_none(self, minimal_opendrive_path):
        hdmap = HDMap.from_path(minimal_opendrive_path)
        planner = HDMapGlobalPlanner(hdmap)
        assert planner.plan() is None

    def test_plan_on_single_lane_fixture_is_monotonic(self, minimal_opendrive_path):
        hdmap = HDMap.from_path(minimal_opendrive_path)
        # Isolated fixture roads are parsed but not graph-connected (empty <link/>).
        # Seed nodes so plan() can Dijkstra a same-lane route.
        for road in hdmap.roads:
            hdmap.road_network.add_node(road.id)
        for lane in hdmap.lanes:
            hdmap.lane_network.add_node(lane.uid)
        planner = HDMapGlobalPlanner(hdmap, max_velocity=10.0, wp_to_full_velocity=5)
        # Lane -1 (right) center sits near y = -width/2 on this straight road.
        planner.set_start_goal((10.0, -1.75), (80.0, -1.75))
        plan = planner.plan()
        assert plan is not None
        assert len(plan.path) >= 2
        xs = [p[0] for p in plan.path]
        assert xs[-1] > xs[0]
        assert len(plan.velocity) == len(plan.path)
        assert len(plan.left_boundary_d) == len(plan.path)
        assert plan.trajectory is not None
        assert plan.race_mode is False
        margin = PlanningSettings.c20_boundary_margin
        assert plan.left_boundary_d[0] == pytest.approx(1.75 - margin)
        assert plan.right_boundary_d[0] == pytest.approx(-1.75 + margin)
        assert plan.velocity[0] == pytest.approx(PlanningSettings.c20_min_ramp_start_velocity)
        assert max(plan.velocity) == pytest.approx(10.0)
        # Decel samples sit closer than smoothen min_spacing, so the terminal
        # 0 m/s waypoint can be dropped; the ramp start must still be kept.
