"""Tests for GlobalCenterlineRacePlanner and GlobalRacePlanner geometry."""
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, Point

from avlite.c10_perception.c11_perception_model import RaceMap
from avlite.c20_planning.c25_global_race_planners import (
    GlobalCenterlineRacePlanner,
    GlobalRacePlanner,
)
from avlite.c60_apps.c68_paths import DataPaths


def _max_corridor_bias(center_pts, left_pts, right_pts, n_samples: int = 200) -> float:
    left_ls = LineString(left_pts)
    right_ls = LineString(right_pts)
    center_ls = LineString(center_pts)
    biases = []
    for s in np.linspace(0, center_ls.length, n_samples):
        pt = center_ls.interpolate(s)
        d_left = pt.distance(left_ls)
        d_right = pt.distance(right_ls)
        if d_left + d_right > 1.0:
            biases.append(abs((d_left - d_right) / (d_left + d_right)))
    return max(biases) if biases else 0.0


class TestGlobalCenterlineRacePlannerSynthetic:
    def test_centerline_is_corridor_centered(self, minimal_corridor_map_path):
        planner = GlobalCenterlineRacePlanner(str(minimal_corridor_map_path), margin=0.0)
        plan = planner.plan()

        center = np.array(list(zip(plan.trajectory.path_x, plan.trajectory.path_y)))
        left = np.array(list(zip(plan.left_boundary_x, plan.left_boundary_y)))
        right = np.array(list(zip(plan.right_boundary_x, plan.right_boundary_y)))

        assert len(center) > 0
        assert _max_corridor_bias(center, left, right) < 0.1

    def test_margin_narrows_corridor(self, minimal_corridor_map_path):
        plan_no_margin = GlobalCenterlineRacePlanner(str(minimal_corridor_map_path), margin=0.0).plan()
        plan_with_margin = GlobalCenterlineRacePlanner(str(minimal_corridor_map_path), margin=0.5).plan()

        center_no = np.array(list(zip(plan_no_margin.trajectory.path_x, plan_no_margin.trajectory.path_y)))
        left_no = np.array(list(zip(plan_no_margin.left_boundary_x, plan_no_margin.left_boundary_y)))
        right_no = np.array(list(zip(plan_no_margin.right_boundary_x, plan_no_margin.right_boundary_y)))

        center_m = np.array(list(zip(plan_with_margin.trajectory.path_x, plan_with_margin.trajectory.path_y)))
        left_m = np.array(list(zip(plan_with_margin.left_boundary_x, plan_with_margin.left_boundary_y)))
        right_m = np.array(list(zip(plan_with_margin.right_boundary_x, plan_with_margin.right_boundary_y)))

        left_ls_no = LineString(left_no)
        right_ls_no = LineString(right_no)
        left_ls_m = LineString(left_m)
        right_ls_m = LineString(right_m)
        center_ls_no = LineString(center_no)
        center_ls_m = LineString(center_m)

        def avg_corridor_width(center_ls, left_ls, right_ls, n_samples: int = 100) -> float:
            widths = []
            for s in np.linspace(0, center_ls.length, n_samples):
                pt = center_ls.interpolate(s)
                widths.append(pt.distance(left_ls) + pt.distance(right_ls))
            return float(np.mean(widths))

        width_no = avg_corridor_width(center_ls_no, left_ls_no, right_ls_no)
        width_m = avg_corridor_width(center_ls_m, left_ls_m, right_ls_m)

        assert width_m < width_no
        assert width_no - width_m > 0.5


def _path_curvature(pts: np.ndarray) -> np.ndarray:
    dx = np.gradient(pts[:, 0])
    dy = np.gradient(pts[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    return np.abs(dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2) ** 1.5, 1e-12)


def _quarter_arc_race_map(radius: float = 30.0, half_width: float = 5.0, n: int = 60) -> RaceMap:
    """90-degree left-curving corner corridor centered on the origin.

    Travel direction is counter-clockwise, so the inner (smaller-radius)
    boundary is on the left of the direction of travel.
    """
    theta = np.linspace(0.0, np.pi / 2.0, n)
    inner = np.column_stack([(radius - half_width) * np.cos(theta), (radius - half_width) * np.sin(theta)])
    outer = np.column_stack([(radius + half_width) * np.cos(theta), (radius + half_width) * np.sin(theta)])
    return RaceMap(source_path="synthetic_quarter_arc", left_bound=inner, right_bound=outer)


class TestGlobalRacePlannerSynthetic:
    def test_straight_corridor_raceline_is_straight_and_in_bounds(self, minimal_corridor_map_path):
        margin = 0.5
        plan = GlobalRacePlanner(str(minimal_corridor_map_path), margin=margin).plan()
        raceline = np.array(plan.path)

        assert len(raceline) > 5
        # Straight corridor along y in [-5, 5]: raceline stays inside minus margin.
        assert np.all(raceline[:, 1] <= 5.0 - margin + 1e-3)
        assert np.all(raceline[:, 1] >= -5.0 + margin - 1e-3)
        # Optimal line is straight: negligible curvature everywhere.
        assert float(np.max(_path_curvature(raceline)[2:-2])) < 1e-3
        # Not longer than the centerline.
        centerline_plan = GlobalCenterlineRacePlanner(str(minimal_corridor_map_path), margin=margin).plan()
        assert LineString(raceline).length <= LineString(centerline_plan.path).length + 0.1

    def test_corner_raceline_shorter_and_flatter_than_centerline(self):
        radius, half_width, margin = 30.0, 5.0, 0.5
        race_map = _quarter_arc_race_map(radius, half_width)

        race_plan = GlobalRacePlanner(race_map, margin=margin).plan()
        center_plan = GlobalCenterlineRacePlanner(race_map, margin=margin).plan()
        raceline = np.array(race_plan.path)
        centerline = np.array(center_plan.path)

        # Raceline stays inside the annular corridor minus the margin.
        radii = np.linalg.norm(raceline, axis=1)
        assert np.all(radii >= radius - half_width + margin - 1e-2)
        assert np.all(radii <= radius + half_width - margin + 1e-2)

        assert LineString(raceline).length < LineString(centerline).length
        assert float(np.max(_path_curvature(raceline)[2:-2])) < float(
            np.max(_path_curvature(centerline)[2:-2])
        )

    def test_velocity_respects_acceleration_limits(self):
        max_lat, max_lon, max_brake, v_max = 4.0, 2.0, 3.0, 15.0
        plan = GlobalRacePlanner(
            _quarter_arc_race_map(),
            max_velocity=v_max,
            max_lateral_accel=max_lat,
            max_longitudinal_accel=max_lon,
            max_braking_decel=max_brake,
            margin=0.5,
        ).plan()

        v = np.array(plan.velocity)
        pts = np.array(plan.path)
        assert np.all(v > 0.0)
        assert np.all(v <= v_max + 1e-6)

        # Lateral: v²·κ ≤ a_lat (interior points; endpoints have one-sided diffs).
        kappa = _path_curvature(pts)
        assert np.all((v**2 * kappa)[2:-2] <= max_lat * 1.05 + 1e-6)

        # Longitudinal: v² changes bounded by 2·a·Δs on every segment.
        ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        dv2 = np.diff(v**2)
        assert np.all(dv2 <= 2.0 * max_lon * ds + 1e-6)
        assert np.all(-dv2 <= 2.0 * max_brake * ds + 1e-6)

    def test_boundary_offsets_are_waypoint_aligned_and_signed(self):
        plan = GlobalRacePlanner(_quarter_arc_race_map(), margin=0.5).plan()
        n = len(plan.path)
        assert len(plan.left_boundary_d) == n
        assert len(plan.right_boundary_d) == n
        assert len(plan.left_boundary_x) == n
        assert len(plan.right_boundary_x) == n
        # Positive d = left, negative d = right, everywhere.
        assert np.all(np.array(plan.left_boundary_d) > 0.0)
        assert np.all(np.array(plan.right_boundary_d) < 0.0)


@pytest.fixture(scope="module")
def yas_marina_map_path():
    path = Path(DataPaths.resolve("data/race_boundary_yas_marina.map.json"))
    if not path.is_file():
        pytest.skip("Yas Marina boundary map not available")
    return str(path)


@pytest.mark.requires_data
class TestGlobalCenterlineRacePlannerYasMarina:
    def test_centerline_is_corridor_centered(self, yas_marina_map_path):
        planner = GlobalCenterlineRacePlanner(yas_marina_map_path, margin=0.0)
        plan = planner.plan()

        center = np.array(list(zip(plan.trajectory.path_x, plan.trajectory.path_y)))
        left = np.array(list(zip(plan.left_boundary_x, plan.left_boundary_y)))
        right = np.array(list(zip(plan.right_boundary_x, plan.right_boundary_y)))

        assert len(center) > 0
        assert _max_corridor_bias(center, left, right) < 0.1

    def test_hairpin_region_is_not_heavily_biased(self, yas_marina_map_path):
        planner = GlobalCenterlineRacePlanner(yas_marina_map_path, margin=0.0)
        plan = planner.plan()

        center = np.array(list(zip(plan.trajectory.path_x, plan.trajectory.path_y)))
        left = np.array(list(zip(plan.left_boundary_x, plan.left_boundary_y)))
        right = np.array(list(zip(plan.right_boundary_x, plan.right_boundary_y)))
        left_ls = LineString(left)
        right_ls = LineString(right)

        hairpin = np.array([435.0, -705.0])
        idx = int(np.argmin(np.linalg.norm(center - hairpin, axis=1)))
        pt = Point(center[idx])
        d_left = pt.distance(left_ls)
        d_right = pt.distance(right_ls)
        bias = abs((d_left - d_right) / (d_left + d_right))
        assert bias < 0.05


@pytest.mark.requires_data
class TestGlobalRacePlannerYasMarina:
    def test_raceline_in_bounds_and_faster_than_centerline(self, yas_marina_map_path):
        # Same velocity limits for both planners so the lap-time comparison
        # reflects the raceline geometry, not the (Super Formula) c25 defaults.
        margin, v_max, a_lat = 0.5, 10.0, 5.0
        race_plan = GlobalRacePlanner(
            yas_marina_map_path, max_velocity=v_max, max_lateral_accel=a_lat, margin=margin
        ).plan()
        center_plan = GlobalCenterlineRacePlanner(
            yas_marina_map_path, max_velocity=v_max, max_lateral_accel=a_lat, margin=margin
        ).plan()

        raceline = np.array(race_plan.path)
        assert len(raceline) > 100

        race_map = RaceMap.from_path(yas_marina_map_path)
        left_ls = LineString(race_map.left_bound)
        right_ls = LineString(race_map.right_bound)
        for p in raceline:
            pt = Point(p)
            assert pt.distance(left_ls) >= margin - 0.1
            assert pt.distance(right_ls) >= margin - 0.1

        def lap_time(plan) -> float:
            pts = np.array(plan.path)
            v = np.array(plan.velocity)
            ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            v_seg = np.maximum((v[:-1] + v[1:]) / 2.0, 1e-3)
            return float(np.sum(ds / v_seg))

        assert lap_time(race_plan) < lap_time(center_plan)
