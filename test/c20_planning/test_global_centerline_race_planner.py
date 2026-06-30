"""Tests for GlobalCenterlineRacePlanner corridor centerline geometry."""
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, Point

from avlite.c20_planning.c25_global_race_planners import GlobalCenterlineRacePlanner
from avlite.c60_common.c67_paths import DataPaths


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
