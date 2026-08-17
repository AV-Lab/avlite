"""Tests for GlobalPlan file validation and load-time velocity ramping."""

from __future__ import annotations

import json

import pytest

from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c20_planning.c29_settings import PlanningSettings


def _valid_plan_payload(**overrides) -> dict:
    data = {
        "ReferenceLine": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        "ReferenceSpeed": [8.0, 8.0, 8.0],
        "LeftBound": [2.0, 2.0, 2.0],
        "RightBound": [-2.0, -2.0, -2.0],
    }
    data.update(overrides)
    return data


def _write_plan(tmp_path, payload: dict, name: str = "plan.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestGlobalPlanIsLoadable:
    def test_accepts_valid_payload(self, tmp_path):
        path = _write_plan(tmp_path, _valid_plan_payload())
        assert GlobalPlan.is_loadable(path) is True

    def test_rejects_missing_keys(self, tmp_path):
        payload = _valid_plan_payload()
        del payload["ReferenceSpeed"]
        assert GlobalPlan.is_loadable(_write_plan(tmp_path, payload)) is False

    def test_rejects_empty_reference_line(self, tmp_path):
        payload = _valid_plan_payload(ReferenceLine=[], ReferenceSpeed=[], LeftBound=[], RightBound=[])
        assert GlobalPlan.is_loadable(_write_plan(tmp_path, payload)) is False

    def test_rejects_nested_bounds(self, tmp_path):
        payload = _valid_plan_payload(LeftBound=[[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
        assert GlobalPlan.is_loadable(_write_plan(tmp_path, payload)) is False

    def test_rejects_non_json(self, tmp_path):
        path = tmp_path / "plan.json"
        path.write_text("not-json", encoding="utf-8")
        assert GlobalPlan.is_loadable(path) is False

    def test_rejects_wrong_suffix(self, tmp_path):
        path = _write_plan(tmp_path, _valid_plan_payload(), name="plan.txt")
        assert GlobalPlan.is_loadable(path) is False


class TestGlobalPlanFromFile:
    def test_loads_path_velocity_and_frenet_bounds(self, tmp_path):
        path = _write_plan(tmp_path, _valid_plan_payload())
        plan = GlobalPlan.from_file(path)
        assert plan.start_point == (0.0, 0.0)
        assert plan.goal_point == (20.0, 0.0)
        assert plan.path == [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
        assert plan.velocity == [8.0, 8.0, 8.0]
        assert plan.left_boundary_d == [2.0, 2.0, 2.0]
        assert plan.right_boundary_d == [-2.0, -2.0, -2.0]
        assert plan.trajectory is not None
        assert len(plan.left_boundary_x) == 3
        assert plan.left_boundary_y[0] == pytest.approx(2.0)

    def test_zero_start_speed_is_replaced_with_min_ramp(self, tmp_path):
        payload = _valid_plan_payload(ReferenceSpeed=[0.0, 8.0, 8.0])
        plan = GlobalPlan.from_file(_write_plan(tmp_path, payload))
        assert plan.velocity[0] == pytest.approx(PlanningSettings.c20_min_ramp_start_velocity)
        assert plan.velocity[1:] == [8.0, 8.0]
        assert plan.trajectory.velocity[0] == pytest.approx(
            PlanningSettings.c20_min_ramp_start_velocity
        )

    def test_negative_start_speed_is_replaced_with_min_ramp(self, tmp_path):
        payload = _valid_plan_payload(ReferenceSpeed=[-1.0, 5.0, 5.0])
        plan = GlobalPlan.from_file(_write_plan(tmp_path, payload))
        assert plan.velocity[0] == pytest.approx(PlanningSettings.c20_min_ramp_start_velocity)
