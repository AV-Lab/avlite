"""Tests for AVLite issue #64: car doesn't stop at the goal.

HDMapGlobalPlanner plans (and the shipped saved plan) end above 0 m/s,
so the only thing that can stop the car — the velocity profile reaching
exactly 0 at the final waypoint — never happens. Three mechanisms:

1. smoothen_path_savgol() thins points closer than min_spacing (0.5 m)
   together with their velocity entries, dropping the terminal point
   and its 0.0 entry (the shipped plan's ReferenceSpeed ends at
   0.526 m/s, above Stanley's 0.5 m/s emergency-braking threshold).
2. Short plans (n < 2 * wp_to_full_velocity = 40 waypoints) get a FLAT
   max_velocity profile with no deceleration at all.
3. The file-loaded path (GlobalPlan.from_file) never enforces a stop.
4. Separately, HDMapGlobalPlanner.plan() resets the plan metadata with
   a bare GlobalPlan(), so goal_point is (0.0, 0.0) and the
   GoalArrivalMonitor watches the map origin instead of the real goal.

Status: APPLIED 28 Nov 2026 on branch fix/goal-arrival-stop.
No CARLA, no simulator, no HD map file: the velocity/smoothing/file
logic is tested directly on synthetic plans and the shipped JSON.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c20_planning.c24_global_hdmap_planners import (
    smoothen_path_savgol,
    velocity_profile_by_arclength,
)
from avlite.c40_execution.c47_execution_tasks import GoalArrivalMonitor

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_PLAN = REPO_ROOT / "avlite" / "data" / "20260702_045340_global_plan.json"


def _dense_plan(n=60, spacing=0.3):
    """Straight dense path; velocity ramps 10 -> 0 like the old profile."""
    path = [(i * spacing, 0.0) for i in range(n)]
    velocity = list(np.linspace(10.0, 0.0, n))
    return GlobalPlan(
        path=path,
        velocity=velocity,
        left_boundary_d=[1.0] * n,
        right_boundary_d=[-1.0] * n,
    )


# ---------------------------------------------------------------------------
# Hole 2 — smoothing drops the terminal point and its 0.0 velocity entry
# ---------------------------------------------------------------------------


def test_smoothen_keeps_terminal_point_and_velocity():
    plan = _dense_plan()
    n_before = len(plan.path)
    last_point, last_v = plan.path[-1], plan.velocity[-1]
    assert last_v == 0.0
    out = smoothen_path_savgol(plan, min_spacing=0.5, window_length=7, polyorder=3)
    assert len(out.path) < n_before  # thinning actually happened
    # The goal waypoint survives thinning. Savgol smoothing itself shifts
    # coordinates slightly (here ~0.02 m), so compare with tolerance;
    # the pre-fix failure dropped the point entirely (last x was 17.4).
    assert abs(float(out.path[-1][0]) - last_point[0]) < 0.05
    assert abs(float(out.path[-1][1]) - last_point[1]) < 0.05
    assert out.velocity[-1] == 0.0  # ... and so does the standstill


# ---------------------------------------------------------------------------
# Velocity profile — built AFTER smoothing, by arc length (new helper)
# ---------------------------------------------------------------------------


def test_velocity_profile_short_plan_tapers_to_zero():
    # 4.5 m total: shorter than 2 * ramp_m, so the ramps meet mid-plan
    # (triangular profile). The old code gave such plans a FLAT
    # max_velocity profile with no deceleration at all.
    path = [(i * 0.5, 0.0) for i in range(10)]
    v = velocity_profile_by_arclength(path, max_velocity=10.0, min_v=3.0, ramp_m=10.0)
    assert len(v) == 10
    assert v[0] == 3.0
    assert max(v) < 10.0  # never reaches cruise on a 4.5 m plan
    assert v[-1] == 0.0


def test_velocity_profile_long_plan_ramps_cruises_and_stops():
    path = [(i * 0.5, 0.0) for i in range(101)]  # 50 m
    v = velocity_profile_by_arclength(path, max_velocity=10.0, min_v=3.0, ramp_m=10.0)
    assert len(v) == 101
    assert v[0] == 3.0
    assert max(v) == 10.0
    assert v[-1] == 0.0
    # Decel tail: the last 10 m strictly decrease to 0.
    tail = v[-21:]
    assert all(b < a for a, b in zip(tail, tail[1:]))


def test_velocity_profile_degenerate_paths():
    assert velocity_profile_by_arclength([], 10.0, 3.0) == []
    assert velocity_profile_by_arclength([(0.0, 0.0)], 10.0, 3.0) == [0.0]
    assert velocity_profile_by_arclength([(0.0, 0.0), (0.0, 0.0)], 10.0, 3.0) == [0.0, 0.0]


# ---------------------------------------------------------------------------
# Hole 3 — file-loaded plans must end at a standstill
# ---------------------------------------------------------------------------


def test_from_file_shipped_plan_ends_at_standstill():
    plan = GlobalPlan.from_file(str(SHIPPED_PLAN))
    assert plan.velocity[-1] == 0.0
    assert plan.goal_point == plan.path[-1]
    assert plan.start_point == plan.path[0]
    assert len(plan.velocity) == len(plan.path)


# ---------------------------------------------------------------------------
# Hole D end to end — plan() keeps the goal metadata (stub HD map)
# ---------------------------------------------------------------------------

import networkx as nx

from avlite.c20_planning.c24_global_hdmap_planners import HDMapGlobalPlanner


def _stub_plan():
    """One straight lane, x = 0..100 m, points every 0.5 m."""
    xs = np.arange(0.0, 100.5, 0.5)
    lane = SimpleNamespace(
        uid="lane-1",
        id="-1",
        road_id="road-1",
        lane_section_idx=0,
        width=3.5,
        center_line=np.vstack([xs, np.zeros_like(xs)]),
        reverse=None,
    )
    graph = nx.DiGraph()
    graph.add_node("lane-1")
    hdmap = SimpleNamespace(
        road_network={"road-1": True},
        lane_network=graph,
        lane_by_uid={"lane-1": lane},
        find_nearest_road=lambda *xy: SimpleNamespace(uid="road-1"),
        find_nearest_lane_and_idx=lambda x, y: (
            lane,
            int(np.argmin(np.abs(xs - x))),
        ),
    )
    planner = HDMapGlobalPlanner(hdmap, max_velocity=10.0, wp_to_full_velocity=20)
    planner.set_start_goal((5.0, 0.0), (95.0, 0.0))
    return planner.plan()


def test_plan_keeps_goal_metadata_and_stops():
    plan = _stub_plan()
    # Hole D: pre-fix, plan() reset these to (0.0, 0.0).
    assert plan.start_point == (5.0, 0.0)
    assert plan.goal_point == (95.0, 0.0)
    # Holes 1+2: the profile ends at a standstill on the smoothed path.
    assert len(plan.velocity) == len(plan.path)
    assert plan.velocity[-1] == 0.0
    assert abs(float(plan.path[-1][0]) - 95.0) < 0.1
    assert plan.race_mode is False


# ---------------------------------------------------------------------------
# Hole D consequence — the monitor watches global_plan.goal_point
# ---------------------------------------------------------------------------


def _executer(goal, ego_xy):
    notified = []
    task_runner = SimpleNamespace(notify=lambda event: notified.append(event))
    local_planner = SimpleNamespace(global_plan=GlobalPlan(goal_point=goal))
    ego_state = SimpleNamespace(x=ego_xy[0], y=ego_xy[1])
    executer = SimpleNamespace(
        local_planner=local_planner, ego_state=ego_state, task_runner=task_runner
    )
    return executer, notified


def test_goal_arrival_monitor_fires_at_real_goal():
    executer, notified = _executer(goal=(100.0, 100.0), ego_xy=(100.0, 100.0))
    GoalArrivalMonitor().execute(executer)
    assert len(notified) == 1  # GoalArrived raised


def test_goal_arrival_monitor_ignores_origin_when_goal_is_elsewhere():
    executer, notified = _executer(goal=(100.0, 100.0), ego_xy=(0.0, 0.0))
    GoalArrivalMonitor().execute(executer)
    assert notified == []  # ego at the origin is NOT at the goal


def test_goal_arrival_monitor_symptom_of_reset_goal():
    # Hole D as shipped: plan() resets goal_point to (0.0, 0.0), so the
    # monitor watches the origin. An ego AT the real goal never fires it.
    executer, notified = _executer(goal=(0.0, 0.0), ego_xy=(100.0, 100.0))
    GoalArrivalMonitor().execute(executer)
    assert notified == []  # symptom: arrival at the real goal is missed
