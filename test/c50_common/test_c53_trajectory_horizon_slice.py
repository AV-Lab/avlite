"""Tests for slice_trajectory_horizon (avlite.c50_common.c53_trajectory_tracker)."""

import json

from avlite.c50_common.c53_trajectory_tracker import TrajectoryTracker, slice_trajectory_horizon


def _long_trajectory(n: int = 500, start_wp: int = 10) -> TrajectoryTracker:
    path = [(float(i), float(i * 0.5)) for i in range(n)]
    velocity = [5.0] * n
    traj = TrajectoryTracker(path=path, velocity=velocity)
    traj.current_wp = start_wp
    traj.next_wp = start_wp + 1
    return traj


def test_slice_trajectory_starts_at_current_wp():
    traj = _long_trajectory(n=500, start_wp=42)
    sliced = slice_trajectory_horizon(traj, max_points=50)
    assert len(sliced.path) == 50
    assert sliced.path[0] == traj.path[42]
    assert sliced.path[-1] == traj.path[91]


def test_slice_trajectory_resets_current_wp():
    traj = _long_trajectory(start_wp=25)
    sliced = slice_trajectory_horizon(traj, max_points=50)
    assert sliced.current_wp == 0
    assert sliced.next_wp == 1


def test_slice_trajectory_sets_parent_trajectory():
    traj = _long_trajectory(start_wp=25)
    sliced = slice_trajectory_horizon(traj, max_points=50)
    assert sliced.parent_trajectory is traj


def test_slice_trajectory_json_payload_non_empty():
    traj = _long_trajectory()
    sliced = slice_trajectory_horizon(traj, max_points=50)
    payload = {
        "path": [(float(p[0]), float(p[1])) for p in sliced.path],
        "velocity": [float(v) for v in sliced.velocity],
    }
    data = json.loads(json.dumps(payload))
    assert len(data["path"]) == 50
    assert len(data["velocity"]) == 50


def test_slice_trajectory_near_end():
    traj = _long_trajectory(n=100, start_wp=95)
    sliced = slice_trajectory_horizon(traj, max_points=50)
    assert len(sliced.path) == 5
    assert sliced.path[0] == traj.path[95]


def test_slice_trajectory_zero_max_points_returns_remainder():
    traj = _long_trajectory(n=100, start_wp=10)
    sliced = slice_trajectory_horizon(traj, max_points=0)
    assert len(sliced.path) == 90
    assert sliced.path[0] == traj.path[10]
