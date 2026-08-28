"""Regression tests for TrajectoryTracker waypoint index updates."""

import pytest

from avlite.c50_common.c54_trajectory_tracker import TrajectoryTracker


def _path_tj(n: int = 3) -> TrajectoryTracker:
    path = [(float(i) * 10.0, 0.0) for i in range(n)]
    return TrajectoryTracker(path=path, velocity=[1.0] * n)


def test_update_waypoint_by_wp_at_last_index_keeps_next_in_bounds():
    """``current_wp + 1 % n`` mis-parses as ``current_wp + 1`` and sets next_wp == n."""
    tj = _path_tj(3)
    tj.update_waypoint_by_wp(2)
    assert tj.current_wp == 2
    assert tj.next_wp == 2
    # Must be indexable by plot / step_wp / convert_sd_orientation helpers.
    _ = tj.path_x[tj.next_wp]
    _ = tj.path_y[tj.next_wp]


def test_update_to_next_waypoint_clamps_at_end():
    tj = _path_tj(3)
    tj.update_waypoint_by_wp(1)
    tj.update_to_next_waypoint()
    assert tj.current_wp == 2
    assert tj.next_wp == 2
    tj.update_to_next_waypoint()
    assert tj.current_wp == 2
    assert tj.next_wp == 2


def test_update_waypoint_by_wp_mid_path_advances_next():
    tj = _path_tj(5)
    tj.update_waypoint_by_wp(2)
    assert tj.current_wp == 2
    assert tj.next_wp == 3


def test_update_waypoint_by_xy_before_path_start_resets_indices():
    """Teleport/reset upstream of wp0 must not leave a stale far-path current_wp.

    Stanley uses ``get_current_heading()`` / ``velocity[current_wp]``; a stale
    index after an L-path corner commanded max steer / wrong target speed.
    """
    # Horizontal then vertical: heading at the end is π/2.
    path = [(float(i), 0.0) for i in range(21)] + [(20.0, float(j)) for j in range(1, 11)]
    tj = TrajectoryTracker(path=path, velocity=[10.0] * len(path))
    tj.update_waypoint_by_xy(20.0, 8.0)
    assert tj.current_wp > 20

    tj.update_waypoint_by_xy(-0.5, 0.0)
    assert tj.current_wp == 0
    assert tj.next_wp == 1
    assert tj.get_current_heading() == pytest.approx(0.0, abs=1e-6)


def test_create_quintic_trajectory_sd_honors_boundary_derivatives():
    """b-vector must match A rows: value, value, 1st, 1st, 2nd, 2nd (start then end)."""
    path = [(float(i), 0.0) for i in range(40)]
    tj = TrajectoryTracker(path=path, velocity=[5.0] * 40)
    s0, d0, s1, d1 = 5.0, 0.5, 15.0, -0.25
    local = tj.create_quintic_trajectory_sd(
        s_start=s0,
        d_start=d0,
        s_end=s1,
        d_end=d1,
        start_d_1st_derv=0.2,
        end_d_1st_derv=-0.1,
        start_d_2nd_derv=0.3,
        end_d_2nd_derv=0.05,
        num_points=20,
    )
    poly = local.poly_d
    d1p = poly.deriv(1)
    d2p = poly.deriv(2)
    assert poly(s0) == pytest.approx(d0, abs=1e-9)
    assert poly(s1) == pytest.approx(d1, abs=1e-9)
    assert d1p(s0) == pytest.approx(0.2, abs=1e-9)
    assert d1p(s1) == pytest.approx(-0.1, abs=1e-9)
    assert d2p(s0) == pytest.approx(0.3, abs=1e-9)
    assert d2p(s1) == pytest.approx(0.05, abs=1e-9)
