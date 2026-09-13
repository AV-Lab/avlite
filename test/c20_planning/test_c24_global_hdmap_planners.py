"""Tests for HDMap global-plan smoothing."""

import pytest

from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c20_planning.c24_global_hdmap_planners import smoothen_path_savgol


def _plan_with_n_points(n: int, spacing: float = 1.0) -> GlobalPlan:
    plan = GlobalPlan()
    plan.path = [(i * spacing, 0.0) for i in range(n)]
    plan.velocity = [1.0] * n
    plan.left_boundary_d = [1.0] * n
    plan.right_boundary_d = [-1.0] * n
    return plan


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8])
def test_smoothen_path_savgol_handles_short_paths(n):
    """Near-duplicate pruning can leave 3–6 points on a short same-lane HDMap route."""
    out = smoothen_path_savgol(_plan_with_n_points(n))
    assert len(out.path) == n
    assert len(out.velocity) == n
    assert len(out.left_boundary_d) == n
    assert len(out.right_boundary_d) == n


def test_smoothen_path_savgol_prunes_near_duplicates_then_smooths():
    plan = _plan_with_n_points(4, spacing=0.1)
    plan.path.append((3.0, 0.0))
    plan.velocity.append(1.0)
    plan.left_boundary_d.append(1.0)
    plan.right_boundary_d.append(-1.0)
    out = smoothen_path_savgol(plan, min_spacing=0.5)
    assert len(out.path) >= 2
    assert len(out.path) == len(out.velocity)
