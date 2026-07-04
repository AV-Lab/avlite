"""Unit tests for TrajectoryTracker.concatenate.

Tests verify:
- Junction deduplication when endpoints are coincident.
- Junction deduplication when gap is within tolerance.
- Bridge insertion when gap exceeds tolerance.
- Resulting trajectory is initialised and path_s is monotonically increasing.
- parent_trajectory is None (standalone result).
- Custom name is forwarded; default name combines both names.
- Empty velocity lists are handled gracefully.
"""
import pytest

from avlite.c50_common.c53_trajectory_tracker import TrajectoryTracker


def _straight(x_start, x_end, n=10, velocity=1.0):
    """Horizontal straight line from (x_start, 0) to (x_end, 0)."""
    xs = [x_start + (x_end - x_start) * i / (n - 1) for i in range(n)]
    path = [(x, 0.0) for x in xs]
    vel = [velocity] * n
    return TrajectoryTracker(path=path, velocity=vel)


class TestConcatenateCoincidentJunction:
    def test_coincident_no_duplicate(self):
        t1 = _straight(0, 10)
        t2 = _straight(10, 20)
        result = t1.concatenate(t2)
        assert len(result.path) == len(t1.path) + len(t2.path) - 1

    def test_velocity_deduplicated(self):
        t1 = _straight(0, 10, velocity=1.0)
        t2 = _straight(10, 20, velocity=2.0)
        result = t1.concatenate(t2)
        assert len(result.velocity) == len(t1.velocity) + len(t2.velocity) - 1

    def test_path_endpoints_correct(self):
        t1 = _straight(0, 10)
        t2 = _straight(10, 20)
        result = t1.concatenate(t2)
        assert result.path[0] == pytest.approx(t1.path[0])
        assert result.path[-1] == pytest.approx(t2.path[-1])


class TestConcatenateWithinTolerance:
    def test_small_gap_treated_as_coincident(self):
        t1 = _straight(0, 10)
        t2 = _straight(10.5, 20)   # gap = 0.5 m, default tolerance = 1.0
        result = t1.concatenate(t2, gap_tolerance=1.0)
        assert len(result.path) == len(t1.path) + len(t2.path) - 1


class TestConcatenateBridging:
    def test_large_gap_inserts_bridge(self):
        t1 = _straight(0, 10)
        t2 = _straight(15, 25)   # gap = 5 m
        bridge_points = 4
        result = t1.concatenate(t2, gap_tolerance=1.0, bridge_points=bridge_points)
        assert len(result.path) == len(t1.path) + bridge_points + len(t2.path)

    def test_large_gap_emits_warning(self):
        from unittest.mock import patch
        import avlite.c50_common.c53_trajectory_tracker as mod
        t1 = _straight(0, 10)
        t2 = _straight(15, 25)
        with patch.object(mod.log, "warning") as mock_warn:
            t1.concatenate(t2, gap_tolerance=1.0)
        mock_warn.assert_called_once()
        assert "bridging" in mock_warn.call_args[0][0]

    def test_bridge_is_straight_line(self):
        t1 = _straight(0, 10)
        t2 = _straight(20, 30)
        result = t1.concatenate(t2, gap_tolerance=1.0, bridge_points=3)
        bridge_start = len(t1.path)
        bridge_end = bridge_start + 3
        for pt in result.path[bridge_start:bridge_end]:
            assert pt[1] == pytest.approx(0.0)
        for pt in result.path[bridge_start:bridge_end]:
            assert 10.0 < pt[0] < 20.0


class TestConcatenateResultProperties:
    def test_is_initialized(self):
        result = _straight(0, 10).concatenate(_straight(10, 20))
        assert result.is_initialized

    def test_path_s_monotonically_increasing(self):
        result = _straight(0, 10).concatenate(_straight(10, 20))
        for a, b in zip(result.path_s, result.path_s[1:]):
            assert b >= a

    def test_parent_trajectory_is_none(self):
        result = _straight(0, 10).concatenate(_straight(10, 20))
        assert result.parent_trajectory is None

    def test_default_name_combines_both(self):
        t1 = _straight(0, 10)
        t1.name = "Seg1"
        t2 = _straight(10, 20)
        t2.name = "Seg2"
        result = t1.concatenate(t2)
        assert "Seg1" in result.name and "Seg2" in result.name

    def test_custom_name_forwarded(self):
        result = _straight(0, 10).concatenate(_straight(10, 20), name="Combined")
        assert result.name == "Combined"


class TestConcatenateEmptyVelocity:
    def test_empty_velocity_does_not_raise(self):
        path1 = [(float(i), 0.0) for i in range(10)]
        path2 = [(float(i + 10), 0.0) for i in range(10)]
        t1 = TrajectoryTracker(path=path1, velocity=[])
        t2 = TrajectoryTracker(path=path2, velocity=[])
        result = t1.concatenate(t2)
        assert result.is_initialized
