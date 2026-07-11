"""Unit tests for FpsTracker (avlite.c50_common.c56_fps_tracker).

Tests verify:
- First tick always returns 0.0 (no prior timestamp).
- Subsequent ticks return the wall-clock rate (approximately).
- floor_dt caps FPS at 1/floor_dt when wall-clock is faster.
- floor_dt does NOT inflate FPS when wall-clock is slower.
- reset() restores the tracker to its initial state.
"""
import time

import pytest

from avlite.c50_common.c56_fps_tracker import FpsTracker


class _FakeClock:
    def __init__(self, start: float = 100.0):
        self._t = start

    def time(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += dt


@pytest.fixture
def fake_clock(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(time, "time", clock.time)
    return clock


class TestFpsTrackerFirstTick:
    def test_first_tick_returns_zero(self):
        tracker = FpsTracker()
        fps = tracker.tick()
        assert fps == 0.0

    def test_first_tick_with_floor_dt_returns_zero(self):
        tracker = FpsTracker()
        fps = tracker.tick(floor_dt=0.05)
        assert fps == 0.0

    def test_last_set_after_first_tick(self, fake_clock):
        tracker = FpsTracker()
        tracker.tick()
        assert fake_clock.time() == pytest.approx(tracker.last)


class TestFpsTrackerWallClock:
    def test_second_tick_reflects_elapsed_time(self, fake_clock):
        tracker = FpsTracker()
        tracker.tick()
        fake_clock.advance(0.1)
        fps = tracker.tick()
        assert fps == pytest.approx(10.0, rel=0.1)

    def test_fps_increases_for_shorter_interval(self, fake_clock):
        tracker = FpsTracker()
        tracker.tick()
        fake_clock.advance(0.2)
        fps_slow = tracker.tick()

        tracker.reset()
        tracker.tick()
        fake_clock.advance(0.05)
        fps_fast = tracker.tick()

        assert fps_fast > fps_slow

    def test_last_updated_after_each_tick(self, fake_clock):
        tracker = FpsTracker()
        tracker.tick()
        t1 = tracker.last
        fake_clock.advance(0.02)
        tracker.tick()
        t2 = tracker.last
        assert t2 > t1


class TestFpsTrackerFloorDt:
    def test_floor_dt_caps_fps_when_faster(self, fake_clock):
        tracker = FpsTracker()
        tracker.tick(floor_dt=0.1)
        fake_clock.advance(0.001)
        fps = tracker.tick(floor_dt=0.1)
        expected_cap = 1.0 / 0.1
        assert fps <= expected_cap + 0.5

    def test_floor_dt_does_not_inflate_fps_when_slower(self, fake_clock):
        tracker = FpsTracker()
        tracker.tick(floor_dt=0.01)
        fake_clock.advance(0.2)
        fps = tracker.tick(floor_dt=0.01)
        assert fps <= 20.0

    def test_floor_dt_zero_is_pure_wall_clock(self, fake_clock):
        tracker_a = FpsTracker()
        tracker_b = FpsTracker()
        tracker_a.tick()
        tracker_b.tick()
        fake_clock.advance(0.05)
        fps_a = tracker_a.tick(floor_dt=0.0)
        fps_b = tracker_b.tick()
        assert abs(fps_a - fps_b) < 5.0


class TestFpsTrackerReset:
    def test_reset_clears_last(self, fake_clock):
        tracker = FpsTracker()
        tracker.tick()
        tracker.reset()
        assert tracker.last == 0.0

    def test_first_tick_after_reset_returns_zero(self, fake_clock):
        tracker = FpsTracker()
        tracker.tick()
        fake_clock.advance(0.05)
        tracker.tick()
        tracker.reset()
        fps = tracker.tick()
        assert fps == 0.0

    def test_tracker_measures_correctly_after_reset(self, fake_clock):
        tracker = FpsTracker()
        tracker.tick()
        fake_clock.advance(0.1)
        tracker.tick()
        tracker.reset()
        tracker.tick()
        fake_clock.advance(0.05)
        fps = tracker.tick()
        assert fps >= 10.0
