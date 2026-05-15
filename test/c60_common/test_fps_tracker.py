"""Unit tests for FpsTracker (avlite.c60_common.c65_fps_tracker).

Tests verify:
- First tick always returns 0.0 (no prior timestamp).
- Subsequent ticks return the wall-clock rate (approximately).
- floor_dt caps FPS at 1/floor_dt when wall-clock is faster.
- floor_dt does NOT inflate FPS when wall-clock is slower.
- reset() restores the tracker to its initial state.
"""
import time

import pytest

from avlite.c60_common.c65_fps_tracker import FpsTracker


class TestFpsTrackerFirstTick:
    def test_first_tick_returns_zero(self):
        tracker = FpsTracker()
        fps = tracker.tick()
        assert fps == 0.0

    def test_first_tick_with_floor_dt_returns_zero(self):
        tracker = FpsTracker()
        fps = tracker.tick(floor_dt=0.05)
        assert fps == 0.0

    def test_last_set_after_first_tick(self):
        tracker = FpsTracker()
        before = time.time()
        tracker.tick()
        after = time.time()
        assert before <= tracker.last <= after


class TestFpsTrackerWallClock:
    def test_second_tick_reflects_elapsed_time(self):
        tracker = FpsTracker()
        tracker.tick()
        time.sleep(0.1)
        fps = tracker.tick()
        # Sleep ≈ 0.1 s → expect ≈ 10 fps; allow generous tolerance for CI
        assert 5.0 <= fps <= 50.0

    def test_fps_increases_for_shorter_interval(self):
        tracker = FpsTracker()
        tracker.tick()
        time.sleep(0.2)
        fps_slow = tracker.tick()

        tracker.reset()
        tracker.tick()
        time.sleep(0.05)
        fps_fast = tracker.tick()

        assert fps_fast > fps_slow

    def test_last_updated_after_each_tick(self):
        tracker = FpsTracker()
        tracker.tick()
        t1 = tracker.last
        time.sleep(0.02)
        tracker.tick()
        t2 = tracker.last
        assert t2 > t1


class TestFpsTrackerFloorDt:
    def test_floor_dt_caps_fps_when_faster(self):
        """Wall-clock faster than floor_dt → FPS == 1/floor_dt."""
        tracker = FpsTracker()
        tracker.tick(floor_dt=0.1)
        # No sleep → near-zero wall-clock dt, well below floor_dt=0.1
        fps = tracker.tick(floor_dt=0.1)
        expected_cap = 1.0 / 0.1  # 10.0
        # Should be at or below cap (floating point allows tiny overshoot)
        assert fps <= expected_cap + 0.5

    def test_floor_dt_does_not_inflate_fps_when_slower(self):
        """Wall-clock slower than floor_dt → FPS reflects real rate, not cap."""
        tracker = FpsTracker()
        tracker.tick(floor_dt=0.01)  # floor = 100 Hz
        time.sleep(0.2)             # real rate ≈ 5 Hz (much slower)
        fps = tracker.tick(floor_dt=0.01)
        # Should be around 5 Hz, NOT inflated to 100 Hz
        assert fps <= 20.0

    def test_floor_dt_zero_is_pure_wall_clock(self):
        """floor_dt=0 must behave identically to no floor."""
        tracker_a = FpsTracker()
        tracker_b = FpsTracker()
        tracker_a.tick()
        tracker_b.tick()
        time.sleep(0.05)
        fps_a = tracker_a.tick(floor_dt=0.0)
        fps_b = tracker_b.tick()
        assert abs(fps_a - fps_b) < 5.0  # same within 5 fps given same sleep


class TestFpsTrackerReset:
    def test_reset_clears_last(self):
        tracker = FpsTracker()
        tracker.tick()
        tracker.reset()
        assert tracker.last == 0.0

    def test_first_tick_after_reset_returns_zero(self):
        tracker = FpsTracker()
        tracker.tick()
        time.sleep(0.05)
        tracker.tick()
        tracker.reset()
        fps = tracker.tick()
        assert fps == 0.0

    def test_tracker_measures_correctly_after_reset(self):
        tracker = FpsTracker()
        tracker.tick()
        time.sleep(0.1)
        tracker.tick()
        tracker.reset()
        tracker.tick()
        time.sleep(0.05)
        fps = tracker.tick()
        # Should reflect 0.05 s interval, not the pre-reset history
        assert fps >= 10.0
