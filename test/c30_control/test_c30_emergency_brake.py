"""Regression tests for controller emergency brake, anti-windup, and CTE slow-down.

Covers Stanley (default path-follower) and PurePursuit.velocity_pid — the shared
safety contract that overrides weak PID when the reference speed collapses,
clears integral windup at rest, and refuses further deceleration when stopped.
"""

import pytest

from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c30_control.c34_stanley import StanleyController
from avlite.c30_control.c35_pure_pursuit import PurePursuitController
from avlite.c30_control.c39_settings import ControlSettings, ControlSettingsSchema
from avlite.c50_common.c54_trajectory_tracker import TrajectoryTracker


def _straight_path(
    x_end: float = 100.0,
    n: int = 21,
    velocity: float | list[float] = 5.0,
) -> TrajectoryTracker:
    xs = [x_end * i / (n - 1) for i in range(n)]
    path = [(x, 0.0) for x in xs]
    if isinstance(velocity, (int, float)):
        speeds = [float(velocity)] * n
    else:
        speeds = list(velocity)
    return TrajectoryTracker(path=path, velocity=speeds)


def _stanley_settings(**overrides) -> ControlSettingsSchema:
    defaults = {
        "c34_stanley_k": 2.0,
        "c34_stanley_k_soft": 0.01,
        "c34_stanley_lookahead": 0,
        "c34_stanley_valpha": 0.0,
        "c34_stanley_vbeta": 0.0,
        "c34_stanley_vgamma": 0.0,
        "c34_stanley_slow_down_cte": 0.5,
        "c34_stanley_slow_down_heading_cte": 0.5,
        "c34_stanley_slow_down_vel_threshold": 3.0,
        "c32_ego_min_acceleration": -20.0,
        "c32_ego_max_acceleration": 10.0,
    }
    defaults.update(overrides)
    return ControlSettingsSchema(**defaults)


def _pp_settings(**overrides) -> ControlSettingsSchema:
    defaults = {
        "c35_lookahead_distance": 8.0,
        "c35_min_lookahead": 3.0,
        "c35_max_lookahead": 20.0,
        "c35_lookahead_speed_gain": 0.0,
        "c35_valpha": 0.0,
        "c35_vbeta": 0.0,
        "c35_vgamma": 0.0,
        "c32_ego_min_acceleration": -20.0,
        "c32_ego_max_acceleration": 10.0,
    }
    defaults.update(overrides)
    return ControlSettingsSchema(**defaults)


class TestStanleyEmergencyBrake:
    def test_zero_target_while_moving_applies_emergency_decel(self):
        """Near-zero reference speed must hard-brake even when velocity PID gains are zero."""
        trajectory = _straight_path(velocity=0.0)
        controller = StanleyController(tj=trajectory, setting=_stanley_settings())
        ego = EgoState(x=50.0, y=0.0, theta=0.0, velocity=5.0)
        cmd = controller.control(ego)
        expected = (
            ControlSettings.c32_ego_min_acceleration
            * ControlSettings.c30_emergency_braking_factor
        )
        assert cmd.acceleration == pytest.approx(expected, abs=1e-6)
        assert cmd.acceleration < -10.0

    def test_slow_ego_skips_emergency_override(self):
        """Below c30_emergency_min_moving_velocity, weak PID must not be overridden."""
        trajectory = _straight_path(velocity=0.0)
        controller = StanleyController(tj=trajectory, setting=_stanley_settings())
        ego = EgoState(x=50.0, y=0.0, theta=0.0, velocity=0.5)
        cmd = controller.control(ego)
        # v=0.5 ≤ min_moving (1.0) → no emergency; zero gains → acc ~ 0 (floor may apply).
        assert cmd.acceleration == pytest.approx(0.0, abs=1e-6)


class TestStanleyAntiWindupAndFloor:
    def test_stopped_clears_positive_integral_and_floors_negative_acc(self):
        trajectory = _straight_path(velocity=0.0)
        controller = StanleyController(
            tj=trajectory,
            setting=_stanley_settings(c34_stanley_vbeta=1.0),
        )
        controller.cte_v_sum = 100.0
        ego = EgoState(x=50.0, y=0.0, theta=0.0, velocity=0.0)
        cmd = controller.control(ego)
        assert controller.cte_v_sum == pytest.approx(0.0, abs=1e-9)
        # Without the floor, I-term would command large braking at rest.
        assert cmd.acceleration == pytest.approx(0.0, abs=1e-6)


class TestStanleyCteSlowDown:
    def test_large_cte_reduces_acceleration_while_fast(self):
        """Large lateral error must dump accel when above slow-down speed threshold."""
        trajectory = _straight_path(velocity=5.0)
        # Match target speed with zero velocity gains so baseline acc is ~0 on-path.
        setting = _stanley_settings(
            c34_stanley_slow_down_cte=0.5,
            c34_stanley_slow_down_vel_threshold=3.0,
        )
        on_path = StanleyController(tj=_straight_path(velocity=5.0), setting=setting)
        offset = StanleyController(tj=trajectory, setting=setting)
        ego_on = EgoState(x=50.0, y=0.0, theta=0.0, velocity=5.0)
        ego_off = EgoState(x=50.0, y=2.0, theta=0.0, velocity=5.0)
        acc_on = on_path.control(ego_on).acceleration
        acc_off = offset.control(ego_off).acceleration
        assert acc_on == pytest.approx(0.0, abs=0.5)
        assert acc_off < acc_on - 5.0
        assert acc_off <= setting.c32_ego_min_acceleration + 1e-6


class TestPurePursuitVelocityPidSafety:
    def test_emergency_brake_overrides_zero_gains(self):
        controller = PurePursuitController(tj=_straight_path(), setting=_pp_settings())
        ego = EgoState(x=20.0, y=0.0, theta=0.0, velocity=5.0)
        acc = controller.velocity_pid(ego, target_velocity=0.0)
        expected = (
            ControlSettings.c32_ego_min_acceleration
            * ControlSettings.c30_emergency_braking_factor
        )
        assert acc == pytest.approx(expected, abs=1e-6)

    def test_stopped_anti_windup_and_accel_floor(self):
        controller = PurePursuitController(
            tj=_straight_path(),
            setting=_pp_settings(c35_vbeta=1.0),
        )
        controller.cte_v_sum = 50.0
        ego = EgoState(x=20.0, y=0.0, theta=0.0, velocity=0.0)
        acc = controller.velocity_pid(ego, target_velocity=0.0)
        assert controller.cte_v_sum == pytest.approx(0.0, abs=1e-9)
        assert acc == pytest.approx(0.0, abs=1e-6)
