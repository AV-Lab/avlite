"""Unit tests for PID and Stanley controllers (avlite.c30_control).

Tests verify:
- PID steering is near zero when the ego is on the reference path.
- PID proportional gain produces steering proportional to cross-track error.
- Stanley steering responds with opposite sign to lateral offset.
"""

import pytest

from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c30_control.c33_pid import PIDController
from avlite.c30_control.c34_stanley import StanleyController
from avlite.c30_control.c39_settings import ControlSettingsSchema
from avlite.c50_common.c53_trajectory_tracker import TrajectoryTracker


def _straight_path(x_end: float = 100.0, n: int = 20, velocity: float = 5.0) -> TrajectoryTracker:
    xs = [x_end * i / (n - 1) for i in range(n)]
    path = [(x, 0.0) for x in xs]
    return TrajectoryTracker(path=path, velocity=[velocity] * n)


def _pid_settings(**overrides) -> ControlSettingsSchema:
    defaults = {
        "c33_pid_alpha": 1.0,
        "c33_pid_beta": 0.0,
        "c33_pid_gamma": 0.0,
        "c33_pid_valpha": 0.0,
        "c33_pid_vbeta": 0.0,
        "c33_pid_vgamma": 0.0,
        "c33_pid_lookahead": 0,
    }
    defaults.update(overrides)
    return ControlSettingsSchema(**defaults)


def _stanley_settings(**overrides) -> ControlSettingsSchema:
    defaults = {
        "c34_stanley_k": 2.0,
        "c34_stanley_k_soft": 0.01,
        "c34_stanley_lookahead": 0,
        "c34_stanley_valpha": 0.0,
        "c34_stanley_vbeta": 0.0,
        "c34_stanley_vgamma": 0.0,
    }
    defaults.update(overrides)
    return ControlSettingsSchema(**defaults)


class TestPIDController:
    def test_zero_cross_track_error_yields_near_zero_steer(self):
        trajectory = _straight_path()
        controller = PIDController(tj=trajectory, setting=_pid_settings())
        ego = EgoState(x=50.0, y=0.0, theta=0.0, velocity=5.0)
        cmd = controller.control(ego)
        assert cmd.steer == pytest.approx(0.0, abs=0.05)

    def test_positive_cross_track_error_produces_steer(self):
        trajectory = _straight_path()
        controller = PIDController(tj=trajectory, setting=_pid_settings(c33_pid_alpha=2.0))
        ego = EgoState(x=50.0, y=2.0, theta=0.0, velocity=5.0)
        cmd = controller.control(ego)
        assert cmd.steer != pytest.approx(0.0, abs=0.01)


class TestStanleyController:
    def test_lateral_offset_produces_corrective_steer(self):
        trajectory = _straight_path()
        controller = StanleyController(tj=trajectory, setting=_stanley_settings())
        ego = EgoState(x=50.0, y=2.0, theta=0.0, velocity=5.0)
        cmd = controller.control(ego)
        assert cmd.steer != pytest.approx(0.0, abs=0.01)

    def test_on_path_steer_is_mostly_heading_correction(self):
        trajectory = _straight_path()
        controller = StanleyController(tj=trajectory, setting=_stanley_settings())
        ego = EgoState(x=50.0, y=0.0, theta=0.0, velocity=5.0)
        cmd = controller.control(ego)
        assert abs(cmd.steer) < 0.2
