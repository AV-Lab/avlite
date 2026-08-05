"""Regression tests for ConstantVelocityPrediction (default predictor)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from avlite.c10_perception.c11_perception_model import AgentState, EgoState, PerceptionModel
from avlite.c10_perception.c15_perception_algs import ConstantVelocityPrediction
from avlite.c10_perception.c19_settings import PerceptionSettings


def test_constant_velocity_predict_requires_perception_model():
    with pytest.raises(ValueError, match="perception_model is required"):
        ConstantVelocityPrediction().predict(perception_model=None)


def test_constant_velocity_predict_empty_agents_clears_trajectories():
    pm = PerceptionModel(ego_vehicle=EgoState(), agent_vehicles=[])
    out = ConstantVelocityPrediction().predict(pm)
    assert out.prediction is not None
    assert out.prediction.trajectories == {}
    assert out.prediction.predict_delta_t == PerceptionSettings.c11_predict_delta_t


def test_constant_velocity_predict_extrapolates_along_heading():
    dt = PerceptionSettings.c11_predict_delta_t
    horizon = PerceptionSettings.c15_prediction_horizon
    n_steps = max(1, int(round(horizon / dt)))

    a_east = AgentState(x=0.0, y=0.0, theta=0.0, velocity=10.0, agent_id=7)
    a_north = AgentState(x=1.0, y=2.0, theta=math.pi / 2, velocity=2.0, agent_id=8)
    pm = PerceptionModel(ego_vehicle=EgoState(), agent_vehicles=[a_east, a_north])

    out = ConstantVelocityPrediction().predict(pm)
    trajs = out.prediction.trajectories
    assert sorted(trajs) == [7, 8]
    assert out.prediction.predict_delta_t == dt
    assert trajs[7].shape == (n_steps, 2)

    expected_east = np.array(
        [[10.0 * (t + 1) * dt, 0.0] for t in range(n_steps)], dtype=float
    )
    expected_north = np.array(
        [[1.0, 2.0 + 2.0 * (t + 1) * dt] for t in range(n_steps)], dtype=float
    )
    np.testing.assert_allclose(trajs[7], expected_east, atol=1e-9)
    np.testing.assert_allclose(trajs[8], expected_north, atol=1e-9)
