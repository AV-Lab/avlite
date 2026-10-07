import numpy as np
import pytest

from avlite.c10_perception.c11_perception_model import (
    AgentState,
    GMM,
    GP,
    MultiTrajectory,
    OccupancyFlow,
    PerceptionModel,
    SingleTrajectory,
)
from avlite.c10_perception.c12_perception_strategy import PredictionStrategy
from avlite.c10_perception.c15_perception_algs import (
    ConstantVelocityGMM,
    ConstantVelocityGP,
    ConstantVelocityMultiTrajectory,
    ConstantVelocityOccupancyFlow,
    ConstantVelocityPrediction,
)
from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c20_planning.c29_settings import PlanningSettings
from avlite.c50_common.c51_capabilities import StackCapability


def _agent() -> AgentState:
    return AgentState(x=10.0, y=2.0, theta=0.3, velocity=4.0, agent_id=1)


def _predict(cls, agent: AgentState | None = None) -> PerceptionModel:
    agents = [] if agent is None else [agent]
    return cls().predict(PerceptionModel(agent_vehicles=agents))


def test_strategies_register_and_advertise_their_payload():
    assert PredictionStrategy.registry["ConstantVelocityGP"] is ConstantVelocityGP
    assert PredictionStrategy.registry["ConstantVelocityGMM"] is ConstantVelocityGMM
    assert PredictionStrategy.registry["ConstantVelocityMultiTrajectory"] is ConstantVelocityMultiTrajectory
    assert PredictionStrategy.registry["ConstantVelocityOccupancyFlow"] is ConstantVelocityOccupancyFlow
    assert ConstantVelocityGP.stack_capabilities == frozenset({StackCapability.PREDICTION_GP})
    assert ConstantVelocityGMM.stack_capabilities == frozenset({StackCapability.PREDICTION_GMM})
    assert ConstantVelocityOccupancyFlow.stack_capabilities == frozenset({
        StackCapability.PREDICTION_OCCUPANCY,
    })
    assert ConstantVelocityMultiTrajectory.stack_capabilities == frozenset({
        StackCapability.PREDICTION_MULTI_TRAJECTORY,
    })


@pytest.mark.parametrize("cls", [
    ConstantVelocityPrediction,
    ConstantVelocityGP,
    ConstantVelocityGMM,
    ConstantVelocityMultiTrajectory,
    ConstantVelocityOccupancyFlow,
])
def test_prediction_requires_a_perception_model(cls):
    with pytest.raises(ValueError):
        cls().predict(None)


def test_gp_mean_matches_constant_velocity_and_covariance_grows():
    agent = _agent()
    cv = _predict(ConstantVelocityPrediction, agent).prediction
    gp = _predict(ConstantVelocityGP, agent).prediction
    assert isinstance(cv, SingleTrajectory)
    assert isinstance(gp, GP)
    mean = gp.means[agent.agent_id]
    np.testing.assert_allclose(mean, cv.trajectories[agent.agent_id])
    n = mean.shape[0]
    cov = gp.covariance[agent.agent_id]
    assert cov.shape == (2 * n, 2 * n)
    dt = PerceptionSettings.c11_predict_delta_t
    c, s = np.cos(agent.theta), np.sin(agent.theta)
    rotation = np.array([[c, -s], [s, c]])
    expected = rotation @ np.diag([(2.0 * dt) ** 2, (0.8 * dt) ** 2]) @ rotation.T
    np.testing.assert_allclose(cov[:2, :2], expected)
    assert abs(cov[0, 1]) > 0.0
    vals, vecs = np.linalg.eigh(cov[:2, :2])
    assert vals[-1] > vals[0]
    assert abs(np.dot(vecs[:, -1], np.array([c, s]))) > 0.99
    assert np.trace(cov[-2:, -2:]) > np.trace(cov[:2, :2])


def test_empty_agent_list_writes_an_empty_payload():
    assert isinstance(_predict(ConstantVelocityGP).prediction, GP)
    assert _predict(ConstantVelocityGP).prediction.means == {}
    gmm = _predict(ConstantVelocityGMM).prediction
    assert isinstance(gmm, GMM)
    assert gmm.trajectories == {} and gmm.weights == {} and gmm.covariances == {}
    multi = _predict(ConstantVelocityMultiTrajectory).prediction
    assert isinstance(multi, MultiTrajectory)
    assert multi.trajectories == {} and multi.weights == {}
    flow = _predict(ConstantVelocityOccupancyFlow).prediction
    assert isinstance(flow, OccupancyFlow)
    assert flow.occupancy_flow == {}


def test_mode_counts_follow_settings_and_middle_mode_is_straight():
    agent = _agent()
    old_gmm = PerceptionSettings.c15_gmm_n_modes
    old_multi = PerceptionSettings.c15_multitrajectory_n_modes
    PerceptionSettings.c15_gmm_n_modes = 3
    PerceptionSettings.c15_multitrajectory_n_modes = 5
    try:
        cv = _predict(ConstantVelocityPrediction, agent).prediction
        gmm = _predict(ConstantVelocityGMM, agent).prediction
        multi = _predict(ConstantVelocityMultiTrajectory, agent).prediction
        assert isinstance(cv, SingleTrajectory)
        assert isinstance(gmm, GMM)
        assert isinstance(multi, MultiTrajectory)
        straight = cv.trajectories[agent.agent_id]
        n_steps = straight.shape[0]

        gmm_modes = gmm.trajectories[agent.agent_id]
        assert gmm_modes.shape == (3, n_steps, 2)
        assert gmm.covariances[agent.agent_id].shape == (3, n_steps, 2, 2)
        np.testing.assert_allclose(gmm.weights[agent.agent_id].sum(), 1.0)
        np.testing.assert_allclose(gmm_modes[1], straight)
        assert gmm.weights[agent.agent_id][1] > gmm.weights[agent.agent_id][0]
        assert gmm.weights[agent.agent_id].min() > 0.5 * gmm.weights[agent.agent_id].max()
        assert not np.allclose(gmm_modes[0], straight)
        middle = gmm.covariances[agent.agent_id][1, 0]
        gp = _predict(ConstantVelocityGP, agent).prediction
        np.testing.assert_allclose(middle, gp.covariance[agent.agent_id][:2, :2])
        assert not np.allclose(gmm.covariances[agent.agent_id][0, 0], middle)

        multi_modes = multi.trajectories[agent.agent_id]
        assert multi_modes.shape == (5, n_steps, 2)
        np.testing.assert_allclose(multi.weights[agent.agent_id].sum(), 1.0)
        np.testing.assert_allclose(multi_modes[2], straight)
        assert multi.weights[agent.agent_id][2] > multi.weights[agent.agent_id][0]
        assert multi.weights[agent.agent_id].min() > 0.5 * multi.weights[agent.agent_id].max()
    finally:
        PerceptionSettings.c15_gmm_n_modes = old_gmm
        PerceptionSettings.c15_multitrajectory_n_modes = old_multi


def test_mode_count_below_one_is_a_single_straight_path():
    agent = _agent()
    old_gmm = PerceptionSettings.c15_gmm_n_modes
    old_multi = PerceptionSettings.c15_multitrajectory_n_modes
    PerceptionSettings.c15_gmm_n_modes = 0
    PerceptionSettings.c15_multitrajectory_n_modes = 0
    try:
        cv = _predict(ConstantVelocityPrediction, agent).prediction
        gmm = _predict(ConstantVelocityGMM, agent).prediction
        multi = _predict(ConstantVelocityMultiTrajectory, agent).prediction
        straight = cv.trajectories[agent.agent_id]
        np.testing.assert_allclose(gmm.trajectories[agent.agent_id][0], straight)
        np.testing.assert_allclose(gmm.weights[agent.agent_id], [1.0])
        np.testing.assert_allclose(multi.trajectories[agent.agent_id][0], straight)
        np.testing.assert_allclose(multi.weights[agent.agent_id], [1.0])
        assert gmm.covariances[agent.agent_id].shape[0] == 1
    finally:
        PerceptionSettings.c15_gmm_n_modes = old_gmm
        PerceptionSettings.c15_multitrajectory_n_modes = old_multi


def _cell_center(pred: OccupancyFlow, row: int, col: int) -> np.ndarray:
    return np.array([
        pred.origin_x + (col + 0.5) * pred.resolution,
        pred.origin_y + (row + 0.5) * pred.resolution,
    ])


def _axis_variance(grid: np.ndarray, pred: OccupancyFlow, mean: np.ndarray, axis: np.ndarray) -> float:
    rows, cols = np.indices(grid.shape)
    centers = np.stack((
        pred.origin_x + (cols + 0.5) * pred.resolution,
        pred.origin_y + (rows + 0.5) * pred.resolution,
    ), axis=-1)
    offset = centers - mean
    projection = offset @ axis
    return float((grid * projection ** 2).sum())


def test_occupancy_flow_is_the_gp_gaussian_on_a_grid():
    agent = _agent()
    gp = _predict(ConstantVelocityGP, agent).prediction
    flow = _predict(ConstantVelocityOccupancyFlow, agent).prediction
    assert isinstance(gp, GP)
    assert isinstance(flow, OccupancyFlow)
    grids = flow.occupancy_flow[agent.agent_id]
    mean = gp.means[agent.agent_id]
    assert len(grids) == len(mean)
    margin = float(PlanningSettings.c20_obstacle_inflation_margin)
    half_along = agent.length / 2.0 + margin
    half_cross = agent.width / 2.0 + margin
    heading = np.array([np.cos(agent.theta), np.sin(agent.theta)])
    across = np.array([-heading[1], heading[0]])
    first = grids[0]
    assert first.sum() > 1.0
    assert first[0, 0] < 1e-6
    rows, cols = np.indices(first.shape)
    centers = np.stack((
        flow.origin_x + (cols + 0.5) * flow.resolution,
        flow.origin_y + (rows + 0.5) * flow.resolution,
    ), axis=-1)
    offset = centers - mean[0]
    off_along = offset @ heading
    off_cross = offset @ across
    dist = np.hypot(off_along, off_cross)
    covered = first > 0.5
    assert int(covered.sum()) > 1
    assert np.all(np.abs(off_along[covered]) <= half_along + flow.resolution)
    assert np.all(np.abs(off_cross[covered]) <= half_cross + flow.resolution)
    assert np.any(covered & (dist > 2.0))
    outside = dist > np.hypot(half_along, half_cross) + 3.0 * flow.resolution
    assert outside.any()
    assert first[outside].max() < 1e-3
    peak = np.unravel_index(int(np.argmax(first)), first.shape)
    peak_offset = _cell_center(flow, *peak) - mean[0]
    assert abs(float(peak_offset @ heading)) <= half_along + flow.resolution
    assert abs(float(peak_offset @ across)) <= half_cross + flow.resolution

    last = grids[-1]
    assert last.max() < first.max()
    assert _axis_variance(last, flow, mean[-1], heading) > _axis_variance(last, flow, mean[-1], across)
    peak = np.unravel_index(int(np.argmax(last)), last.shape)
    peak_offset = _cell_center(flow, *peak) - mean[-1]
    assert abs(float(peak_offset @ heading)) <= half_along + flow.resolution
    assert abs(float(peak_offset @ across)) <= half_cross + flow.resolution
