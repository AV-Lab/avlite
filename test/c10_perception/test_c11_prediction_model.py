import numpy as np

from avlite.c10_perception.c11_perception_model import (
    AgentState,
    AggregatedOccupancyFlow,
    OccupancyFlow,
    PerceptionModel,
    SingleTrajectory,
)
from avlite.c10_perception.c19_settings import PerceptionSettings


def test_single_trajectory_default_predict_delta_t():
    pred = SingleTrajectory()
    assert pred.predict_delta_t == PerceptionSettings.c11_predict_delta_t


def test_single_trajectory_lookup_by_agent_id():
    a1 = AgentState(x=0.0, y=0.0, agent_id=1)
    a2 = AgentState(x=5.0, y=0.0, agent_id=2)
    path1 = np.array([[1.0, 0.0], [2.0, 0.0]])
    path2 = np.array([[5.0, 1.0], [6.0, 1.0]])
    pm = PerceptionModel(
        agent_vehicles=[a1, a2],
        prediction=SingleTrajectory(trajectories={1: path1, 2: path2}),
    )

    assert isinstance(pm.prediction, SingleTrajectory)
    np.testing.assert_allclose(pm.prediction.trajectories[1], path1)
    np.testing.assert_allclose(pm.prediction.trajectories[2], path2)


def test_agent_id_dict_survives_agent_list_reorder():
    path1 = np.array([[1.0, 0.0], [2.0, 0.0]])
    path2 = np.array([[5.0, 1.0], [6.0, 1.0]])
    pm = PerceptionModel(
        prediction=SingleTrajectory(trajectories={1: path1, 2: path2}),
    )
    pm.agent_vehicles = [AgentState(agent_id=2), AgentState(agent_id=1)]
    assert pm.prediction is not None
    np.testing.assert_allclose(pm.prediction.trajectories[1], path1)
    np.testing.assert_allclose(pm.prediction.trajectories[2], path2)


def test_reset_clears_prediction():
    pm = PerceptionModel(
        prediction=SingleTrajectory(trajectories={1: np.zeros((2, 2))}),
    )
    pm.reset()
    assert pm.prediction is None


def test_occupancy_flow_default_window():
    pred = OccupancyFlow()
    assert pred.origin_x == 0.0
    assert pred.origin_y == 0.0
    assert pred.resolution == PerceptionSettings.c17_resolution


def test_aggregated_occupancy_flow_default_window():
    pred = AggregatedOccupancyFlow()
    assert pred.origin_x == 0.0
    assert pred.origin_y == 0.0
    assert pred.resolution == PerceptionSettings.c17_resolution


def test_aggregated_occupancy_flow_implied_extent():
    grid = np.zeros((2, 4), dtype=np.float32)
    pred = AggregatedOccupancyFlow(occupancy_flow=[grid])
    h, w = pred.occupancy_flow[0].shape
    res = pred.resolution
    extent = (
        pred.origin_x,
        pred.origin_x + w * res,
        pred.origin_y,
        pred.origin_y + h * res,
    )
    assert extent == (0.0, 4 * res, 0.0, 2 * res)
