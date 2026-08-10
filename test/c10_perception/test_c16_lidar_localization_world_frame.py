"""LidarLocalization must track motion when fed AVLite world-frame LiDAR via the executer."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pytest

from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c10_perception.c16_localization_algs import LidarLocalization
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c30_control.c31_control_model import ControlCommand
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c40_execution.c41_world_bridge import WorldBridge
from avlite.c40_execution.c42_execution_strategy import ExecutionStrategy
from avlite.c40_execution.c44_sync_executer import SyncExecuter
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c50_common.c51_capabilities import StackCapability, WorldCapability
from avlite.c50_common.c52_world_sensor_datatypes import (
    SensorFrame,
    lidar_2d_to_4,
    world_lidar_to_ego_frame,
)


def _static_room_world_lidar(ego_x: float, ego_y: float, ego_theta: float, n: int = 72) -> np.ndarray:
    """Raycast against a fixed square room (world-frame hits), like BasicSim."""
    walls = [
        (np.array([10.0, -20.0]), np.array([10.0, 20.0])),
        (np.array([-10.0, -20.0]), np.array([-10.0, 20.0])),
        (np.array([-20.0, 10.0]), np.array([20.0, 10.0])),
        (np.array([-20.0, -10.0]), np.array([20.0, -10.0])),
    ]
    hits = []
    angles = ego_theta + np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    origin = np.array([ego_x, ego_y], dtype=float)
    for a in angles:
        d = np.array([math.cos(a), math.sin(a)])
        best_t = None
        for p, q in walls:
            e = q - p
            denom = d[0] * e[1] - d[1] * e[0]
            if abs(denom) < 1e-12:
                continue
            diff = p - origin
            t = (diff[0] * e[1] - diff[1] * e[0]) / denom
            u = (diff[0] * d[1] - diff[1] * d[0]) / denom
            if t > 0.01 and t <= 30.0 and 0.0 <= u <= 1.0:
                if best_t is None or t < best_t:
                    best_t = t
        if best_t is not None:
            hits.append(origin + best_t * d)
    return np.asarray(hits, dtype=float) if hits else np.empty((0, 2))


@dataclass
class _LidarPlantWorld(WorldBridge):
    """Plant with world-frame LiDAR; GT LOCALIZATION capability present but filterable."""

    ego_state: EgoState = field(default_factory=lambda: EgoState(x=0.0, y=0.0, theta=0.0))
    perception_model: Optional[PerceptionModel] = None
    world_capabilities = frozenset({WorldCapability.LIDAR_2D})
    stack_capabilities = frozenset({StackCapability.LOCALIZATION})

    def control_ego_state(self, cmd: ControlCommand, dt: float = 0.01):
        # Integrate a simple forward step so plant pose advances under control.
        v = max(float(self.ego_state.velocity), 0.0) + float(cmd.acceleration) * dt
        self.ego_state.velocity = max(0.0, v)
        self.ego_state.x += self.ego_state.velocity * math.cos(self.ego_state.theta) * dt
        self.ego_state.y += self.ego_state.velocity * math.sin(self.ego_state.theta) * dt
        self.ego_state.theta += float(cmd.steer) * dt

    def get_lidar_data(self, agent_id: int = 0):
        pts = _static_room_world_lidar(self.ego_state.x, self.ego_state.y, self.ego_state.theta)
        return lidar_2d_to_4(pts)

    def get_ground_truth_perception_model(self):
        if self.perception_model is None:
            self.perception_model = PerceptionModel(ego_vehicle=self.ego_state)
        return self.perception_model


class _StubPlanner(LocalPlanningStrategy):
    world_requirements = frozenset()
    stack_requirements = frozenset()
    stack_capabilities = frozenset({StackCapability.LOCAL_PLAN})

    def __init__(self):
        self.lap = 0

    def replan(self, perception_model=None, sensors=None):
        pass

    def step(self, ego_state):
        pass

    def get_local_plan(self):
        return None

    def reset(self):
        pass

    def __init_subclass__(cls, **kwargs):
        pass


class _CruiseController(ControlStrategy, abstract=True):
    stack_requirements = frozenset({StackCapability.LOCALIZATION})

    def control(self, ego, plan=None, control_dt=None, perception_model=None, sensors=None):
        # Drive forward at ~2 m/s.
        acc = 1.0 if ego.velocity < 2.0 else 0.0
        return ControlCommand(acceleration=acc, steer=0.0)

    def reset(self):
        pass


@pytest.fixture(autouse=True)
def _restore_stack_cap_filter():
    prev = ExecutionSettings.c41_world_stack_capabilities
    yield
    ExecutionSettings.c41_world_stack_capabilities = prev


def test_world_lidar_to_ego_frame_roundtrip_at_identity():
    pts = np.array([[10.0, 0.0, 0.0, 1.0], [0.0, 5.0, 0.0, 1.0]])
    out = world_lidar_to_ego_frame(pts, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(out[:, :2], pts[:, :2])
    np.testing.assert_allclose(out[:, 2:], pts[:, 2:])


def test_world_lidar_to_ego_frame_translates_and_rotates():
    # World hit at (3, 0); ego at (1, 0) heading +90° → body (0, -2) (x fwd, y left).
    pts = np.array([[3.0, 0.0]])
    out = world_lidar_to_ego_frame(pts, 1.0, 0.0, math.pi / 2)
    np.testing.assert_allclose(out[0], [0.0, -2.0], atol=1e-9)


def test_direct_world_frame_localize_freezes_near_seed():
    """Sanity: feeding world-frame hits straight into localize (no adapter) fails."""
    ego = EgoState(x=0.0, y=0.0, theta=0.0)
    pm = PerceptionModel(ego_vehicle=ego)
    loc = LidarLocalization(pm)
    for x in (0.0, 2.0, 4.0):
        scan = _static_room_world_lidar(x, 0.0, 0.0)
        loc.localize(pm, SensorFrame(lidar=lidar_2d_to_4(scan)))
    assert abs(ego.x) < 0.5  # stuck near origin, not at x≈4


@pytest.mark.parametrize("start_x", [-3.0, 0.0, 3.0])
def test_executer_adapts_world_lidar_so_icp_tracks_plant(start_x: float):
    """GT LOCALIZATION off + LidarLocalization: stack ego must follow plant motion.

    Includes non-origin seeds (profile ``c40_start_pose``) — the reference map
    must be stored in world frame, not raw body points. Drive stays inside the
    ±10 m room so ICP geometry remains well constrained.
    """
    ExecutionSettings.c41_world_stack_capabilities = []  # disable GT LOCALIZATION
    world_ego = EgoState(x=start_x, y=0.0, theta=0.0, velocity=0.0)
    stack_ego = EgoState(x=start_x, y=0.0, theta=0.0, velocity=0.0)
    world = _LidarPlantWorld(ego_state=world_ego)
    pm = PerceptionModel(ego_vehicle=stack_ego)
    loc = LidarLocalization(pm)
    exec_ = SyncExecuter(
        perception_model=pm,
        perception=None,
        global_planner=None,
        local_planner=_StubPlanner(),
        controller=_CruiseController(),
        world=world,
        localization=loc,
        control_dt=0.0,
        localization_dt=0.0,
        replan_dt=99.0,
        perception_dt=99.0,
    )

    for _ in range(50):
        exec_.step(
            sim_dt=0.05,
            control_dt=0.05,
            localization_dt=0.0,
            replan_dt=99.0,
            perception_dt=99.0,
            call_replan=False,
            call_perceive=False,
            call_localize=True,
            call_control=True,
            pace_control=False,
            pace_sim=True,
        )

    # Plant has moved several metres; stack estimate must track (not freeze at seed).
    assert world_ego.x > start_x + 2.0
    assert stack_ego.x == pytest.approx(world_ego.x, abs=0.35)
    assert stack_ego.y == pytest.approx(world_ego.y, abs=0.35)
    # Without world-frame map seeding, non-origin starts lagged by metres.
    assert abs(stack_ego.x - start_x) > 2.0


def test_reference_map_seeded_in_world_frame_at_non_origin():
    """First body scan must be lifted into world before becoming the ICP map."""
    ego = EgoState(x=3.0, y=0.0, theta=0.0)
    pm = PerceptionModel(ego_vehicle=ego)
    loc = LidarLocalization(pm)
    # Body-frame hit 7 m ahead → world x=10 when ego is at x=3.
    body = np.array([[7.0, 0.0], [7.0, 1.0], [7.0, -1.0], [5.0, 0.0]])
    loc.localize(pm, SensorFrame(lidar=lidar_2d_to_4(body)))
    assert loc._map is not None
    np.testing.assert_allclose(loc._map[0], [10.0, 0.0], atol=1e-9)


def test_sensors_for_localization_uses_plant_not_stack_ego():
    """Adapter must convert with plant pose even when stack ego has drifted."""
    ExecutionSettings.c41_world_stack_capabilities = []
    world_ego = EgoState(x=2.0, y=0.0, theta=0.0)
    stack_ego = EgoState(x=0.0, y=0.0, theta=0.0)  # stale estimate
    world = _LidarPlantWorld(ego_state=world_ego)
    pm = PerceptionModel(ego_vehicle=stack_ego)
    exec_ = SyncExecuter(
        perception_model=pm,
        perception=None,
        global_planner=None,
        local_planner=_StubPlanner(),
        controller=_CruiseController(),
        world=world,
        localization=LidarLocalization(pm),
    )
    sensors = world.get_sensor_frame()
    adapted = ExecutionStrategy._sensors_for_localization(exec_, sensors)
    assert adapted is not None and adapted.lidar is not None
    # A world hit near (10, 0) with plant at x=2 → body x≈8.
    body = np.asarray(adapted.lidar)
    assert body[:, 0].max() > 7.0
    # Stack ego was NOT used (would leave body x≈10).
    assert body[:, 0].max() < 9.5
