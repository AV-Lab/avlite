"""Smoke tests for executor_factory stack assembly (avlite.c60_apps.c62_factory).

Tests verify:
- executor_factory wires core components with plugins disabled.
- Returned executer is a SyncExecuter with world bridge and controller attached.
"""

import pytest

from avlite.c10_perception.c12_perception_strategy import PerceptionPipeline
from avlite.c10_perception.c14_mapping_strategy import MapReader
from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c60_apps.c62_factory import executor_factory
from avlite.c40_execution.c44_sync_executer import SyncExecuter
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c30_control.c34_stanley import StanleyController
from avlite.c50_common.c51_capabilities import StackCapability


def test_executor_factory_builds_sync_executer(minimal_corridor_map_path):
    ExecutionSettings.c40_map = str(minimal_corridor_map_path.resolve())
    ExecutionSettings.c40_mapping = MapReader.__name__
    ExecutionSettings.c40_executer_type = SyncExecuter.__name__
    ExecutionSettings.c40_bridge = "BasicSim"
    ExecutionSettings.c40_perception = ""
    ExecutionSettings.c40_localization = ""
    ExecutionSettings.c40_global_planner = "GlobalCenterlineRacePlanner"
    ExecutionSettings.c40_local_planner = "GreedyLatticePlanner"
    ExecutionSettings.c40_controller = StanleyController.__name__

    executer = executor_factory(load_plugins=False)

    assert isinstance(executer, SyncExecuter)
    assert executer.world is not None
    assert executer.controller is not None
    assert executer.local_planner is not None
    assert executer.global_planner is not None
    assert executer.pm is not None
    assert executer.mapping is not None
    assert isinstance(executer.mapping, MapReader)
    assert executer.world.map is not None
    assert executer.global_planner.map is executer.world.map
    assert executer.mapping.map is executer.world.map
    assert StackCapability.MAP_RACE_TRACK in executer.available_stack_capabilities()
    assert StackCapability.MAP_RACE_TRACK not in executer.world.stack_capabilities
    assert StackCapability.MAP_HD not in executer.available_stack_capabilities()


def test_executor_factory_allows_empty_modules(minimal_corridor_map_path):
    ExecutionSettings.c40_map = str(minimal_corridor_map_path.resolve())
    ExecutionSettings.c40_mapping = ""

    executer = executor_factory(
        load_plugins=False,
        executer_type=SyncExecuter.__name__,
        bridge="BasicSim",
        perception_strategy_name="",
        localization_strategy_name="",
        mapping_strategy_name="",
        global_planner_strategy_name="",
        local_planner_strategy_name="",
        controller_strategy_name="",
    )

    assert isinstance(executer, SyncExecuter)
    assert executer.perception is None
    assert executer.localization is None
    assert executer.mapping is None
    assert executer.global_planner is None
    assert executer.local_planner is None
    assert executer.controller is None
    executer.step(call_replan=True, call_control=True, call_perceive=True, call_localize=True)
    executer.reset()


def test_executor_factory_raises_for_unknown_local_planner(minimal_corridor_map_path):
    ExecutionSettings.c40_map = str(minimal_corridor_map_path.resolve())
    ExecutionSettings.c40_mapping = MapReader.__name__

    with pytest.raises(ValueError, match="local planner 'NonExistentLocalPlanner'"):
        executor_factory(
            load_plugins=False,
            executer_type=SyncExecuter.__name__,
            bridge="BasicSim",
            perception_strategy_name="",
            localization_strategy_name="",
            mapping_strategy_name=MapReader.__name__,
            global_planner_strategy_name="GlobalCenterlineRacePlanner",
            local_planner_strategy_name="NonExistentLocalPlanner",
            controller_strategy_name=StanleyController.__name__,
        )


def test_executor_factory_raises_for_missing_global_plan(minimal_corridor_map_path):
    ExecutionSettings.c40_map = str(minimal_corridor_map_path.resolve())
    ExecutionSettings.c40_mapping = MapReader.__name__

    with pytest.raises(Exception):
        executor_factory(
            load_plugins=False,
            executer_type=SyncExecuter.__name__,
            bridge="BasicSim",
            perception_strategy_name="",
            localization_strategy_name="",
            mapping_strategy_name=MapReader.__name__,
            global_planner_strategy_name="GlobalCenterlineRacePlanner",
            local_planner_strategy_name="GreedyLatticePlanner",
            controller_strategy_name=StanleyController.__name__,
            default_global_trajectory_file="nonexistent/global_plan.json",
        )


def test_executor_factory_raises_for_missing_pipeline_detection(minimal_corridor_map_path):
    ExecutionSettings.c40_map = str(minimal_corridor_map_path.resolve())
    ExecutionSettings.c40_mapping = MapReader.__name__
    PerceptionSettings.c12_detection_strategy = "NonExistentDetector"
    PerceptionSettings.c12_tracking_strategy = ""
    PerceptionSettings.c12_prediction_strategy = ""

    with pytest.raises(ValueError, match="PerceptionPipeline sub-strategies: detection 'NonExistentDetector'"):
        executor_factory(
            load_plugins=False,
            executer_type=SyncExecuter.__name__,
            bridge="BasicSim",
            perception_strategy_name=PerceptionPipeline.__name__,
            localization_strategy_name="",
            mapping_strategy_name=MapReader.__name__,
            global_planner_strategy_name="GlobalCenterlineRacePlanner",
            local_planner_strategy_name="GreedyLatticePlanner",
            controller_strategy_name=StanleyController.__name__,
        )


def test_executor_factory_raises_for_multiple_missing_pipeline_substrategies(minimal_corridor_map_path):
    ExecutionSettings.c40_map = str(minimal_corridor_map_path.resolve())
    ExecutionSettings.c40_mapping = MapReader.__name__
    PerceptionSettings.c12_detection_strategy = "BadDetector"
    PerceptionSettings.c12_tracking_strategy = "BadTracker"
    PerceptionSettings.c12_prediction_strategy = ""

    with pytest.raises(ValueError, match="PerceptionPipeline sub-strategies:") as exc_info:
        executor_factory(
            load_plugins=False,
            executer_type=SyncExecuter.__name__,
            bridge="BasicSim",
            perception_strategy_name=PerceptionPipeline.__name__,
            localization_strategy_name="",
            mapping_strategy_name=MapReader.__name__,
            global_planner_strategy_name="GlobalCenterlineRacePlanner",
            local_planner_strategy_name="GreedyLatticePlanner",
            controller_strategy_name=StanleyController.__name__,
        )

    message = str(exc_info.value)
    assert "detection 'BadDetector'" in message
    assert "tracking 'BadTracker'" in message


def test_executor_factory_empty_global_plan(minimal_corridor_map_path):
    ExecutionSettings.c40_map = str(minimal_corridor_map_path.resolve())
    ExecutionSettings.c40_mapping = MapReader.__name__

    executer = executor_factory(
        load_plugins=False,
        executer_type=SyncExecuter.__name__,
        bridge="BasicSim",
        perception_strategy_name="",
        localization_strategy_name="",
        mapping_strategy_name=MapReader.__name__,
        global_planner_strategy_name="GlobalCenterlineRacePlanner",
        local_planner_strategy_name="",
        controller_strategy_name="",
        default_global_trajectory_file="",
    )

    assert isinstance(executer, SyncExecuter)
    assert executer.global_planner is not None
    assert executer.ego_state.x == 0.0
    assert executer.ego_state.y == 0.0


def test_executor_factory_uses_c40_start_pose_for_world_and_stack_ego(minimal_corridor_map_path):
    ExecutionSettings.c40_map = str(minimal_corridor_map_path.resolve())
    ExecutionSettings.c40_mapping = MapReader.__name__
    ExecutionSettings.c40_start_pose = [12.5, -3.25, 0.75]

    executer = executor_factory(
        load_plugins=False,
        executer_type=SyncExecuter.__name__,
        bridge="BasicSim",
        perception_strategy_name="",
        localization_strategy_name="",
        mapping_strategy_name=MapReader.__name__,
        global_planner_strategy_name="GlobalCenterlineRacePlanner",
        local_planner_strategy_name="",
        controller_strategy_name="",
        default_global_trajectory_file="",
    )

    assert executer.ego_state.x == pytest.approx(12.5)
    assert executer.ego_state.y == pytest.approx(-3.25)
    assert executer.ego_state.theta == pytest.approx(0.75)
    assert executer.pm.ego_vehicle.x == pytest.approx(12.5)
    assert executer.pm.ego_vehicle.y == pytest.approx(-3.25)
    assert executer.pm.ego_vehicle.theta == pytest.approx(0.75)
    # World/stack ego must remain distinct objects sharing the start pose.
    assert executer.ego_state is not executer.pm.ego_vehicle


def test_executor_factory_empty_start_pose_falls_back_to_plan_start(minimal_corridor_map_path):
    ExecutionSettings.c40_map = str(minimal_corridor_map_path.resolve())
    ExecutionSettings.c40_mapping = MapReader.__name__
    ExecutionSettings.c40_start_pose = []

    executer = executor_factory(
        load_plugins=False,
        executer_type=SyncExecuter.__name__,
        bridge="BasicSim",
        perception_strategy_name="",
        localization_strategy_name="",
        mapping_strategy_name=MapReader.__name__,
        global_planner_strategy_name="GlobalCenterlineRacePlanner",
        local_planner_strategy_name="",
        controller_strategy_name="",
        default_global_trajectory_file="",
    )

    assert executer.ego_state.x == 0.0
    assert executer.ego_state.y == 0.0


def test_executor_factory_raises_for_map_reader_without_map():
    ExecutionSettings.c40_map = ""

    with pytest.raises(ValueError, match="requires a map file"):
        executor_factory(
            load_plugins=False,
            executer_type=SyncExecuter.__name__,
            bridge="BasicSim",
            perception_strategy_name="",
            localization_strategy_name="",
            mapping_strategy_name=MapReader.__name__,
            global_planner_strategy_name="",
            local_planner_strategy_name="",
            controller_strategy_name="",
            default_global_trajectory_file="",
            map_file="",
        )


def test_executor_factory_raises_for_race_planner_without_map():
    with pytest.raises(ValueError, match="requires a RaceMap"):
        executor_factory(
            load_plugins=False,
            executer_type=SyncExecuter.__name__,
            bridge="BasicSim",
            perception_strategy_name="",
            localization_strategy_name="",
            mapping_strategy_name="",
            global_planner_strategy_name="GlobalCenterlineRacePlanner",
            local_planner_strategy_name="",
            controller_strategy_name="",
            default_global_trajectory_file="",
            map_file="",
        )


def test_executor_factory_raises_for_race_planner_without_mapping(minimal_corridor_map_path):
    """Map file alone is not enough: MapReader must provide MAP_RACE_TRACK."""
    with pytest.raises(ValueError, match="stack_requirements not satisfied"):
        executor_factory(
            load_plugins=False,
            executer_type=SyncExecuter.__name__,
            bridge="BasicSim",
            perception_strategy_name="",
            localization_strategy_name="",
            mapping_strategy_name="",
            global_planner_strategy_name="GlobalCenterlineRacePlanner",
            local_planner_strategy_name="",
            controller_strategy_name="",
            default_global_trajectory_file="",
            map_file=str(minimal_corridor_map_path.resolve()),
        )
