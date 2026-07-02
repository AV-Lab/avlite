"""Smoke tests for executor_factory stack assembly (avlite.c40_execution.c43_factory).

Tests verify:
- executor_factory wires core components with plugins disabled.
- Returned executer is a SyncExecuter with world bridge and controller attached.
"""

import pytest

from avlite.c10_perception.c12_perception_strategy import PerceptionPipeline
from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c40_execution.c43_factory import executor_factory
from avlite.c40_execution.c44_sync_executer import SyncExecuter
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c30_control.c34_stanley import StanleyController


def test_executor_factory_builds_sync_executer(minimal_corridor_map_path):
    ExecutionSettings.c43_race_boundary_map = str(minimal_corridor_map_path.resolve())
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


def test_executor_factory_raises_for_unknown_local_planner(minimal_corridor_map_path):
    ExecutionSettings.c43_race_boundary_map = str(minimal_corridor_map_path.resolve())

    with pytest.raises(ValueError, match="local planner 'NonExistentLocalPlanner'"):
        executor_factory(
            load_plugins=False,
            executer_type=SyncExecuter.__name__,
            bridge="BasicSim",
            perception_strategy_name="",
            localization_strategy_name="",
            global_planner_strategy_name="GlobalCenterlineRacePlanner",
            local_planner_strategy_name="NonExistentLocalPlanner",
            controller_strategy_name=StanleyController.__name__,
        )


def test_executor_factory_raises_for_missing_global_plan(minimal_corridor_map_path):
    ExecutionSettings.c43_race_boundary_map = str(minimal_corridor_map_path.resolve())

    with pytest.raises(Exception):
        executor_factory(
            load_plugins=False,
            executer_type=SyncExecuter.__name__,
            bridge="BasicSim",
            perception_strategy_name="",
            localization_strategy_name="",
            global_planner_strategy_name="GlobalCenterlineRacePlanner",
            local_planner_strategy_name="GreedyLatticePlanner",
            controller_strategy_name=StanleyController.__name__,
            default_global_trajectory_file="nonexistent/global_plan.json",
        )


def test_executor_factory_raises_for_missing_pipeline_detection(minimal_corridor_map_path):
    ExecutionSettings.c43_race_boundary_map = str(minimal_corridor_map_path.resolve())
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
            global_planner_strategy_name="GlobalCenterlineRacePlanner",
            local_planner_strategy_name="GreedyLatticePlanner",
            controller_strategy_name=StanleyController.__name__,
        )


def test_executor_factory_raises_for_multiple_missing_pipeline_substrategies(minimal_corridor_map_path):
    ExecutionSettings.c43_race_boundary_map = str(minimal_corridor_map_path.resolve())
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
            global_planner_strategy_name="GlobalCenterlineRacePlanner",
            local_planner_strategy_name="GreedyLatticePlanner",
            controller_strategy_name=StanleyController.__name__,
        )

    message = str(exc_info.value)
    assert "detection 'BadDetector'" in message
    assert "tracking 'BadTracker'" in message
