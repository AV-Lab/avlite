"""Smoke tests for executor_factory stack assembly (avlite.c40_execution.c43_factory).

Tests verify:
- executor_factory wires core components with plugins disabled.
- Returned executer is a SyncExecuter with world bridge and controller attached.
"""

from avlite.c20_planning.c27_local_lattice_planners import GreedyLatticePlanner
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
    assert getattr(executer, "_factory_fallbacks", ()) == ()


def test_executor_factory_records_local_planner_fallback(minimal_corridor_map_path):
    ExecutionSettings.c43_race_boundary_map = str(minimal_corridor_map_path.resolve())
    ExecutionSettings.c40_executer_type = SyncExecuter.__name__
    ExecutionSettings.c40_bridge = "BasicSim"
    ExecutionSettings.c40_perception = ""
    ExecutionSettings.c40_localization = ""
    ExecutionSettings.c40_global_planner = "GlobalCenterlineRacePlanner"
    ExecutionSettings.c40_local_planner = "NonExistentLocalPlanner"
    ExecutionSettings.c40_controller = StanleyController.__name__

    executer = executor_factory(
        load_plugins=False,
        executer_type=SyncExecuter.__name__,
        bridge="BasicSim",
        perception_strategy_name="",
        localization_strategy_name="",
        global_planner_strategy_name="GlobalCenterlineRacePlanner",
        local_planner_strategy_name="NonExistentLocalPlanner",
        controller_strategy_name=StanleyController.__name__,
    )

    fallbacks = executer._factory_fallbacks
    assert len(fallbacks) == 1
    assert fallbacks[0].component == "local_planner"
    assert fallbacks[0].requested == "NonExistentLocalPlanner"
    assert fallbacks[0].used == GreedyLatticePlanner.__name__
    assert fallbacks[0].reason == "not recognized"
    assert "local planner" in fallbacks[0].message


def test_executor_factory_records_global_plan_fallback(minimal_corridor_map_path):
    ExecutionSettings.c43_race_boundary_map = str(minimal_corridor_map_path.resolve())

    executer = executor_factory(
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

    fallbacks = executer._factory_fallbacks
    assert len(fallbacks) == 1
    assert fallbacks[0].component == "global_plan"
    assert fallbacks[0].requested == "nonexistent/global_plan.json"
    assert fallbacks[0].used == "race centerline"
    assert "global plan" in fallbacks[0].message
