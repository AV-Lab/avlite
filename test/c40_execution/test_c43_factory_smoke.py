"""Smoke tests for executor_factory stack assembly (avlite.c40_execution.c43_factory).

Tests verify:
- executor_factory wires core components with plugins disabled.
- Returned executer is a SyncExecuter with world bridge and controller attached.
"""

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
