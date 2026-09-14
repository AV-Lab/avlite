"""Smoke tests for executor_factory stack assembly (avlite.c60_apps.c62_factory).

Tests verify:
- executor_factory wires core components with plugins disabled.
- Returned executer is a SyncExecuter with world bridge and controller attached.
"""

from pathlib import Path

import pytest

from avlite.c10_perception.c12_perception_strategy import PerceptionPipeline
from avlite.c10_perception.c14_mapping_strategy import MapReader
from avlite.c10_perception.c17_mapping_algs import OccupancyMapper
from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c40_execution.c46_basic_sim import BasicSim
from avlite.c60_apps.c62_factory import executor_factory
from avlite.c60_apps.c67_plugin_env import PluginEnv
from avlite.c60_apps.c69_settings import AppSettings
from avlite.c40_execution.c44_sync_executer import SyncExecuter
from avlite.c40_execution.c47_execution_tasks import MappingTask
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c30_control.c34_stanley import StanleyController
from avlite.c30_control.c36_keyboard import KeyboardController
from avlite.c50_common.c51_capabilities import StackCapability


def test_keyboard_controller_is_registered():
    assert KeyboardController.__name__ in ControlStrategy.registry


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
    assert not any(isinstance(t, MappingTask) for t in executer.task_runner.tasks)
    assert executer.world.map is not None
    assert executer.global_planner.map is executer.world.map
    assert executer.mapping.map is executer.world.map
    assert StackCapability.MAP_RACE_TRACK in executer.mapping.stack_capabilities
    assert StackCapability.MAP_RACE_TRACK in executer.available_stack_capabilities()
    assert StackCapability.MAP_RACE_TRACK not in executer.world.stack_capabilities
    assert StackCapability.MAP_HD not in executer.available_stack_capabilities()


def test_executor_factory_skips_apply_when_plugins_disabled(
    minimal_corridor_map_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(PluginEnv, "apply", lambda self: calls.append(self))
    ExecutionSettings.c40_map = str(minimal_corridor_map_path.resolve())
    executor_factory(
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
    assert calls == []


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
    assert not any(isinstance(t, MappingTask) for t in executer.task_runner.tasks)
    assert executer.global_planner is None
    assert executer.local_planner is None
    assert executer.controller is None
    executer.step(call_replan=True, call_control=True, call_perceive=True)
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


def test_executor_factory_raises_when_selected_plugin_needs_ros(
    tmp_path, monkeypatch, minimal_corridor_map_path
):
    plugin = tmp_path / "fake-ros"
    plugin.mkdir()
    (plugin / ".avlite-registry.yaml").write_text(
        "name: fake-ros\nrequire_ros: true\nmin_ros_version: humble\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(AppSettings, "c62_community_plugins", {"fake-ros": str(plugin)})
    monkeypatch.setattr("avlite.c60_apps.c62_factory.unregister_plugin_package", lambda _name: None)
    monkeypatch.setattr(BasicSim, "__module__", "avlite.plugins.fake_ros.bridge")
    monkeypatch.setattr(PluginEnv, "PREFIX", tmp_path / "no-ros")
    monkeypatch.delenv("ROS_DISTRO", raising=False)
    monkeypatch.delenv("AVLITE_ROS_DISTRO", raising=False)
    monkeypatch.setattr(AppSettings, "c60_ros_distro", "")
    ExecutionSettings.c40_map = str(minimal_corridor_map_path.resolve())
    ExecutionSettings.c40_mapping = MapReader.__name__

    with pytest.raises(PluginEnv.Error, match="needs ROS 2"):
        executor_factory(
            load_plugins=False,
            executer_type=SyncExecuter.__name__,
            bridge="BasicSim",
            perception_strategy_name="",
            localization_strategy_name="",
            mapping_strategy_name=MapReader.__name__,
            global_planner_strategy_name="GlobalCenterlineRacePlanner",
            local_planner_strategy_name="",
            controller_strategy_name="",
        )


def test_executor_factory_ignores_unselected_ros_plugin(
    tmp_path, monkeypatch, minimal_corridor_map_path
):
    plugin = tmp_path / "fake-ros"
    plugin.mkdir()
    (plugin / ".avlite-registry.yaml").write_text(
        "name: fake-ros\nrequire_ros: true\nmin_ros_version: humble\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(AppSettings, "c62_community_plugins", {"fake-ros": str(plugin)})
    monkeypatch.setattr(PluginEnv, "PREFIX", tmp_path / "no-ros")
    monkeypatch.delenv("ROS_DISTRO", raising=False)
    monkeypatch.delenv("AVLITE_ROS_DISTRO", raising=False)
    monkeypatch.setattr(AppSettings, "c60_ros_distro", "")
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
    )
    assert isinstance(executer, SyncExecuter)


def test_occupancy_mapper_populates_pm_on_step(minimal_corridor_map_path):
    ExecutionSettings.c40_map = str(minimal_corridor_map_path.resolve())
    ExecutionSettings.c40_mapping = OccupancyMapper.__name__

    mapped = executor_factory(
        load_plugins=False,
        executer_type=SyncExecuter.__name__,
        bridge="BasicSim",
        perception_strategy_name="",
        localization_strategy_name="",
        mapping_strategy_name=OccupancyMapper.__name__,
        global_planner_strategy_name="GlobalCenterlineRacePlanner",
        local_planner_strategy_name="GreedyLatticePlanner",
        controller_strategy_name=StanleyController.__name__,
        execution_task_names=["MappingTask"],
    )
    mapped.step(
        sim_dt=0.01, perception_dt=0.01, replan_dt=0.01, control_dt=0.01, localization_dt=0.01,
        call_replan=True, call_control=True, call_perceive=True,
    )
    assert isinstance(mapped.mapping, OccupancyMapper)
    mapping_task = next(t for t in mapped.task_runner.tasks if isinstance(t, MappingTask))
    assert mapped.pm.occupancy_map is not None
    assert mapped.pm.occupancy_map.grid.size > 0
    assert StackCapability.MAP_OCCUPANCY in mapping_task.stack_capabilities
    assert StackCapability.MAP_OCCUPANCY in mapped.available_stack_capabilities()

    reader = executor_factory(
        load_plugins=False,
        executer_type=SyncExecuter.__name__,
        bridge="BasicSim",
        perception_strategy_name="",
        localization_strategy_name="",
        mapping_strategy_name=MapReader.__name__,
        global_planner_strategy_name="GlobalCenterlineRacePlanner",
        local_planner_strategy_name="GreedyLatticePlanner",
        controller_strategy_name=StanleyController.__name__,
    )
    reader.step(
        sim_dt=0.01, perception_dt=0.01, replan_dt=0.01, control_dt=0.01, localization_dt=0.01,
        call_replan=True, call_control=True, call_perceive=True,
    )
    assert isinstance(reader.mapping, MapReader)
    assert reader.pm.occupancy_map is None


def test_occupancy_mapper_without_mapping_task_does_not_update(minimal_corridor_map_path):
    ExecutionSettings.c40_map = str(minimal_corridor_map_path.resolve())
    ExecutionSettings.c40_mapping = OccupancyMapper.__name__

    mapped = executor_factory(
        load_plugins=False,
        executer_type=SyncExecuter.__name__,
        bridge="BasicSim",
        perception_strategy_name="",
        localization_strategy_name="",
        mapping_strategy_name=OccupancyMapper.__name__,
        global_planner_strategy_name="GlobalCenterlineRacePlanner",
        local_planner_strategy_name="GreedyLatticePlanner",
        controller_strategy_name=StanleyController.__name__,
        execution_task_names=[],
    )
    mapped.step(
        sim_dt=0.01, perception_dt=0.01, replan_dt=0.01, control_dt=0.01, localization_dt=0.01,
        call_replan=True, call_control=True, call_perceive=True,
    )
    assert isinstance(mapped.mapping, OccupancyMapper)
    assert not any(isinstance(t, MappingTask) for t in mapped.task_runner.tasks)
    assert mapped.pm.occupancy_map is None
    assert StackCapability.MAP_OCCUPANCY in mapped.available_stack_capabilities()


def test_mapping_task_is_registered():
    from avlite.c40_execution.c43_task_strategy import TaskStrategy

    assert MappingTask.__name__ in TaskStrategy.registry
    assert TaskStrategy.registry["MappingTask"] is MappingTask


def test_factory_listed_mapping_task_gets_mapper_caps(minimal_corridor_map_path):
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
        execution_task_names=["MappingTask"],
    )
    mapping_tasks = [t for t in executer.task_runner.tasks if isinstance(t, MappingTask)]
    assert len(mapping_tasks) == 1
    assert StackCapability.MAP_RACE_TRACK in mapping_tasks[0].stack_capabilities
    assert StackCapability.MAP_RACE_TRACK in executer.available_stack_capabilities()


def test_sync_and_async_workers_have_no_inline_mapping_step():
    from avlite.c40_execution import c42_execution_strategy, c44_sync_executer, c45_async_threaded_executer

    strategy_src = Path(c42_execution_strategy.__file__).read_text(encoding="utf-8")
    sync_src = Path(c44_sync_executer.__file__).read_text(encoding="utf-8")
    async_src = Path(c45_async_threaded_executer.__file__).read_text(encoding="utf-8")
    assert "_mapping_step" not in strategy_src
    assert "_mapping_step" not in sync_src
    assert "_mapping_step" not in async_src


def test_factory_source_has_no_registry_yaml():
    text = Path(executor_factory.__code__.co_filename).read_text(encoding="utf-8")
    assert "yaml.safe_load" not in text
    assert ".avlite-registry" not in text
