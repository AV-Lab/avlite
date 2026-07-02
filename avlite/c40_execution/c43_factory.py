import inspect
import logging
from typing import Any

from avlite.c10_perception.c11_perception_model import Map, RaceMap
from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c20_planning.c29_settings import PlanningSettings
from avlite.c30_control.c39_settings import ControlSettings
from avlite.c60_common.c66_plugins import (
    import_plugin_modules,
    load_builtin_plugin_settings,
    reload_lib,
    sync_builtin_plugins,
    sync_community_plugins,
    unregister_plugin_package,
)
from avlite.c60_common.c67_paths import DataPaths, PluginPaths
from avlite.c60_common.c69_setting_utils import load_setting

from avlite.c10_perception.c11_perception_model import PerceptionModel, EgoState, AgentState, EGO_AGENT_ID
from avlite.c10_perception.c18_hdmap_parser import HDMap
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c10_perception.c12_perception_strategy import (
    DetectionStrategy,
    PerceptionPipeline,
    PerceptionStrategy,
    PredictionStrategy,
    TrackingStrategy,
)
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c10_perception.c13_localization_strategy import LocalizationStrategy
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c20_planning.c24_global_hdmap_planners import HDMapGlobalPlanner
from avlite.c20_planning.c25_global_race_planners import GlobalCenterlineRacePlanner
from avlite.c20_planning.c26_local_planners import VelocityLocalPlanner  # noqa: F401 — registers in LocalPlanningStrategy.registry
from avlite.c20_planning.c27_local_lattice_planners import GreedyLatticePlanner  # noqa: F401 — registers in LocalPlanningStrategy.registry
from avlite.c30_control.c33_pid import PIDController  # noqa: F401 — registers in ControlStrategy.registry
from avlite.c30_control.c34_stanley import StanleyController  # noqa: F401 — registers in ControlStrategy.registry
from avlite.c10_perception.c15_perception_algs import ConstantVelocityPrediction  # noqa: F401 — registers in PredictionStrategy.registry
from avlite.c10_perception.c16_localization_algs import LidarLocalization  # noqa: F401 — registers in LocalizationStrategy.registry
from avlite.c40_execution.c41_world_bridge import WorldBridge
from avlite.c40_execution.c42_executer import Executer
from avlite.c40_execution.c44_sync_executer import SyncExecuter  # noqa: F401 — registers in Executer.registry
from avlite.c40_execution.c45_async_threaded_executer import AsyncThreadedExecuter
from avlite.c40_execution.c46_basic_sim import BasicSim  # noqa: F401 — registers in WorldBridge.registry



log = logging.getLogger(__name__)


def executor_factory(
    executer_type = ExecutionSettings.c40_executer_type,
    bridge = ExecutionSettings.c40_bridge,
    perception_strategy_name = ExecutionSettings.c40_perception,
    localization_strategy_name = ExecutionSettings.c40_localization,
    global_planner_strategy_name = ExecutionSettings.c40_global_planner,
    local_planner_strategy_name = ExecutionSettings.c40_local_planner,
    controller_strategy_name = ExecutionSettings.c40_controller,
    perception_dt = ExecutionSettings.c40_perception_dt,
    localization_dt = ExecutionSettings.c40_localization_dt,
    replan_dt = ExecutionSettings.c40_replan_dt,
    control_dt = ExecutionSettings.c40_control_dt,
    default_global_trajectory_file = ExecutionSettings.c40_global_trajectory,
    hd_map = ExecutionSettings.c40_hd_map,
    load_plugins=True,
    async_combined_perception_planning: bool = ExecutionSettings.c40_async_combined_perception_planning,
) -> "Executer":
    """
    Factory method to create an instance of the Executer class based on the provided configuration.
    """

    if load_plugins:
        sync_builtin_plugins(list(ExecutionSettings.c40_default_plugins))
        sync_community_plugins(ExecutionSettings.c40_community_plugins)
    else:
        sync_builtin_plugins([])
        for k in ExecutionSettings.c40_community_plugins:
            unregister_plugin_package(k)

    global_plan_path = DataPaths.resolve_stored(default_global_trajectory_file)
    default_global_plan = GlobalPlan.from_file(global_plan_path)
    log.debug(f"Default global plan loaded from {global_plan_path}")

    ego_state = EgoState(x=default_global_plan.start_point[0], y=default_global_plan.start_point[1])
    ego_state.agent_id = EGO_AGENT_ID
    pm = PerceptionModel(ego_vehicle=ego_state)

    ###################
    # Loading default
    # global planner
    ###################

    if global_planner_strategy_name == HDMapGlobalPlanner.__name__:
        hdmap = HDMap(xodr_file_name=DataPaths.resolve_stored(hd_map))
        pm.map = hdmap
        gp = HDMapGlobalPlanner(hdmap)
        gp.global_plan = default_global_plan
        log.debug("GlobalHDMapPlanner loaded")
    elif global_planner_strategy_name == GlobalCenterlineRacePlanner.__name__:
        gp = GlobalCenterlineRacePlanner(
            DataPaths.resolve_stored(ExecutionSettings.c43_race_boundary_map),
        )
        gp.global_plan = default_global_plan
    else:
        gp_cls = _require_registered(
            global_planner_strategy_name, GlobalPlannerStrategy.registry, "global planner"
        )
        gp = gp_cls()
        gp.global_plan = default_global_plan

    ##############################
    # Loading perception strategy
    ##############################
    pr = None
    if perception_strategy_name:
        _require_registered(perception_strategy_name, PerceptionStrategy.registry, "perception")
        if perception_strategy_name == PerceptionPipeline.__name__:
            _require_pipeline_substrategies()
        pr = PerceptionStrategy.registry[perception_strategy_name](perception_model=pm)
        log.info("Perception Module Loaded!")

    #################################
    # Loading localization strategy
    #################################
    loc = None
    if localization_strategy_name:
        loc_cls = _require_registered(
            localization_strategy_name, LocalizationStrategy.registry, "localization"
        )
        loc = loc_cls(perception_model=pm)
        log.info("Localization Module Loaded!")

    ########################
    # Loading local planner
    #######################

    local_global_plan = default_global_plan
    if global_planner_strategy_name == HDMapGlobalPlanner.__name__:
        local_global_plan = GlobalPlan(
            start_point=default_global_plan.start_point,
            goal_point=default_global_plan.goal_point,
            path=default_global_plan.path,
            velocity=default_global_plan.velocity,
            trajectory=default_global_plan.trajectory,
        )

    pl_cls = _require_registered(
        local_planner_strategy_name, LocalPlanningStrategy.registry, "local planner"
    )
    pl = pl_cls(global_plan=local_global_plan, env=pm)

    #################
    # Loading controller
    #################
    cn_cls = _require_registered(controller_strategy_name, ControlStrategy.registry, "controller")
    cn = cn_cls()
    if default_global_plan.trajectory is not None:
        cn.set_trajectory(default_global_plan.trajectory)

    #################
    # Loading world
    #################
    # The world owns its own authoritative perception model (holding spawned
    # NPC agents / ground-truth state), kept separate from the executer's
    # perception model `pm` so that perception steps cannot wipe simulated
    # agents. Both share the same ego_state.
    def _bridge_kwargs(bridge_cls, ego_state, world_pm):
        def _reference_point_tuple():
            ref = ExecutionSettings.c40_reference_point
            if ref and len(ref) >= 2:
                return float(ref[0]), float(ref[1])
            return None

        params = inspect.signature(bridge_cls.__init__).parameters
        kwargs: dict = {}
        if "ego_state" in params:
            kwargs["ego_state"] = ego_state
        if "pm" in params:
            kwargs["pm"] = world_pm
        if "reference_point" in params:
            kwargs["reference_point"] = _reference_point_tuple()
        return kwargs

    world_pm = PerceptionModel(ego_vehicle=ego_state)
    bridge_cls = _require_registered(bridge, WorldBridge.registry, "world bridge")
    log.info(f"Loading registered world bridge {bridge}...")
    world = bridge_cls(**_bridge_kwargs(bridge_cls, ego_state, world_pm))

    ego = world.ego_state
    if ego.agent_id != EGO_AGENT_ID:
        log.warning("Ego agent_id=%s; expected %s", ego.agent_id, EGO_AGENT_ID)

    #################
    # Creating Executer
    #################
    executer_cls = _require_registered(executer_type, Executer.registry, "executer")
    kwargs = dict(
        perception_model=pm,
        perception=pr,
        global_planner=gp,
        local_planner=pl,
        controller=cn,
        world=world,
        localization=loc,
        perception_dt=perception_dt,
        replan_dt=replan_dt,
        control_dt=control_dt,
        localization_dt=localization_dt,
    )
    if issubclass(executer_cls, AsyncThreadedExecuter):
        kwargs["combined_perception_planning"] = async_combined_perception_planning
    executer = executer_cls(**kwargs)
    executer._requested_executer_type = executer_type
    return executer


def get_stack_settings_classes() -> list[Any]:
    """Layer singletons plus built-in plugin settings classes for export/import."""
    classes: list[Any] = [
        PerceptionSettings,
        PlanningSettings,
        ControlSettings,
        ExecutionSettings,
    ]
    from avlite.c60_common.c66_plugins import list_plugins

    for plugin in list_plugins():
        cls = load_builtin_plugin_settings(plugin)
        if cls is not None:
            classes.append(cls)
    return classes


def load_stack_settings(profile: str = "default", load_plugins: bool = True) -> None:
    """Load c10–c40 YAML singletons and built-in plugin settings; bootstrap ref point."""
    load_setting(PerceptionSettings, profile=profile)
    load_setting(PlanningSettings, profile=profile)
    load_setting(ControlSettings, profile=profile)
    load_setting(ExecutionSettings, profile=profile)

    StackSettingsSync.bootstrap_reference_point()

    if not load_plugins:
        return

    for name, stored in ExecutionSettings.c40_community_plugins.items():
        path = PluginPaths.resolve(name, stored)
        if path.is_dir():
            import_plugin_modules(str(path), pkg_name=name)

    for plugin in ExecutionSettings.c40_default_plugins:
        cls = load_builtin_plugin_settings(plugin)
        if cls is None:
            continue
        load_setting(cls, profile=profile)


class StackSettingsSync:
    """Apply map/plan picker selections to execution settings."""

    @staticmethod
    def apply_map_selection(rel_path: str) -> None:
        """Route *rel_path* to execution map settings and update reference point."""
        abs_path = DataPaths.resolve_stored(rel_path)
        if rel_path.endswith(".xodr"):
            ExecutionSettings.c40_hd_map = rel_path
            ExecutionSettings.c46_lidar_boundary_file = ""
        elif RaceMap.is_loadable(abs_path):
            ExecutionSettings.c43_race_boundary_map = rel_path
            ExecutionSettings.c46_lidar_boundary_file = rel_path
        m = Map.open(abs_path)
        ref = m.reference_point if m else None
        ExecutionSettings.c40_reference_point = list(ref) if ref else None

    @staticmethod
    def apply_global_plan_selection(rel_path: str) -> None:
        """Set ``c40_global_trajectory`` when *rel_path* is a valid global-plan JSON."""
        if GlobalPlan.is_loadable(DataPaths.resolve_stored(rel_path)):
            ExecutionSettings.c40_global_trajectory = rel_path

    @staticmethod
    def bootstrap_reference_point() -> None:
        """Fill ``c40_reference_point`` from configured maps when YAML omits it."""
        if ExecutionSettings.c40_reference_point is not None:
            return
        for rel_path in (
            ExecutionSettings.c43_race_boundary_map,
            ExecutionSettings.c40_hd_map,
        ):
            m = Map.open(DataPaths.resolve_stored(rel_path))
            if m is not None and m.reference_point is not None:
                ExecutionSettings.c40_reference_point = list(m.reference_point)
                return


def _require_registered(name: str, registry: dict, label: str):
    if name not in registry:
        raise ValueError(f"Could not load {label} '{name}': not recognized.")
    return registry[name]


def _missing_pipeline_substrategies() -> list[tuple[str, str]]:
    missing: list[tuple[str, str]] = []
    for stage, name, registry in (
        ("detection", PerceptionSettings.c12_detection_strategy, DetectionStrategy.registry),
        ("tracking", PerceptionSettings.c12_tracking_strategy, TrackingStrategy.registry),
        ("prediction", PerceptionSettings.c12_prediction_strategy, PredictionStrategy.registry),
    ):
        if name and name not in registry:
            missing.append((stage, name))
    return missing


def _require_pipeline_substrategies() -> None:
    missing = _missing_pipeline_substrategies()
    if missing:
        detail = ", ".join(f"{stage} '{name}'" for stage, name in missing)
        raise ValueError(f"Could not load PerceptionPipeline sub-strategies: {detail}")
