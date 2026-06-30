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
    unregister_plugin_package,
)
from avlite.c60_common.c67_paths import DataPaths, PluginPaths
from avlite.c60_common.c69_setting_utils import load_setting

from avlite.c10_perception.c11_perception_model import PerceptionModel, EgoState, AgentState
from avlite.c10_perception.c18_hdmap_parser import HDMap
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c10_perception.c12_perception_strategy import PerceptionStrategy
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c10_perception.c13_localization_strategy import LocalizationStrategy
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c20_planning.c24_global_hdmap_planners import HDMapGlobalPlanner
from avlite.c20_planning.c25_global_race_planners import GlobalCenterlineRacePlanner
from avlite.c20_planning.c26_local_lattice_planners import GreedyLatticePlanner
from avlite.c30_control.c33_pid import PIDController
from avlite.c30_control.c34_stanley import StanleyController
from avlite.c10_perception.c15_perception_algs import ConstantVelocityPrediction  # noqa: F401 — registers in PredictionStrategy.registry
from avlite.c10_perception.c16_localization_algs import LidarLocalization  # noqa: F401 — registers in LocalizationStrategy.registry
from avlite.c40_execution.c41_world_bridge import WorldBridge
from avlite.c40_execution.c42_executer import Executer
from avlite.c40_execution.c44_sync_executer import SyncExecuter
from avlite.c40_execution.c45_async_threaded_executer import AsyncThreadedExecuter
from avlite.c40_execution.c46_basic_sim import BasicSim



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
        for k, v in ExecutionSettings.c40_community_plugins.items():
            path = PluginPaths.resolve(k, v)
            log.warning("Loading external plugin: %s from %s", k, path)
            import_plugin_modules(str(path), pkg_name=k)
    else:
        sync_builtin_plugins([])
        for k in ExecutionSettings.c40_community_plugins:
            unregister_plugin_package(k)


    try:
        global_plan_path = DataPaths.resolve_stored(default_global_trajectory_file)
        default_global_plan = GlobalPlan.from_file(global_plan_path)
        log.debug(f"Default global trajectory loaded from {global_plan_path}")
    except Exception as e:
        log.warning(
            f"Could not load default global trajectory '{default_global_trajectory_file}': {e}. "
            "Falling back to race boundary centerline."
        )
        _fallback_planner = GlobalCenterlineRacePlanner(
            DataPaths.resolve_stored(ExecutionSettings.c43_race_boundary_map),
        )
        default_global_plan = _fallback_planner.plan()

    ego_state = EgoState(x=default_global_plan.start_point[0], y=default_global_plan.start_point[1])
    pm = PerceptionModel(ego_vehicle=ego_state)

    ###################
    # Loading default
    # global planner
    ###################
    
    try:
        if global_planner_strategy_name == HDMapGlobalPlanner.__name__:
            hdmap = HDMap(xodr_file_name=DataPaths.resolve_stored(hd_map))
            pm.map = hdmap
            gp = HDMapGlobalPlanner(hdmap)
            log.debug("GlobalHDMapPlanner loaded")
        elif global_planner_strategy_name == GlobalCenterlineRacePlanner.__name__:
            gp = GlobalCenterlineRacePlanner(
                DataPaths.resolve_stored(ExecutionSettings.c43_race_boundary_map),
            )
            gp.global_plan = default_global_plan
        elif global_planner_strategy_name in GlobalPlannerStrategy.registry:
            cls = GlobalPlannerStrategy.registry[global_planner_strategy_name]
            gp = cls()
            gp.global_plan = default_global_plan
        else:
            log.error(f"Global planner '{global_planner_strategy_name}' not recognized. Loading default.")
            gp = GlobalCenterlineRacePlanner(
                DataPaths.resolve_stored(ExecutionSettings.c43_race_boundary_map),
            )
            gp.global_plan = default_global_plan

    except Exception as e:
        log.error(f"Failed to load global planner {global_planner_strategy_name}. Loading default")
        gp = GlobalCenterlineRacePlanner(
            DataPaths.resolve_stored(ExecutionSettings.c43_race_boundary_map),
        )
        gp.global_plan = default_global_plan

    ##############################
    # Loading perception strategy
    ##############################
    pr = None
    try:
        if perception_strategy_name is not None and perception_strategy_name != "" and perception_strategy_name in  PerceptionStrategy.registry:
            # load the class
            cls = PerceptionStrategy.registry[perception_strategy_name]
            pr = cls(perception_model=pm)
            log.info("Perception Module Loaded!")
    except Exception as e:
        log.error(f"Error loading perception strategy {perception_strategy_name}: {e}")
        pr = None


    #################################
    # Loading localization strategy
    #################################
    loc = None
    try:
        if localization_strategy_name is not None and localization_strategy_name != "" and localization_strategy_name in LocalizationStrategy.registry:
            cls = LocalizationStrategy.registry[localization_strategy_name]
            loc = cls(perception_model=pm)
            log.info("Localization Module Loaded!")
    except Exception as e:
        log.error(f"Error loading localization strategy {localization_strategy_name}: {e}")
        loc = None


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

    try:
        if local_planner_strategy_name in LocalPlanningStrategy.registry:
            cls = LocalPlanningStrategy.registry[local_planner_strategy_name]
            pl = cls(global_plan=local_global_plan, env=pm)
        else:
            log.error(f"Unable to load local planner {local_planner_strategy_name}. Switching to default.")
            pl = GreedyLatticePlanner(global_plan=local_global_plan, env=pm)

    except Exception as e:
        log.error(f"Failed to load local planner: {e}. Switching to default.")
        pl = GreedyLatticePlanner(global_plan=local_global_plan, env=pm)

    #################
    # Loading controller
    #################
    try:
        if controller_strategy_name in ControlStrategy.registry:
            cls = ControlStrategy.registry[controller_strategy_name]
            cn = cls()

        else:
            log.error(f"Controller {controller_strategy_name} not recognized. Using StanleyController as default.")
            cn = StanleyController()
            
        if default_global_plan.trajectory is not None:
            cn.set_trajectory(default_global_plan.trajectory)

    except Exception as e:
        log.error(f"Error loading controller {e}. Setting controller to Stanley")
        cn = StanleyController()
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
    try:
        if bridge in WorldBridge.registry:
            log.info(f"Loading registered world bridge {bridge}...")
            cls = WorldBridge.registry[bridge]
            world = cls(**_bridge_kwargs(cls, ego_state, world_pm))
        else:
            world = BasicSim(**_bridge_kwargs(BasicSim, ego_state, world_pm))
    except Exception as e:
        log.error(f"Error loading world bridge {bridge}: {e}")
        world = BasicSim(**_bridge_kwargs(BasicSim, ego_state, world_pm))  # fallback to BasicSim



    #################
    # Creating Executer
    #################
    executer = None
    try:
        if executer_type in Executer.registry:
            cls = Executer.registry[executer_type]
            kwargs = dict(perception_model=pm, perception=pr, global_planner=gp, local_planner=pl,
                          controller=cn, world=world, localization=loc,
                          perception_dt=perception_dt, replan_dt=replan_dt, control_dt=control_dt,
                          localization_dt=localization_dt)
            if issubclass(cls, AsyncThreadedExecuter):
                kwargs["combined_perception_planning"] = async_combined_perception_planning
            executer = cls(**kwargs)
        else:
            log.error(f"Invalid Executer. Moving to default executer")
            executer = SyncExecuter(perception_model=pm,perception=pr, global_planner=gp, local_planner=pl,
                           controller=cn, world=world, localization=loc,
                           perception_dt=perception_dt, replan_dt=replan_dt, control_dt=control_dt,
                           localization_dt=localization_dt)
    except Exception as e:
        log.error(f"Error loading executer '{executer_type}': {e}", exc_info=True)
        try:
            executer = SyncExecuter(perception_model=pm,perception=pr, global_planner=gp, local_planner=pl,
                           controller=cn, world=world, localization=loc,
                           perception_dt=perception_dt, replan_dt=replan_dt, control_dt=control_dt,
                           localization_dt=localization_dt)
        except Exception as e2:
            log.error(f"Fallback SyncExecuter also failed: {e2}", exc_info=True)
            raise

    if executer is not None:
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
