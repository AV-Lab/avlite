import logging


from avlite.c10_perception.c11_perception_model import PerceptionModel, EgoState, AgentState
from avlite.c60_common.c66_hdmap import HDMap
from avlite.c30_control.c31_control_model import  ControlComand
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c10_perception.c12_perception_strategy import PerceptionStrategy
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c60_common.c60_plugins import import_plugin_modules, reload_lib
from avlite.c60_common.c67_paths import get_absolute_path, resolve_plugin_path

from avlite.c10_perception.c11_perception_model import PerceptionModel, EgoState
from avlite.c10_perception.c12_perception_strategy import PerceptionStrategy
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
        import_plugin_modules(plugins_filter=list(ExecutionSettings.c40_default_plugins))
        for k, v in ExecutionSettings.c40_community_plugins.items():
            path = resolve_plugin_path(k, v)
            log.warning("Loading external plugin: %s from %s", k, path)
            import_plugin_modules(str(path), pkg_name=k)


    try:
        global_plan_path = get_absolute_path(default_global_trajectory_file)
        default_global_plan = GlobalPlan.from_file(global_plan_path)
        log.debug(f"Default global trajectory loaded from {global_plan_path}")
    except Exception as e:
        log.warning(
            f"Could not load default global trajectory '{default_global_trajectory_file}': {e}. "
            "Falling back to race boundary centerline."
        )
        _fallback_planner = GlobalCenterlineRacePlanner(
            get_absolute_path(ExecutionSettings.c43_race_boundary_map),
            margin=ExecutionSettings.c43_race_boundary_margin,
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
            hdmap = HDMap(xodr_file_name=get_absolute_path(hd_map))
            pm.hd_map = hdmap
            gp = HDMapGlobalPlanner(hdmap)
            log.debug("GlobalHDMapPlanner loaded")
        elif global_planner_strategy_name == GlobalCenterlineRacePlanner.__name__:
            gp = GlobalCenterlineRacePlanner(
                get_absolute_path(ExecutionSettings.c43_race_boundary_map),
                margin=ExecutionSettings.c43_race_boundary_margin,
            )
            gp.plan()
        elif global_planner_strategy_name in GlobalPlannerStrategy.registry:
            cls = GlobalPlannerStrategy.registry[global_planner_strategy_name]
            gp = cls()
            gp.global_plan = default_global_plan
        else:
            log.error(f"Global planner '{global_planner_strategy_name}' not recognized. Loading default.")
            gp = GlobalCenterlineRacePlanner(
                get_absolute_path(ExecutionSettings.c43_race_boundary_map),
                margin=ExecutionSettings.c43_race_boundary_margin,
            )
            gp.plan()

    except Exception as e:
        log.error(f"Failed to load global planner {global_planner_strategy_name}. Loading default")
        gp = GlobalCenterlineRacePlanner(
            get_absolute_path(ExecutionSettings.c43_race_boundary_map),
            margin=ExecutionSettings.c43_race_boundary_margin,
        )
        gp.plan()
        

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

    try:
        if local_planner_strategy_name in LocalPlanningStrategy.registry:
            cls = LocalPlanningStrategy.registry[local_planner_strategy_name]
            pl = cls(global_plan=default_global_plan, env=pm)
        else:
            log.error(f"Unable to load local planner {local_planner_strategy_name}. Switching to default.")
            pl = GreedyLatticePlanner(global_plan=default_global_plan, env=pm)

    except Exception as e:
        log.error(f"Failed to load local planner: {e}. Switching to default.")
        pl = GreedyLatticePlanner(global_plan=default_global_plan, env=pm)

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
    world_pm = PerceptionModel(ego_vehicle=ego_state)
    try:
        if bridge in WorldBridge.registry:
            log.info(f"Loading registered world bridge {bridge}...")
            cls = WorldBridge.registry[bridge]
            world = cls(ego_state=ego_state, pm=world_pm)
        else:
            world = BasicSim(ego_state=ego_state, pm = world_pm)
    except Exception as e:
        log.error(f"Error loading world bridge {bridge}: {e}")
        world = BasicSim(ego_state=ego_state, pm = world_pm)  # fallback to BasicSim



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
