import logging


from avlite.c10_perception.c11_perception_model import PerceptionModel, EgoState, AgentState
from avlite.c10_perception.c18_hdmap import HDMap
from avlite.c30_control.c31_control_model import  ControlComand
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c10_perception.c12_perception_strategy import PerceptionStrategy
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c23_local_planning_strategy import LocalPlannerStrategy
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c60_common.c61_setting_utils import reload_lib, get_absolute_path, import_all_modules

from avlite.c10_perception.c11_perception_model import PerceptionModel, EgoState
from avlite.c10_perception.c12_perception_strategy import PerceptionStrategy
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c20_planning.c24_global_planners import HDMapGlobalPlanner
from avlite.c20_planning.c24_global_planners import RaceGlobalPlanner
from avlite.c20_planning.c26_local_planners import GreedyLatticePlanner
from avlite.c30_control.c33_pid import PIDController
from avlite.c30_control.c34_stanley import StanleyController
from avlite.c40_execution.c41_execution_model import Executer, WorldBridge
from avlite.c40_execution.c43_sync_executer import SyncExecuter
from avlite.c40_execution.c44_async_threaded_executer import AsyncThreadedExecuter
from avlite.c40_execution.c46_basic_sim import BasicSim


log = logging.getLogger(__name__)

def executor_factory(
    executer_type = ExecutionSettings.executer_type,
    bridge = ExecutionSettings.bridge,
    perception_strategy_name = ExecutionSettings.perception,
    global_planner_strategy_name = ExecutionSettings.global_planner,
    local_planner_strategy_name = ExecutionSettings.local_planner,
    controller_strategy_name = ExecutionSettings.controller,
    perception_dt = ExecutionSettings.perception_dt,
    replan_dt = ExecutionSettings.replan_dt,
    control_dt = ExecutionSettings.control_dt,
    default_global_trajectory_file = ExecutionSettings.global_trajectory,
    hd_map = ExecutionSettings.hd_map,
    load_extensions=True,
) -> "Executer":
    """
    Factory method to create an instance of the Executer class based on the provided configuration.
    """


    
    if load_extensions:
        import_all_modules() # loading default extensions
        # loading community extensions
        for k,v in ExecutionSettings.community_extensions.items():
            log.warning(f"Loading external extension: {k} from {v}")
            import_all_modules(v, pkg_name = k)


    global_plan_path =  get_absolute_path(default_global_trajectory_file)
    default_global_plan = GlobalPlan.from_file(global_plan_path)
    ego_state = EgoState(x=default_global_plan.start_point[0], y=default_global_plan.start_point[1])
    pm = PerceptionModel(ego_vehicle=ego_state)
    
    ###################
    # Loading default
    # global planner
    ###################
    
    try:
        if global_planner_strategy_name == HDMapGlobalPlanner.__name__:
            hdmap = HDMap(xodr_file_name=hd_map)
            pm.hd_map = hdmap
            gp = HDMapGlobalPlanner(hdmap)
            log.debug("GlobalHDMapPlanner loaded")
        elif global_planner_strategy_name in GlobalPlannerStrategy.registry:
            cls = GlobalPlannerStrategy.registry[global_planner_strategy_name]
            gp = cls()

    except Exception as e:
        log.error(f"Failed to load global planner {global_planner_strategy_name}. Loading default")
        gp = RaceGlobalPlanner()
        

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


    ########################
    # Loading local planner
    #######################

    try:
        if local_planner_strategy_name in LocalPlannerStrategy.registry:
            cls = LocalPlannerStrategy.registry[local_planner_strategy_name]
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
    try:
        if bridge == "CarlaBridge": # string for lazy loading, beause it could have dependencies that are not available
            print("Loading Carla bridge...")
            from avlite.c40_execution.c47_carla_bridge import CarlaBridge
            world = CarlaBridge(ego_state=ego_state)
        elif bridge == "GazeboIgnitionBridge":
            print("Loading Gazebo bridge...")
            from avlite.c40_execution.c48_gazebo_bridge import GazeboIgnitionBridge
            world = GazeboIgnitionBridge(ego_state=ego_state)
        else:
            world = BasicSim(ego_state=ego_state, pm = pm)
    except Exception as e:
        log.error(f"Error loading world bridge {bridge}: {e}")
        world = BasicSim(ego_state=ego_state, pm = pm)  # fallback to BasicSim



    #################
    # Creating Executer
    #################
    try:
        if executer_type in Executer.registry:
            cls = Executer.registry[executer_type]
            executer = cls(perception_model=pm,perception=pr, global_planner=gp, local_planner=pl,
                           controller=cn, world=world, perception_dt=perception_dt, replan_dt=replan_dt, control_dt=control_dt)
        else:
            log.error(f"Invalid Executer. Moving to default executer")
            executer = SyncExecuter(perception_model=pm,perception=pr, global_planner=gp, local_planner=pl,
                           controller=cn, world=world, perception_dt=perception_dt, replan_dt=replan_dt, control_dt=control_dt)
    except Exception as e:
        log.error(f"Error loading exectuter {e}")

    return executer

