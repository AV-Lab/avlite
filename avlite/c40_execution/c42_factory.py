from typing import Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging 
import numpy as np
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
    async_mode = ExecutionSettings.async_mode,
    bridge = ExecutionSettings.bridge,
    perception = ExecutionSettings.perception,
    global_planner = ExecutionSettings.global_planner,
    local_planner = ExecutionSettings.local_planner,
    controller = ExecutionSettings.controller,
    replan_dt = ExecutionSettings.replan_dt,
    control_dt = ExecutionSettings.control_dt,
    global_trajectory = ExecutionSettings.global_trajectory,
    hd_map = ExecutionSettings.hd_map,
    reload_code = True,
    exclude_reload_settings = False,
    load_extensions=True,
) -> "Executer":
    """
    Factory method to create an instance of the Executer class based on the provided configuration.
    """

    if reload_code:
        reload_lib(exclude_settings=exclude_reload_settings)

    
    if load_extensions:
        import_all_modules()

        log.warning(f"external_extensions: {ExecutionSettings.community_extensions}")
        for k,v in ExecutionSettings.community_extensions.items():
            log.warning(f"Loading external extension: {k} from {v}")
            import_all_modules(v, pkg_name = k)
        log.debug(f"perception registry after: {PerceptionStrategy.registry.keys()}")

        # from extensions.multi_object_prediction.e10_perception.perception import MultiObjectPredictor
        # from extensions.test_ext.e10_perception.test import testClass



    #################
    global_plan_path =  get_absolute_path(global_trajectory)

    # Loading default
    # global planner
    #################
    default_global_plan = GlobalPlan.from_file(global_plan_path)
    
    if global_planner is None:
        gp = RaceGlobalPlanner()
        gp.global_plan = default_global_plan
        log.debug("RaceGlobalPlanner loaded by default. If you want to use a different global planner, please specify it in the config file or as an argument.")

    elif global_planner == RaceGlobalPlanner.__name__:
        gp = RaceGlobalPlanner()
        gp.global_plan = default_global_plan
        log.debug("RaceGlobalPlanner loaded")
    elif global_planner == HDMapGlobalPlanner.__name__:

        hdmap = HDMap(xodr_file_name=hd_map)
        gp = HDMapGlobalPlanner(hdmap)
        log.debug("GlobalHDMapPlanner loaded")

    ego_state = EgoState(x=default_global_plan.start_point[0], y=default_global_plan.start_point[1])
    pm = PerceptionModel(ego_vehicle=ego_state)
        


    ############################
    # Loading perception strategy
    ##############################
    pr = None
    if perception is not None and perception != "" and perception in  PerceptionStrategy.registry:
        # load the class
        cls = PerceptionStrategy.registry[perception]
        pr = cls(perception_model=pm)
        log.info("Perception Module Loaded!")
    # perception = MultiObjectPredictor(perception_model=pm)
    
    # m=load_an_external_module("/home/mkhonji/Desktop/delete_me/extensions/perc.py")
    # log.warning(f"Loaded external extension: {m.__name__} from {m.__file__}")
    # pr = m.ExternalClass(pm)
    # pr.perceive()
    # log.warning(f"perception registry after: {PerceptionStrategy.registry.keys()}")

    #################
    # Loading world
    #################
    if bridge == "CarlaBridge":
        print("Loading Carla bridge...")
        from avlite.c40_execution.c47_carla_bridge import CarlaBridge
        world = CarlaBridge(ego_state=ego_state)
    elif bridge == "GazeboIgnitionBridge":
        print("Loading Gazebo bridge...")
        from avlite.c40_execution.c48_gazebo_bridge import GazeboIgnitionBridge
        world = GazeboIgnitionBridge(ego_state=ego_state)
    else:
        world = BasicSim(ego_state=ego_state, pm = pm)


    #################
    # Loading planner
    #################
    if local_planner is None or local_planner == GreedyLatticePlanner.__name__:
        local_planner = GreedyLatticePlanner(global_plan=default_global_plan, env=pm)
    else:
        log.error(f"Local planner {local_planner} not recognized. Using {GreedyLatticePlanner.__name__} as default.")
        local_planner = GreedyLatticePlanner(global_plan=default_global_plan, env=pm)

    #################
    # Loading planner
    #################
    if controller is None or controller == PIDController.__name__:
        controller = PIDController()
    elif controller == StanleyController.__name__:
        controller = StanleyController()
    else:
        log.error(f"Controller {controller} not recognized. Using PIDController as default.")
        controller = PIDController()


    #################
    # Creating Executer
    #################
    executer = (
        SyncExecuter(perception_model=pm,perception=pr, global_planner=gp, local_planner=local_planner, controller=controller, world=world, replan_dt=replan_dt, control_dt=control_dt)
        if not async_mode
        else AsyncThreadedExecuter(perception_model=pm,perception=pr, global_planner=gp, local_planner=local_planner, controller=controller, world=world, replan_dt=replan_dt, control_dt=control_dt)
    )

    return executer

