from os import wait
from c50_visualization.c51_visualizer_app import VisualizerApp
from c60_tools.c61_utils import load_config, reload_lib, get_absolute_path
from c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from c40_execution.c42_sync_executer import SyncExecuter
from c40_execution.c43_async_threaded_executer import AsyncThreadedExecuter

import numpy as np

import logging

log = logging.getLogger(__name__)


def get_executer(
    config_path="configs/c20_planning.yaml",
    async_mode=False,
    bridge="Basic",
    global_planner=list(GlobalPlannerStrategy.registry.values())[0].__name__,
    source_run=True,
    replan_dt=0.5,
    control_dt=0.05,
) -> (SyncExecuter | AsyncThreadedExecuter):


    reload_lib()
    from c10_perception.c11_perception_model import PerceptionModel, EgoState
    from c20_planning.c21_planning_model import GlobalPlan
    from c20_planning.c25_hdmap_global_planner import HDMapGlobalPlanner
    from c20_planning.c24_race_global_planner import RaceGlobalPlanner
    from c20_planning.c26_sampling_local_planner import RNDPlanner
    from c30_control.c33_pid import PIDController
    from c40_execution.c42_sync_executer import SyncExecuter
    from c40_execution.c44_async_mproc_executer import AsyncExecuter
    from c40_execution.c43_async_threaded_executer import AsyncThreadedExecuter
    from c40_execution.c44_basic_sim import BasicSim
    
    config_data = load_config(config_path=config_path)
    global_plan_path =  get_absolute_path(config_data["global_trajectory"])
    
    global_plan = GlobalPlan.from_file(global_plan_path)
    
    if global_planner == RaceGlobalPlanner.__name__:
        gp = RaceGlobalPlanner()
        gp.global_plan = global_plan
        log.debug("RaceGlobalPlanner loaded")
    elif global_planner == HDMapGlobalPlanner.__name__:
        gp = HDMapGlobalPlanner(xodr_file=config_data["hd_map"])
        log.debug("GlobalHDMapPlanner loaded")

    ego_state = EgoState(x=global_plan.start_point[0], y=global_plan.start_point[1], velocity=global_plan.velocity[0], theta=-np.pi / 4)
    pm = PerceptionModel(ego_vehicle=ego_state)

    # Loading world
    if bridge == "Carla":
        print("Loading Carla bridge...")
        from c40_execution.c45_carla_bridge import CarlaBridge
        world = CarlaBridge(ego_state=ego_state)
    elif bridge == "ROS":
        raise NotImplementedError("ROS bridge not implemented")
    else:
        world = BasicSim(ego_state=ego_state)


    local_planner = RNDPlanner(
        global_plan=global_plan,
        env=pm,
    )
    controller = PIDController()
    executer = (
        SyncExecuter(pm=pm, global_planner=gp, local_planner=local_planner, controller=controller, world=world, replan_dt=replan_dt, control_dt=control_dt)
        if not async_mode
        else AsyncThreadedExecuter(perception_model=pm, global_planner=gp, local_planner=local_planner, controller=controller, world=world, replan_dt=replan_dt, control_dt=control_dt)
    )

    return executer


def run(source_run=True):
    executer = get_executer(config_path="configs/c20_planning.yaml", source_run=source_run)
    app = VisualizerApp(executer, code_reload_function=get_executer)
    app.mainloop()


if __name__ == "__main__":
    source_run = True

    import platform
    import os

    if platform.system() == "Linux":
        os.environ["TK_WINDOWS_FORCE_OPENGL"] = "1"
    else:
        import ctypes

        try:  # >= win 8.
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except:  # win 8.0 or less
            ctypes.windll.user32.SetProcessDPIAware()
        os.environ["TK_WINDOWS_FORCE_OPENGL"] = "1"

    try:
        run(source_run)
    except Exception as e:
        log.error(f"An error occurred: {e}")
