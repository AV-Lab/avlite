from c50_visualization.c51_visualizer_app import VisualizerApp
from c20_planning.c22_base_global_planner import BaseGlobalPlanner
from c40_execution.c41_base_executer import BaseExecuter
from c40_execution.c43_async_threaded_executer import AsyncThreadedExecuter
from c60_tools.c61_utils import load_config, reload_lib

import numpy as np

import logging

log = logging.getLogger(__name__)


def get_executer(
    config_path="configs/c20_planning.yaml",
    async_mode=False,
    bridge="Basic",
    global_planner=list(BaseGlobalPlanner.registry.values())[0].__name__,
    source_run=True,
    replan_dt=0.5,
    control_dt=0.05,
) -> (BaseExecuter | AsyncThreadedExecuter):

    reference_path, reference_velocity, ref_left_boundary_d, ref_right_boundary_d, config_data = load_config(
        config_path=config_path, source_run=source_run
    )

    reload_lib()
    from c10_perception.c11_perception_model import PerceptionModel, EgoState
    from c20_planning.c25_hdmap_global_planner import GlobalHDMapPlanner
    from c20_planning.c24_race_global_planner import RaceGlobalPlanner
    from c20_planning.c26_sampling_local_planner import RNDPlanner
    from c30_control.c33_pid import PIDController
    from c40_execution.c41_base_executer import BaseExecuter
    from c40_execution.c42_async_executer import AsyncExecuter
    from c40_execution.c43_async_threaded_executer import AsyncThreadedExecuter
    from c40_execution.c44_basic_sim import BasicSim

    ego_state = EgoState(x=reference_path[0][0], y=reference_path[0][1], velocity=reference_velocity[0], theta=-np.pi / 4)
    pm = PerceptionModel(ego_vehicle=ego_state)
    # Loading bridge
    if bridge == "Carla":
        print("Loading Carla bridge...")
        from c40_execution.c45_carla_bridge import CarlaBridge

        world = CarlaBridge(ego_state=ego_state)
    elif bridge == "ROS":
        raise NotImplementedError("ROS bridge not implemented")
    else:
        world = BasicSim(ego_state=ego_state)



    if global_planner == RaceGlobalPlanner.__name__:
        gp = RaceGlobalPlanner()
        log.debug("RaceGlobalPlanner loaded")
    elif global_planner == GlobalHDMapPlanner.__name__:
        gp = GlobalHDMapPlanner(xodr_file=config_data["hd_map"])
        log.debug("GlobalHDMapPlanner loaded")

    pl = RNDPlanner(
        global_path=reference_path,
        global_velocity=reference_velocity,
        ref_left_boundary_d=ref_left_boundary_d,
        ref_right_boundary_d=ref_right_boundary_d,
        env=pm,
    )
    cn = PIDController()
    executer = (
        BaseExecuter(pm=pm, glob_pl=gp, pl=pl, cn=cn, world=world, replan_dt=replan_dt, control_dt=control_dt)
        if not async_mode
        else AsyncThreadedExecuter(
            pm=pm, glob_pl=gp, pl=pl, cn=cn, world=world, replan_dt=replan_dt, control_dt=control_dt
        )
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

    run(source_run)
