from c50_visualize.c51_visualizer_app import VisualizerApp
from utils import load_config, reload_lib

import numpy as np

import logging

log = logging.getLogger(__name__)


def get_executer(
    config_path="configs/c20_plan.yaml",
    async_mode=False,
    bridge="Basic",
    source_run=True,
    replan_dt=0.5,
    control_dt=0.05,
):

    reference_path, reference_velocity, ref_left_boundary_d, ref_right_boundary_d, config_data = load_config(
        config_path=config_path, source_run=source_run
    )

    reload_lib()
    from c10_perceive.c11_base_perception import PerceptionModel
    from c20_plan.c25_sampling_local_planner import RNDPlanner
    from c30_control.c32_pid_controller import PIDController
    from c40_execute.c41_base_executer import BaseExecuter
    from c40_execute.c42_async_executer import AsyncExecuter
    from c40_execute.c43_async_threaded_executer import AsyncThreadedExecuter
    from c40_execute.c44_basic_sim import BasicSim
    from c10_perceive.c12_state import EgoState

    ego_state = EgoState(x=reference_path[0][0], y=reference_path[0][1], speed=reference_velocity[0], theta=-np.pi / 4)

    # Loading bridge
    if bridge == "Basic":
        world = BasicSim(ego_state=ego_state)
    elif bridge == "Carla":
        print("Loading Carla bridge...")
        from c40_execute.c45_carla_bridge import CarlaBridge
        world = CarlaBridge(ego_state=ego_state)
    elif bridge == "ROS":
        raise NotImplementedError("ROS bridge not implemented")

    pm = PerceptionModel(ego_state)

    pl = RNDPlanner(
       global_path=reference_path,
        global_velocity=reference_velocity,
        ref_left_boundary_d=ref_left_boundary_d,
        ref_right_boundary_d=ref_right_boundary_d,
        env=pm,
    )
    cn = PIDController()
    executer = (
        BaseExecuter(pm, pl, cn, world, replan_dt=replan_dt, control_dt=control_dt)
        if not async_mode
        else AsyncThreadedExecuter(pm, pl, cn, world, replan_dt=replan_dt, control_dt=control_dt)
    )

    return executer


def run(source_run=True):
    executer = get_executer(config_path="configs/c20_plan.yaml", source_run=source_run)
    app = VisualizerApp(executer, code_reload_function=get_executer)
    app.mainloop()


if __name__ == "__main__":
    source_run = True

    import platform
    import os

    if platform.system() == "Linux":
        # Linux-specific code
        os.environ["TK_WINDOWS_FORCE_OPENGL"] = "1"
    else:

        import ctypes

        try:  # >= win 8.
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except:  # win 8.0 or less
            ctypes.windll.user32.SetProcessDPIAware()
        os.environ["TK_WINDOWS_FORCE_OPENGL"] = "1"

    run(source_run)
