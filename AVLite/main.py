from c50_visualize.c51_visualizer_app import VisualizerApp
from utils import load_config, reload_lib

import numpy as np

import logging

log = logging.getLogger(__name__)


def get_executer(config_path="config.yaml", source_run=True):

    reference_path, reference_velocity, ref_left_boundary_d, ref_right_boundary_d = load_config(
        config_path=config_path, source_run=source_run
    )

    reload_lib()
    from c10_perceive.c11_perception_model import PerceptionModel
    from c20_plan.c22_sampling_planner import RNDPlanner
    from c30_control.c32_pid_controller import PIDController
    from c40_execute.c41_executer import Executer
    from c40_execute.c42_basic_sim import BasicSim
    from c10_perceive.c12_state import EgoState

    world = BasicSim()
    ego_state = EgoState(x=reference_path[0][0], y=reference_path[0][1], speed=reference_velocity[0], theta=-np.pi / 4)
    pm = PerceptionModel(ego_state)
    pl = RNDPlanner(
        global_path=reference_path,
        global_velocity=reference_velocity,
        ref_left_boundary_d=ref_left_boundary_d,
        ref_right_boundary_d=ref_right_boundary_d,
        env=pm,
    )
    cn = PIDController()
    executer = Executer(pm, pl, cn, world)

    return executer


def run(source_run=True):
    executer = get_executer(config_path="config.yaml", source_run=source_run)
    app = VisualizerApp(executer, code_reload_function=get_executer)
    app.mainloop()


if __name__ == "__main__":
    source_run = True
    run(source_run)
