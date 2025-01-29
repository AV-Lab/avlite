from c50_visualize.c51_visualizer_app import VisualizerApp
import c10_perceive.c11_perception_model
import c10_perceive.c12_state
import c20_plan.c22_sampling_planner
import c20_plan.c23_lattice
import c20_plan.c24_trajectory
import c30_control.c32_pid
import c40_execute.c41_executer
import c10_perceive.c12_state
import c40_execute.c41_executer
import c40_execute.c42_basic_sim
from utils import load_config, reload_lib

import numpy as np

import logging

log = logging.getLogger(__name__)


def get_executer(config_path="config.yaml", source_run=True):

    reference_path, ref_left_boundary_d, ref_right_boundary_d = load_config(
        config_path=config_path, source_run=source_run
    )

    reload_lib()
    PerceptionModel = c10_perceive.c11_perception_model.PerceptionModel
    RNDPlanner = c20_plan.c22_sampling_planner.RNDPlanner
    PIDController = c30_control.c32_pid.PIDController
    Executer = c40_execute.c41_executer.Executer
    BasicSim = c40_execute.c42_basic_sim.BasicSim
    EgoState = c10_perceive.c12_state.EgoState

    world = BasicSim()
    ego_state = EgoState(x=reference_path[0][0], y=reference_path[0][1], speed=30, theta=-np.pi / 4)
    pm = PerceptionModel(ego_state)
    pl = RNDPlanner(
        global_path=reference_path,
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
