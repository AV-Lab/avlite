from c50_visualize.c51_tk_visualizer import VisualizerApp
import c10_perceive.c11_environment 
import c10_perceive.c12_state 
import c20_plan.c22_sampling_planner
import c20_plan.c23_lattice
import c20_plan.c24_trajectory
import c30_control.c32_pid
import c40_execute.c41_executer
import c10_perceive.c12_state
import c40_execute.c42_basic_sim_executer
from race_plan_control.utils import load_config, reload_lib

import numpy as np

import logging
log = logging.getLogger(__name__)


def get_executer(config_path="config.yaml", source_run=True):

    reference_path, ref_left_boundary_d, ref_right_boundary_d = load_config(config_path=config_path, source_run=source_run)

    reload_lib()
    Enivronment = c10_perceive.c11_environment.Environment
    RNDPlanner = c20_plan.c22_sampling_planner.RNDPlanner
    PIDController = c30_control.c32_pid.PIDController
    BasicSimExecuter = c40_execute.c42_basic_sim_executer.BasicSimExecuter
    EgoState = c10_perceive.c12_state.EgoState

    ego_state = EgoState(x=reference_path[0][0], y=reference_path[0][1], speed=30, theta=-np.pi / 4)
    env = Enivronment(ego_state) 
    pl = RNDPlanner(
        global_path=reference_path,
        ref_left_boundary_d=ref_left_boundary_d,
        ref_right_boundary_d=ref_right_boundary_d,
        env=env
    )
    cn = PIDController()
    executer = BasicSimExecuter(env, ego_state, pl, cn)

    return executer


def run(source_run=True):
    executer = get_executer(config_path="config.yaml", source_run=source_run)
    app = VisualizerApp(executer, code_reload_function=get_executer)
    app.mainloop()


if __name__ == "__main__":
    source_run = True
    run(source_run)
