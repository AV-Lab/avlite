from x50_visualize.x51_tk_visualizer import VisualizerApp
import c20_plan.c21_planner
import c20_plan.c22_sampling_planner
import c20_plan.c23_lattice
import c20_plan.c24_trajectory
import c30_control.c32_pid
import x40_execute.x41_executer
import c10_perceive.c12_state
from race_plan_control.utils import load_config

import numpy as np
import logging
import logging
import importlib

log = logging.getLogger(__name__)


def get_executer(config_path="config.yaml", source_run=True):
    log.info("Reloading imports...")
    importlib.reload(c20_plan.c21_planner)
    importlib.reload(c20_plan.c22_sampling_planner)
    importlib.reload(c20_plan.c23_lattice)
    importlib.reload(c20_plan.c24_trajectory)
    importlib.reload(c30_control.c32_pid)
    importlib.reload(x40_execute.x41_executer)
    importlib.reload(c10_perceive.c12_state)

    reference_path, ref_left_boundary_d, ref_right_boundary_d = load_config(config_path=config_path, source_run=source_run)

    RNDPlanner = c20_plan.c22_sampling_planner.RNDPlanner
    PIDController = c30_control.c32_pid.PIDController
    SimpleSim = x40_execute.x41_executer.Executer
    VehicleState = c10_perceive.c12_state.VehicleState

    state = VehicleState(x=reference_path[0][0], y=reference_path[0][1], speed=30, theta=-np.pi / 4)
    pl = RNDPlanner(
        reference_path=reference_path,
        ref_left_boundary_d=ref_left_boundary_d,
        ref_right_boundary_d=ref_right_boundary_d,
    )
    cn = PIDController()
    executer = SimpleSim(state, pl, cn)

    return executer


def run(source_run=True):
    executer = get_executer(config_path="config.yaml", source_run=source_run)
    app = VisualizerApp(executer, code_reload_function=get_executer)
    app.mainloop()


if __name__ == "__main__":
    source_run = True
    run(source_run)
