from race_plan_control.visualize.tk_visualizer import VisualizerApp

import yaml
import numpy as np
import logging
import json
import os
from pathlib import Path


import sys 
import pkg_resources
import logging
import importlib 

log = logging.getLogger(__name__)

is_running_from_source = False



def load_config(config_path):
    if os.path.isabs(config_path):
        raise ValueError("config_path should be relative to the project directory")
    
    pkg_config = pkg_resources.resource_filename('race_plan_control', 'config.yaml') 
    if is_running_from_source:
        current_file_name = os.path.realpath(__file__) if sys.argv[0] == '' else sys.argv[0]
    else:
        current_file_name = pkg_config if sys.argv[0] == '' else sys.argv[0]

    print(f"current_file_name: {current_file_name}")
    if is_running_from_source:
        project_dir = Path(current_file_name).parent.parent 
    else:
        project_dir = Path(current_file_name).parent.parent/"share/race_plan_control"
    
    config_file = project_dir / config_path

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
        path_to_track = project_dir / config_data["path_to_track"]
    # loading race trajectory data
    with open(path_to_track, 'r') as f:
        track_data = json.load(f)
        reference_path = [point[:2] for point in track_data["ReferenceLine"]] # ignoring z
        ref_left_boundary_d = track_data["LeftBound"]
        ref_right_boundary_d = track_data["RightBound"]
    logging.info(f"Track data loaded from {path_to_track}")

    return reference_path, ref_left_boundary_d, ref_right_boundary_d

def get_executer():
    global reference_path, ref_left_boundary_d, ref_right_boundary_d
    log.info("Reloading imports...")
    # reload imports
    import race_plan_control.plan.planner
    import race_plan_control.plan.sampling_planner
    import race_plan_control.control.pid
    import race_plan_control.execute.executer_sim
    import race_plan_control.perceive.vehicle_state
    importlib.reload(race_plan_control.plan.planner)
    importlib.reload(race_plan_control.plan.sampling_planner)
    importlib.reload(race_plan_control.control.pid)
    importlib.reload(race_plan_control.execute.executer_sim)
    importlib.reload(race_plan_control.perceive.vehicle_state)
    RNDPlanner = race_plan_control.plan.sampling_planner.RNDPlanner
    PIDController = race_plan_control.control.pid.PIDController
    SimpleSim = race_plan_control.execute.executer_sim.SimpleSim
    VehicleState = race_plan_control.perceive.vehicle_state.VehicleState

    state = VehicleState(x=reference_path[0][0],y=reference_path[0][1], speed=30, theta=-np.pi/4)
    pl = RNDPlanner(reference_path=reference_path, ref_left_boundary_d=ref_left_boundary_d, ref_right_boundary_d=ref_right_boundary_d)
    cn = PIDController()
    executer = SimpleSim(state, pl, cn)

    return executer


def run(config_path="config.yaml", headless=False):
    global reference_path, ref_left_boundary_d, ref_right_boundary_d
    reference_path, ref_left_boundary_d, ref_right_boundary_d = load_config(config_path = config_path)
    executer = get_executer()
    app = VisualizerApp(executer, code_reload_function=get_executer)
    app.mainloop()  


if __name__ == "__main__":
    is_running_from_source = True
    run()

