from execute.visualizer import PlotApp
from plan.planner import Planner
from control.controller import PIDController
import yaml
from execute import executer
from execute.executer_sim import SimpleSim
import numpy as np
import logging
import json
import os

def main():

    # loading config file
    file = "config.yaml"
    config_file = os.path.realpath(file)
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
        path_to_track = config_data["path_to_track"]
    
    # loading race trajectory data
    with open(path_to_track, 'r') as f:
        track_data = json.load(f)
        reference_path = [point[:2] for point in track_data["ReferenceLine"]] # ignoring z
        ref_left_boundary_d = track_data["LeftBound"]
        ref_right_boundary_d = track_data["RightBound"]

    logging.info(f"Track data loaded from {path_to_track}")
    
    state = executer.VehicleState(x=reference_path[0][0],y=reference_path[0][1], speed=30, theta=-np.pi/4)

    pl = Planner(referance_path=reference_path, ref_left_boundary_d=ref_left_boundary_d, ref_right_boundary_d=ref_right_boundary_d)
    cn = PIDController()
    sim = SimpleSim(state, pl, cn)

    app = PlotApp(pl,cn,sim)
    app.mainloop()  


if __name__ == "__main__":
    main()  
