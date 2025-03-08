import yaml
import logging
import json
import os
from pathlib import Path

import sys
import pkg_resources
import logging
import importlib

import c10_perceive.c11_perception_model 
import c10_perceive.c12_state 
import c20_plan.c21_base_planner
import c20_plan.c22_sampling_planner
import c20_plan.c23_lattice
import c20_plan.c24_trajectory
import c30_control.c31_base_controller
import c30_control.c32_pid_controller
import c40_execute.c41_executer
import c40_execute.c42_async_executer 
import c40_execute.c43_async_threaded_executer
import c40_execute.c44_basic_sim

import logging
log = logging.getLogger(__name__)

class Config:
    pass

def load_config(config_path, source_run=True):
    if os.path.isabs(config_path):
        raise ValueError("config_path should be relative to the project directory")

    # pkg_config = pkg_resources.resource_filename("AVLite", "config.yaml")
    if source_run:
        current_file_name = os.path.realpath(__file__) if sys.argv[0] == "" else sys.argv[0]
    else:
        current_file_name = pkg_config if sys.argv[0] == "" else sys.argv[0]

    log.info(f"current_file_name: {current_file_name}")
    if source_run:
        project_dir = Path(current_file_name).parent.parent
    else:
        project_dir = Path(current_file_name).parent.parent / "share/race_plan_control"

    config_file = project_dir / config_path

    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)
        path_to_track = project_dir / config_data["path_to_track"]
    # loading race trajectory data
    with open(path_to_track, "r") as f:
        track_data = json.load(f)
        reference_path = [point[:2] for point in track_data["ReferenceLine"]]  # ignoring z
        reference_velocity = track_data["ReferenceSpeed"]
        ref_left_boundary_d = track_data["LeftBound"]
        ref_right_boundary_d = track_data["RightBound"]
    logging.info(f"Track data loaded from {path_to_track}")
    return reference_path, reference_velocity, ref_left_boundary_d, ref_right_boundary_d

def reload_lib():
    log.info("Reloading imports...")
    importlib.reload(c10_perceive.c11_perception_model)
    importlib.reload(c10_perceive.c12_state)
    importlib.reload(c20_plan.c21_base_planner)
    importlib.reload(c20_plan.c22_sampling_planner)
    importlib.reload(c20_plan.c23_lattice)
    importlib.reload(c20_plan.c24_trajectory)
    importlib.reload(c30_control.c31_base_controller)
    importlib.reload(c30_control.c32_pid_controller)
    importlib.reload(c40_execute.c41_executer)
    importlib.reload(c40_execute.c42_async_executer)
    importlib.reload(c40_execute.c43_async_threaded_executer)
    importlib.reload(c40_execute.c44_basic_sim)

