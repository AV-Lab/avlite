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

import c20_plan.c21_planner
import c20_plan.c22_sampling_planner
import c20_plan.c23_lattice
import c20_plan.c24_trajectory
import c30_control.c32_pid
import x40_execute.x41_executer
import c10_perceive.c12_state

import logging
log = logging.getLogger(__name__)

def load_config(config_path, source_run=True):
    if os.path.isabs(config_path):
        raise ValueError("config_path should be relative to the project directory")

    pkg_config = pkg_resources.resource_filename("race_plan_control", "config.yaml")
    if source_run:
        current_file_name = os.path.realpath(__file__) if sys.argv[0] == "" else sys.argv[0]
    else:
        current_file_name = pkg_config if sys.argv[0] == "" else sys.argv[0]

    print(f"current_file_name: {current_file_name}")
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
        ref_left_boundary_d = track_data["LeftBound"]
        ref_right_boundary_d = track_data["RightBound"]
    logging.info(f"Track data loaded from {path_to_track}")
    return reference_path, ref_left_boundary_d, ref_right_boundary_d

def reload_lib():
    log.info("Reloading imports...")
    importlib.reload(c20_plan.c21_planner)
    importlib.reload(c20_plan.c22_sampling_planner)
    importlib.reload(c20_plan.c23_lattice)
    importlib.reload(c20_plan.c24_trajectory)
    importlib.reload(c30_control.c32_pid)
    importlib.reload(x40_execute.x41_executer)
    importlib.reload(c10_perceive.c12_state)
