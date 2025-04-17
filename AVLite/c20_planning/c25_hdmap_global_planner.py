import math
import xml.etree.ElementTree as ET
import numpy as np
import networkx as nx
from scipy.interpolate import CubicSpline
import logging

from c20_planning.c22_base_global_planner import BaseGlobalPlanner
from c20_planning.c21_planning_model import GlobalPlan

log = logging.getLogger(__name__)

class GlobalHDMapPlanner(BaseGlobalPlanner):
    """
    A global planner that:
      1. Parses a simplified OpenDRIVE (.xodr) file.
      2. Builds a lane-center graph by sampling the parametric road geometry.
      3. Uses A* to find a path from start to goal.
      4. Returns a smooth path and a simple velocity profile.
    """

    def __init__(self, xodr_file, sampling_resolution=1.0):
        """
        :param xodr_file: path to the OpenDRIVE HD map (.xodr).
        :param sampling_resolution: distance (meters) between samples when converting arcs/lines to discrete points.
        """
        super().__init__()
        self.xodr_file = xodr_file
        self.sampling_resolution = sampling_resolution
        try:
            tree = ET.parse(self.xodr_file)
            self.xodr_root = tree.getroot()
        except ET.ParseError as e:
            log.error(f"Error parsing OpenDRIVE file: {e}")



    def plan(self, start: tuple[float, float], goal: tuple[float, float]) -> None:
        pass

