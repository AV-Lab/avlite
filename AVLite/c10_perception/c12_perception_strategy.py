from c10_perception.c11_perception_model import State, AgentState, EgoState, PerceptionModel
from c20_planning.c28_trajectory import Trajectory
import numpy as np
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


class BasePerception:
    pm: PerceptionModel

    def __init__(self):
        pass

    def calibrate(self):
        pass
        
    def perceive(self, rgb_img = None, depth_img = None, lidar_data = None):
        pass

    def predict(self):
        pass

    def reset(self):
        pass
