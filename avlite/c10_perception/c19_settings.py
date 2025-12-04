from dataclasses import dataclass
import numpy as np

@dataclass
class PerceptionSettings:
    exclude = ["exclude"]
    filepath: str="configs/c10_perception.yaml"

    # State
    state_default_heading = 0 #- np.pi / 4 

    # Ego
    ego_max_valocity: float = 30
    ego_max_acceleration: float = 10    
    ego_min_acceleration: float = -20
    ego_max_steering: float = 0.7  # in radians
    ego_min_steering: float = -0.7
    ego_distance_front_axle: float = 2.5  # Distance from center of mass to front axle


    # Perception Model
    perception_model_max_agents: int = 12
    perception_model_prediction_grid_size: int = 100  # Size of the occupancy grid -> 100x100


    # hdmap
    hdmap_sampling_resolution: float = 0.1  # Sampling resolution for the HDMap

