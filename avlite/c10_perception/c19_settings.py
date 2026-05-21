from dataclasses import dataclass
import numpy as np

@dataclass
class PerceptionSettings:
    exclude = ["exclude"]
    filepath: str="configs/c10_perception.yaml"

    # State
    state_default_heading = 0 #- np.pi / 4 

    # Perception Model
    perception_model_max_agents: int = 12
    perception_model_prediction_grid_size: int = 100  # Size of the occupancy grid -> 100x100


    # hdmap
    hdmap_sampling_resolution: float = 0.1  # Sampling resolution for the HDMap

    # Pipeline sub-strategies (empty string = ground truth / skip that stage)
    detection_strategy: str = ""
    tracking_strategy: str = ""
    prediction_strategy: str = ""

    # Prediction horizon in seconds used by PredictionStrategy implementations
    prediction_horizon: float = 2.0

