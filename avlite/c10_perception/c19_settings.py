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

    # KalmanTracker parameters (constant-velocity Kalman filter)
    tracking_dt: float = 0.1            # Time step between tracking updates [s]
    tracking_process_noise: float = 1.0     # Process (acceleration) noise std [m/s^2]
    tracking_measurement_noise: float = 0.5  # Measurement (position) noise std [m]
    tracking_init_velocity_var: float = 25.0  # Initial velocity variance [m2/s2]
    tracking_gate_distance: float = 4.0     # Max association distance det<->track [m]
    tracking_max_missed: int = 12           # Frames a track survives without a match
    tracking_min_speed: float = 0.5         # Speed below which box heading is kept [m/s]

    # FastBEVLidarDetection parameters
    detection_z_min: float = -1.5       # Z-band filter lower bound [m] (3D input only)
    detection_z_max: float = 0.5        # Z-band filter upper bound [m] (3D input only)
    detection_delta_min: float = 1.0    # Min accepted cluster bbox diagonal [m]
    detection_delta_max: float = 6.0    # Max accepted cluster bbox diagonal [m]
    detection_mu: float = 0.5           # Max consecutive-point gap before new cluster [m]
    detection_min_length: float = 0.5   # Min fitted rectangle length [m] (avoids degenerate boxes)
    detection_min_width: float = 0.5    # Min fitted rectangle width [m] (avoids degenerate boxes)
    detection_default_length: float = 4.5  # Fallback car length [m] when fitted shape is degenerate
    detection_default_width: float = 2.0   # Fallback car width [m] when fitted shape is degenerate

    # LidarLocalization parameters (scan-to-map ICP)
    localization_lidar_z_min: float = -1.5      # Z-band filter lower bound [m] (3D input only)
    localization_lidar_z_max: float = 2.0       # Z-band filter upper bound [m] (3D input only)
    localization_icp_max_iterations: int = 30   # Max ICP iterations per localize() call
    localization_icp_tolerance: float = 1e-4    # Convergence tol on mean correspondence shift [m]
    localization_icp_max_correspondence_dist: float = 5.0  # Reject correspondences beyond this [m]
    localization_map_subsample: int = 1         # Keep every k-th map point (1 = keep all)

