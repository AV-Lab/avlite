from dataclasses import dataclass

@dataclass
class PerceptionSettings:
    exclude = ["exclude"]
    filepath: str = "configs/c10_perception.yaml"

    # c11 perception model
    c11_state_default_heading = 0
    c11_max_agents: int = 12
    c11_prediction_grid_size: int = 100

    # c12 perception pipeline sub-strategies (empty string = ground truth / skip that stage)
    c12_detection_strategy: str = ""
    c12_tracking_strategy: str = ""
    c12_prediction_strategy: str = ""

    # c15 prediction / tracking / detection
    c15_prediction_horizon: float = 2.0
    c15_tracking_dt: float = 0.1
    c15_tracking_process_noise: float = 1.0
    c15_tracking_measurement_noise: float = 0.5
    c15_tracking_init_velocity_var: float = 25.0
    c15_tracking_gate_distance: float = 4.0
    c15_tracking_max_missed: int = 12
    c15_tracking_min_speed: float = 0.5
    c15_detection_z_min: float = -1.5
    c15_detection_z_max: float = 0.5
    c15_detection_delta_min: float = 1.0
    c15_detection_delta_max: float = 6.0
    c15_detection_mu: float = 0.5
    c15_detection_min_length: float = 0.5
    c15_detection_min_width: float = 0.5
    c15_detection_default_length: float = 4.5
    c15_detection_default_width: float = 2.0

    # c16 lidar localization (scan-to-map ICP)
    c16_localization_lidar_z_min: float = -1.5
    c16_localization_lidar_z_max: float = 2.0
    c16_localization_icp_max_iterations: int = 30
    c16_localization_icp_tolerance: float = 1e-4
    c16_localization_icp_max_correspondence_dist: float = 5.0
    c16_localization_map_subsample: int = 1
