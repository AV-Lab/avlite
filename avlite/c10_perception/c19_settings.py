from pydantic import Field

from avlite.c60_common.c68_settings_schema import SettingsSchema


class PerceptionSettingsSchema(SettingsSchema):
    c11_state_default_heading: int = Field(default=0, description="Default heading (rad) for new agent states.")
    c11_max_agents: int = Field(default=12, description="Maximum number of agents tracked in the perception model.")
    c11_prediction_grid_size: int = Field(default=100, description="Grid resolution for occupancy-flow prediction output.")

    c12_detection_strategy: str = Field(default="", description="Detection sub-strategy class name; empty uses ground truth.")
    c12_tracking_strategy: str = Field(default="", description="Tracking sub-strategy class name; empty uses ground truth.")
    c12_prediction_strategy: str = Field(default="", description="Prediction sub-strategy class name; empty disables prediction.")

    c15_prediction_horizon: float = Field(default=2.0, description="Prediction horizon in seconds.")
    c15_tracking_dt: float = Field(default=0.1, description="Tracker prediction step interval (seconds).")
    c15_tracking_process_noise: float = Field(default=1.0, description="Kalman process noise scale.")
    c15_tracking_measurement_noise: float = Field(default=0.5, description="Kalman measurement noise scale.")
    c15_tracking_init_velocity_var: float = Field(default=25.0, description="Initial velocity variance for new tracks.")
    c15_tracking_gate_distance: float = Field(default=4.0, description="Max association distance (m) for track updates.")
    c15_tracking_max_missed: int = Field(default=12, description="Max consecutive missed detections before dropping a track.")
    c15_tracking_min_speed: float = Field(default=0.5, description="Minimum speed (m/s) to treat an agent as moving.")
    c15_detection_z_min: float = Field(default=-1.5, description="LiDAR detection z-band minimum (m).")
    c15_detection_z_max: float = Field(default=0.5, description="LiDAR detection z-band maximum (m).")
    c15_detection_delta_min: float = Field(default=1.0, description="Min gap between LiDAR clusters (m).")
    c15_detection_delta_max: float = Field(default=6.0, description="Max gap between LiDAR clusters (m).")
    c15_detection_mu: float = Field(default=0.5, description="BEV detection merge threshold parameter.")
    c15_detection_min_length: float = Field(default=0.5, description="Minimum detected box length (m).")
    c15_detection_min_width: float = Field(default=0.5, description="Minimum detected box width (m).")
    c15_detection_default_length: float = Field(default=4.5, description="Default box length when shape unknown (m).")
    c15_detection_default_width: float = Field(default=2.0, description="Default box width when shape unknown (m).")

    c16_localization_lidar_z_min: float = Field(default=-1.5, description="ICP localization z-band minimum (m).")
    c16_localization_lidar_z_max: float = Field(default=2.0, description="ICP localization z-band maximum (m).")
    c16_localization_icp_max_iterations: int = Field(default=30, description="Max ICP iterations per scan.")
    c16_localization_icp_tolerance: float = Field(default=1e-4, description="ICP convergence tolerance.")
    c16_localization_icp_max_correspondence_dist: float = Field(default=5.0, description="Max ICP correspondence distance (m).")
    c16_localization_map_subsample: int = Field(default=1, description="Subsample factor for reference map points.")


class PerceptionSettings:
    schema = PerceptionSettingsSchema
    exclude = ["exclude", "filepath", "schema"]
    filepath: str = "configs/c10_perception.yaml"

    c11_state_default_heading = 0
    c11_max_agents: int = 12
    c11_prediction_grid_size: int = 100

    c12_detection_strategy: str = ""
    c12_tracking_strategy: str = ""
    c12_prediction_strategy: str = ""

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

    c16_localization_lidar_z_min: float = -1.5
    c16_localization_lidar_z_max: float = 2.0
    c16_localization_icp_max_iterations: int = 30
    c16_localization_icp_tolerance: float = 1e-4
    c16_localization_icp_max_correspondence_dist: float = 5.0
    c16_localization_map_subsample: int = 1
