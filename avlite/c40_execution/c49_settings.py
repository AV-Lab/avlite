from pydantic import Field

from avlite.c60_common.c68_settings_schema import SettingsSchema


class ExecutionSettingsSchema(SettingsSchema):
    c40_executer_type: str = Field(default="SyncExecuter", description="Executer class name.")
    c40_bridge: str = Field(default="BasicSim", description="World bridge class name (e.g. BasicSim, CarlaBridge).")
    c40_perception: str = Field(default="", description="Perception strategy class; empty uses UI/registry default.")
    c40_localization: str = Field(default="", description="Localization strategy class; empty uses default.")
    c40_mapping: str = Field(default="", description="Mapping strategy class; empty uses default.")
    c40_global_planner: str = Field(default="GlobalCenterlineRacePlanner", description="Global planner class name.")
    c40_local_planner: str = Field(default="GreedyLatticePlanner", description="Local planner class name.")
    c40_controller: str = Field(default="StanleyController", description="Controller class name.")
    c40_perception_dt: float = Field(default=0.5, description="Perception tick period (seconds).", ge=0.001)
    c40_localization_dt: float = Field(default=0.1, description="Localization tick period (seconds).", ge=0.001)
    c40_replan_dt: float = Field(default=0.5, description="Replanning period (seconds).", ge=0.001)
    c40_control_dt: float = Field(default=0.05, description="Control loop period (seconds).", ge=0.001)
    c40_sim_dt: float = Field(default=0.01, description="Simulation step period (seconds).", ge=0.001)
    c40_global_trajectory: str = Field(default="data/yas_marina_real_race_line_mue_0_5_3_m_margin.json", description="Default global plan JSON path.")
    c40_hd_map: str = Field(default="data/san_campus.xodr", description="HD map OpenDRIVE file path.")
    c40_community_plugins: dict[str, str] = Field(default_factory=dict, description="Community plugin name to directory map.")
    c40_default_plugins: list[str] = Field(default_factory=list, description="Built-in plugins to load on startup.")
    c40_async_combined_perception_planning: bool = Field(default=True, description="Run perception and planning concurrently.")
    c40_log_level: str = Field(default="INFO", description="Python logging level.")
    c40_log_to_file: bool = Field(default=False, description="Write logs to file.")

    c41_provide_ground_truth: bool = Field(default=False, description="Bridge exposes ground-truth perception.")
    c41_provide_rgb: bool = Field(default=False, description="Bridge exposes RGB camera data.")
    c41_provide_lidar: bool = Field(default=False, description="Bridge exposes LiDAR data.")
    c41_provide_depth: bool = Field(default=False, description="Bridge exposes depth camera data.")

    c43_race_boundary_map: str = Field(default="data/race_boundary_yas_marina.json", description="Race boundary JSON for centerline planner.")
    c43_race_boundary_margin: float = Field(default=0.0, description="Inflation margin applied to race boundary (m).")

    c46_default_trajectory: str = Field(default="data/yas_marina_real_race_line_mue_0_5_3_m_margin.json", description="BasicSim default trajectory path.")
    c46_npc_speed_factor: float = Field(default=0.8, description="NPC speed as fraction of plan speed.")
    c46_npc_control: bool = Field(default=True, description="Enable NPC vehicle controllers in BasicSim.")
    c46_lidar_boundary_file: str = Field(default="data/yasmarina.track.json", description="Track boundary file for simulated LiDAR.")
    c46_lidar_range: float = Field(default=50.0, description="Simulated LiDAR max range (m).")
    c46_lidar_num_beams: int = Field(default=360, description="Number of simulated LiDAR beams.")
    c46_lidar_fov_deg: float = Field(default=360.0, description="Simulated LiDAR field of view (degrees).")


class ExecutionSettings:
    schema = ExecutionSettingsSchema
    exclude = ["exclude", "filepath", "schema"]
    filepath: str = "configs/c40_execution.yaml"

    c40_executer_type = "SyncExecuter"
    c40_bridge = "BasicSim"
    c40_perception = ""
    c40_localization = ""
    c40_mapping = ""
    c40_global_planner = "GlobalCenterlineRacePlanner"
    c40_local_planner = "GreedyLatticePlanner"
    c40_controller = "StanleyController"
    c40_perception_dt = 0.5
    c40_localization_dt = 0.1
    c40_replan_dt = 0.5
    c40_control_dt = 0.05
    c40_sim_dt = 0.01
    c40_global_trajectory = "data/yas_marina_real_race_line_mue_0_5_3_m_margin.json"
    c40_hd_map = "data/san_campus.xodr"
    c40_community_plugins: dict[str, str] = {}
    c40_default_plugins: list[str] = []
    c40_async_combined_perception_planning: bool = True
    c40_log_level = "INFO"
    c40_log_to_file = False

    c41_provide_ground_truth = False
    c41_provide_rgb = False
    c41_provide_lidar = False
    c41_provide_depth = False

    c43_race_boundary_map: str = "data/race_boundary_yas_marina.json"
    c43_race_boundary_margin: float = 0.0

    c46_default_trajectory = "data/yas_marina_real_race_line_mue_0_5_3_m_margin.json"
    c46_npc_speed_factor = 0.8
    c46_npc_control = True
    c46_lidar_boundary_file = "data/yasmarina.track.json"
    c46_lidar_range = 50.0
    c46_lidar_num_beams = 360
    c46_lidar_fov_deg = 360.0
