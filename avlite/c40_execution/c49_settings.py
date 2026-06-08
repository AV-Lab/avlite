
class ExecutionSettings:
    exclude = ["exclude"]
    filepath: str = "configs/c40_execution.yaml"

    # c40 orchestration (factory, executer, UI)
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
    c40_community_plugins: dict[str, str] = {
        "delete_me": "/home/mkhonji/Dropbox/20-development/21-software-dev/21.2-AVlite/avlite-plugins/delete_me"
    }
    c40_default_extensions: list[str] = []
    c40_async_combined_perception_planning: bool = True
    c40_log_level = "INFO"
    c40_log_to_file = False

    # c41 execution model / bridge sensor flags
    c41_provide_ground_truth = False
    c41_provide_rgb = False
    c41_provide_lidar = False

    # c42 factory fallback global planner
    c42_race_boundary_map: str = "data/race_boundary_yas_marina.json"
    c42_race_boundary_margin: float = 0.0

    # c46 BasicSim bridge
    c46_default_trajectory = "data/yas_marina_real_race_line_mue_0_5_3_m_margin.json"
    c46_npc_speed_factor = 0.8
    c46_npc_control = True
    c46_lidar_boundary_file = "data/yasmarina.track.json"
    c46_lidar_range = 50.0
    c46_lidar_num_beams = 360
    c46_lidar_fov_deg = 360.0
