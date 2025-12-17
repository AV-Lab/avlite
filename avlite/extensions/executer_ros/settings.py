class ExtensionSettings:
    """Settings for ROS executor with Autoware message topics."""
    exclude = ["exclude", "filepath"]  # attributes to exclude from saving/loading
    filepath: str = "configs/ext_ros_executer.yaml"

    # Autoware localization topic (subscribes to VehicleKinematicState)
    localization_topic: str = "/localization/kinematic_state"
    
    # Autoware perception topic (subscribes to TrackedObjects)
    perception_topic: str = "/perception/object_recognition/tracking/objects"
    
    # Autoware planning topic (subscribes to Trajectory from external planner)
    trajectory_topic: str = "/planning/scenario_planning/trajectory"
    
    # Autoware control topic (subscribes to AckermannControlCommand from external controller)
    control_cmd_topic: str = "/control/command/control_cmd"
    
    # Output topic for AVLite-computed trajectory (if running internal planner)
    trajectory_out_topic: str = "/avlite/planning/trajectory"
    
    # Output topic for AVLite-computed control (if running internal controller)
    control_out_topic: str = "/avlite/control/control_cmd"
    
    # Frame IDs
    map_frame: str = "map"
    base_frame: str = "base_link"
    
    # Collection frequency for syncing ROS data to visualizer
    collection_hz: float = 20.0


