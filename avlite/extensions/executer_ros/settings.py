class ExtensionSettings:
    """Settings for ROS executor with Autoware message topics."""
    exclude = ["exclude", "filepath"]  # attributes to exclude from saving/loading
    filepath: str = "configs/ext_ros_executer.yaml"

    # Whether to use Autoware messages (True) or JSON strings (False)
    # When False, all messages are published/subscribed as std_msgs/String with JSON content
    use_autoware_msgs: bool = True

    # Autoware localization topic (subscribes to VehicleKinematicState or String)
    localization_topic: str = "/localization/kinematic_state"
    
    # Autoware perception topic (subscribes to BoundingBoxArray or String)
    perception_topic: str = "/perception/object_recognition/tracking/objects"
    
    # Autoware planning topic (subscribes to Trajectory or String from external planner)
    trajectory_topic: str = "/planning/scenario_planning/trajectory"
    
    # Autoware control topic (subscribes to VehicleControlCommand or String from external controller)
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
    
    # Timing settings (dt = delta time between ticks)
    sim_dt: float = 0.02       # Simulation step dt (50 Hz default)
    perception_dt: float = 0.1  # Perception publish rate (10 Hz default)
    replan_dt: float = 0.1      # Planning rate (10 Hz default)
    control_dt: float = 0.02    # Control rate (50 Hz default)


