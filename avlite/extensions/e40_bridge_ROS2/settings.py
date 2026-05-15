class ExtensionSettings:
    """Settings for the ROS2 World Bridge plugin."""
    exclude = ["exclude", "filepath"]  # attributes to exclude from saving/loading
    filepath: str = "configs/ext_ROS2_worldbridge.yaml"

    # Whether to use Autoware messages (True) or JSON strings over std_msgs/String (False)
    use_autoware_msgs: bool = True

    # Topic bridge receives ego state from (VehicleKinematicState or String)
    localization_topic: str = "/localization/kinematic_state"

    # Topic bridge receives tracked objects from (BoundingBoxArray or String)
    perception_topic: str = "/perception/object_recognition/tracking/objects"

    # Topic bridge publishes control commands to (VehicleControlCommand or String)
    control_out_topic: str = "/control/command/control_cmd"

    # ROS frame IDs
    map_frame: str = "map"
    base_frame: str = "base_link"

    # LiDAR point-cloud topic (sensor_msgs/PointCloud2). Empty string = disabled.
    lidar_topic: str = "/sensing/lidar/concatenated/pointcloud"

    # RGB camera topic (sensor_msgs/Image, encoding bgr8 or rgb8). Empty string = disabled.
    rgb_topic: str = "/sensing/camera/traffic_light/image_raw"
