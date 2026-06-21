from pydantic import Field

from avlite.c60_common.c68_settings_schema import SettingsSchema


class ExtensionSettingsSchema(SettingsSchema):
    use_autoware_msgs: bool = Field(default=True, description="Use Autoware message types instead of JSON strings.")
    localization_topic: str = Field(default="/localization/kinematic_state", description="Ego state input topic.")
    perception_topic: str = Field(
        default="/perception/object_recognition/tracking/objects",
        description="Tracked objects input topic.",
    )
    control_out_topic: str = Field(default="/control/command/control_cmd", description="Control command output topic.")
    map_frame: str = Field(default="map", description="ROS map frame id.")
    base_frame: str = Field(default="base_link", description="ROS base link frame id.")
    lidar_topic: str = Field(default="/sensing/lidar/concatenated/pointcloud", description="LiDAR topic; empty disables.")
    rgb_topic: str = Field(default="/sensing/camera/traffic_light/image_raw", description="RGB camera topic; empty disables.")


class ExtensionSettings:
    schema = ExtensionSettingsSchema
    exclude = ["exclude", "filepath", "schema"]
    filepath: str = "configs/ext_ROS2_worldbridge.yaml"

    use_autoware_msgs: bool = True
    localization_topic: str = "/localization/kinematic_state"
    perception_topic: str = "/perception/object_recognition/tracking/objects"
    control_out_topic: str = "/control/command/control_cmd"
    map_frame: str = "map"
    base_frame: str = "base_link"
    lidar_topic: str = "/sensing/lidar/concatenated/pointcloud"
    rgb_topic: str = "/sensing/camera/traffic_light/image_raw"
