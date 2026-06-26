from pydantic import Field

from avlite.c60_common.c68_settings_schema import SettingsSchema


class PluginSettingsSchema(SettingsSchema):
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


# Settings singleton; filepath is assigned by the plugin loader from the directory name.
PluginSettings = PluginSettingsSchema()
