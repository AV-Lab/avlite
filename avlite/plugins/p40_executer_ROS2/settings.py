from pydantic import Field

from avlite.c60_common.c68_settings_schema import SettingsSchema


class PluginSettingsSchema(SettingsSchema):
    use_autoware_msgs: bool = Field(default=True, description="Use Autoware message types instead of JSON strings.")
    localization_topic: str = Field(default="/localization/kinematic_state", description="Localization input topic.")
    perception_topic: str = Field(
        default="/perception/object_recognition/tracking/objects",
        description="Perception input topic.",
    )
    trajectory_topic: str = Field(default="/planning/scenario_planning/trajectory", description="External trajectory input topic.")
    control_cmd_topic: str = Field(default="/control/command/control_cmd", description="External control input topic.")
    trajectory_out_topic: str = Field(default="/avlite/planning/trajectory", description="AVLite trajectory output topic.")
    control_out_topic: str = Field(default="/avlite/control/control_cmd", description="AVLite control output topic.")
    map_frame: str = Field(default="map", description="ROS map frame id.")
    base_frame: str = Field(default="base_link", description="ROS base link frame id.")
    collection_hz: float = Field(default=20.0, description="ROS data collection rate (Hz).")
    sim_dt: float = Field(default=0.02, description="Simulation step dt (seconds).")
    perception_dt: float = Field(default=0.1, description="Perception publish dt (seconds).")
    replan_dt: float = Field(default=0.1, description="Planning dt (seconds).")
    control_dt: float = Field(default=0.02, description="Control dt (seconds).")


# Settings singleton. The plugin loader derives and assigns the YAML filepath from
# the plugin directory name, so none is declared here.
PluginSettings = PluginSettingsSchema()
