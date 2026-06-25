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


class PluginSettings:
    schema = PluginSettingsSchema
    exclude = ["exclude", "filepath", "schema"]
    filepath: str = "configs/plugin_ros_executer.yaml"

    use_autoware_msgs: bool = True
    localization_topic: str = "/localization/kinematic_state"
    perception_topic: str = "/perception/object_recognition/tracking/objects"
    trajectory_topic: str = "/planning/scenario_planning/trajectory"
    control_cmd_topic: str = "/control/command/control_cmd"
    trajectory_out_topic: str = "/avlite/planning/trajectory"
    control_out_topic: str = "/avlite/control/control_cmd"
    map_frame: str = "map"
    base_frame: str = "base_link"
    collection_hz: float = 20.0
    sim_dt: float = 0.02
    perception_dt: float = 0.1
    replan_dt: float = 0.1
    control_dt: float = 0.02
